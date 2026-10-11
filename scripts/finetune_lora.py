"""
Smoke LoRA/QLoRA fine-tune for google/gemma-3-1b-it.

Designed for a small GPU (e.g. 4 GiB): 4-bit QLoRA, batch=1, short max_steps.
Example:
  python scripts/finetune_lora.py --lang bangla --max-steps 20
  python scripts/finetune_lora.py --lang english --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/gemma-3-1b-it"

BN_TRAIN_FILE = ROOT / "data" / "train_bangla.jsonl"
EN_TRAIN_FILE = ROOT / "data" / "train_english.jsonl"

LANG_CHOICES = ("bangla", "english", "bilingual")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("instruction", "response"):
                if key not in row or not str(row[key]).strip():
                    raise ValueError(f"Missing '{key}' on line {line_no} of {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def format_example(tokenizer, instruction: str, response: str) -> str:
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return f"User: {instruction}\nAssistant: {response}"


def build_dataset(lang: str, tokenizer) -> Dataset:
    if lang == "bangla":
        rows = load_jsonl(BN_TRAIN_FILE)
    elif lang == "english":
        rows = load_jsonl(EN_TRAIN_FILE)
    elif lang == "bilingual":
        bn_rows = load_jsonl(BN_TRAIN_FILE)
        en_rows = load_jsonl(EN_TRAIN_FILE)
        # Interleave Bangla and English rows for balanced bilingual exposure
        rows = []
        max_len = max(len(bn_rows), len(en_rows))
        for i in range(max_len):
            if i < len(bn_rows):
                rows.append(bn_rows[i])
            if i < len(en_rows):
                rows.append(en_rows[i])
        print(f"Loaded {len(rows)} bilingual examples ({len(bn_rows)} Bangla, {len(en_rows)} English)")
    else:
        raise ValueError(f"Unsupported lang: {lang}")

    texts = [
        format_example(tokenizer, r["instruction"], r["response"]) for r in rows
    ]
    print(f"Formatted {len(texts)} examples for training (lang={lang})")
    return Dataset.from_dict({"text": texts})


def pick_compute_dtype() -> tuple[torch.dtype, bool, bool]:
    """Return (compute_dtype, use_fp16, use_bf16) for QLoRA + AMP."""
    if not torch.cuda.is_available():
        return torch.float32, False, False
    # Ampere (3050 Ti) supports bf16; GradScaler+fp16 crashes if params are bf16.
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    return torch.float16, True, False


def load_model_and_tokenizer(use_4bit: bool, compute_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires CUDA.")
        print(f"Loading model in 4-bit (QLoRA), compute_dtype={compute_dtype} ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"Loading model in {compute_dtype} (LoRA) ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return tokenizer, model


def attach_lora(model, r: int = 8, compute_dtype: torch.dtype | None = None):
    # Broad target set works across Llama/Gemma-style modules.
    config = LoraConfig(
        r=r,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    if compute_dtype is not None:
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype != compute_dtype:
                param.data = param.data.to(compute_dtype)
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke LoRA fine-tune for Gemma 3 1B")
    p.add_argument("--lang", choices=LANG_CHOICES, default="bilingual", help="bangla, english, or bilingual")
    p.add_argument("--max-steps", type=int, default=20, help="Short run for smoke test")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable QLoRA (needs more VRAM).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: models/gemma3-1b-<lang>-lora-smoke",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    out_dir = args.output_dir or (
        ROOT / "models" / f"gemma3-1b-{args.lang}-lora-smoke"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_4bit = not args.no_4bit
    compute_dtype, use_fp16, use_bf16 = pick_compute_dtype()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"({props.total_memory / 1024**3:.1f} GiB)"
        )
        print(f"AMP: fp16={use_fp16} bf16={use_bf16} compute={compute_dtype}")
    else:
        print("WARNING: CUDA not available — training will be very slow on CPU.")
        use_4bit = False

    tokenizer, model = load_model_and_tokenizer(
        use_4bit=use_4bit, compute_dtype=compute_dtype
    )
    model = attach_lora(model, r=args.lora_r, compute_dtype=compute_dtype)
    model.print_trainable_parameters()

    dataset = build_dataset(args.lang, tokenizer)

    # TRL 1.10 + transformers 5.x: text-field / max length live on SFTConfig;
    # SFTTrainer takes processing_class (not tokenizer / dataset_text_field kwargs).
    training_args = SFTConfig(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        save_total_limit=1,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.max_steps // 10),
        report_to=[],
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    print(
        f"\nSmoke train: lang={args.lang} steps={args.max_steps} "
        f"4bit={use_4bit} -> {out_dir}"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved LoRA adapter + tokenizer to {out_dir}")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM: {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
