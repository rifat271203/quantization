"""
QLoRA fine-tuning and benchmarking script for google/gemma-3-1b-it.

Features:
- Loads base model in 4-bit NF4 (bitsandbytes) with bfloat16 compute dtype.
- Pinned to exact HF commit hash: dcc83ea841ab6100d6b47a070329e1ba4cf78752.
- LoRA config: r=16, alpha=32, dropout=0.05 targeting attention + MLP projection layers:
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
- Gradient checkpointing enabled.
- max_seq_length=1024, batch_size=1, gradient_accumulation_steps=8, lr=2e-4, bf16=True.
- Benchmark mode: when --max_steps > 0, measures peak VRAM, throughput (sec/step, tokens/sec),
  extrapolates 3-epoch wall-clock time, and skips saving checkpoint.
- Full run mode: runs for specified --epochs (default: 3) and saves LoRA adapter + manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-3-1b-it"
PINNED_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA training for Gemma-3-1b-it")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to training jsonl file (e.g. data/train_bilingual.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/gemma3-1b-bilingual-lora",
        help="Directory to save fine-tuned LoRA adapter",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0, runs in benchmark mode for max_steps and skips saving adapter",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of full training epochs (ignored if --max_steps > 0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank r (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Max sequence length in tokens (default: 1024)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per device train batch size (default: 1)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Logging steps interval (default: 1)",
    )
    return parser.parse_args()


def load_dataset_from_jsonl(path_str: str, tokenizer, max_seq_length: int) -> tuple[Dataset, int, int]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    texts = []
    exceed_count = 0
    total_tokens = 0

    for r in rows:
        messages = r.get("messages")
        if not messages:
            # Fallback if raw instruction/response exists
            messages = [
                {"role": "user", "content": r.get("instruction", "")},
                {"role": "assistant", "content": r.get("response", "")},
            ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tok_len = len(tokenizer.encode(chat_text, add_special_tokens=False))
        total_tokens += min(tok_len, max_seq_length)
        if tok_len > max_seq_length:
            exceed_count += 1
        texts.append(chat_text)

    dataset = Dataset.from_dict({"text": texts})
    return dataset, len(rows), exceed_count


class StepMetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.step_times: list[float] = []
        self._last_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        elapsed = now - self._last_time
        self.step_times.append(elapsed)
        self._last_time = now


def main() -> int:
    configure_stdout()
    args = parse_args()

    print("=" * 70)
    print("QLoRA Fine-Tuning & Benchmark for google/gemma-3-1b-it")
    print("=" * 70)
    print(f"HF Model ID:       {MODEL_ID}")
    print(f"Pinned Revision:   {PINNED_REVISION}")
    print(f"Dataset:           {args.data_file}")
    print(f"Mode:              {'BENCHMARK (max_steps=' + str(args.max_steps) + ')' if args.max_steps > 0 else 'FULL RUN (' + str(args.epochs) + ' epochs)'}")
    print(f"Target Modules:    {TARGET_MODULES}")
    print(f"LoRA Config:       r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Seq Length:        {args.max_seq_length}")
    print(f"Batch Size:        {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"Grad Accum:        {args.grad_accum}")
    print(f"Learning Rate:     {args.lr}")
    print(f"BF16 Enabled:      True")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Fine-tuning requires an active GPU.")

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Hardware:          {device_name} ({total_vram_gb:.2f} GB VRAM)")
    print(f"PyTorch Version:   {torch.__version__}")
    print("=" * 70)

    # 1. Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Dataset
    print(f"Loading dataset from {args.data_file}...")
    dataset, total_records, exceed_count = load_dataset_from_jsonl(
        args.data_file, tokenizer, args.max_seq_length
    )
    print(f"  Total records:           {total_records}")
    print(f"  Exceeding max_seq_length: {exceed_count} ({exceed_count / total_records * 100:.2f}%)")

    # 3. Model with 4-bit NF4
    print("\nLoading model in 4-bit NF4 (bitsandbytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # 4. Prepare for k-bit training and attach LoRA
    print("Preparing model for k-bit training and attaching LoRA...")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training configuration
    is_benchmark = args.max_steps > 0
    steps = args.max_steps if is_benchmark else -1
    epochs = args.epochs if not is_benchmark else 1

    warmup_s = max(1, (args.max_steps if is_benchmark else 100) // 10)
    training_args = SFTConfig(
        output_dir=args.output_dir if not is_benchmark else "./tmp_benchmark",
        max_steps=steps,
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="no" if is_benchmark else "epoch",
        save_total_limit=1,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=warmup_s,
        report_to=[],
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    step_cb = StepMetricsCallback()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[step_cb],
    )

    # 6. Run Training / Benchmark
    print("\nStarting execution...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    oom_occurred = False

    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        oom_occurred = True
        print("\n" + "!" * 70)
        print("ERROR: Out of Memory (OOM) encountered during training!")
        print(f"Exception details: {e}")
        print("!" * 70)
        return 1

    total_wall_clock = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_vram_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"OOM Occurred:        {oom_occurred}")
    print(f"Peak VRAM Allocated: {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB ({peak_vram_gb / total_vram_gb * 100:.1f}%)")
    print(f"Peak VRAM Reserved:  {peak_vram_reserved_gb:.2f} GB")
    print(f"Total Wall-Clock:    {total_wall_clock:.2f} s ({total_wall_clock / 60:.2f} min)")

    if step_cb.step_times:
        # discard step 1 warmup for throughput estimate if multiple steps
        times = step_cb.step_times[1:] if len(step_cb.step_times) > 1 else step_cb.step_times
        avg_sec_per_step = sum(times) / len(times)
        effective_batch = args.batch_size * args.grad_accum
        # Approx tokens per step: effective_batch * avg_seq_length (approximated as min(avg, 1024))
        tokens_per_sec = (effective_batch * args.max_seq_length) / avg_sec_per_step if avg_sec_per_step > 0 else 0
        print(f"Avg Sec / Opt Step:  {avg_sec_per_step:.2f} s/step")
        print(f"Approx Throughput:   ~{tokens_per_sec:.1f} tokens/sec")

        # Extrapolation for full 3-epoch run
        steps_per_epoch = total_records // effective_batch
        total_3epoch_steps = steps_per_epoch * 3
        est_total_seconds = total_3epoch_steps * avg_sec_per_step
        est_hours = est_total_seconds / 3600
        print("-" * 70)
        print(f"Full 3-Epoch Extrapolation ({total_records} records, {total_3epoch_steps} optimization steps):")
        print(f"  Estimated Wall-Clock: {est_hours:.2f} hours ({est_total_seconds / 60:.1f} minutes)")
        print(f"  Comparison to ~1-2.5 hr target: {'WITHIN ESTIMATE' if 0.8 <= est_hours <= 3.0 else 'CHECK DISCREPANCY'}")

    if hasattr(train_result, "training_loss"):
        print(f"Final Training Loss: {train_result.training_loss:.4f}")

    # 7. Save adapter if not in benchmark mode
    if is_benchmark:
        print("\n[BENCHMARK MODE] --max_steps was specified. Skipping adapter save.")
        print("Hardware benchmark complete! Ready for full training decision.")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving LoRA adapter and tokenizer to {out_dir}...")
        trainer.model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        # Save run manifest for exact reproducibility
        manifest = {
            "model_id": MODEL_ID,
            "revision": PINNED_REVISION,
            "data_file": str(args.data_file),
            "total_records": total_records,
            "target_modules": TARGET_MODULES,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.lr,
            "max_seq_length": args.max_seq_length,
            "bf16": True,
            "wall_clock_seconds": total_wall_clock,
            "final_training_loss": getattr(train_result, "training_loss", None),
            "peak_vram_gb": peak_vram_gb,
            "hardware": device_name,
        }
        manifest_path = out_dir / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Run manifest saved to {manifest_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
