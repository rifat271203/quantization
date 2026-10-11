"""
Evaluation Runner for Cross-Lingual Quantization vs Hallucination Study.

Evaluates models across the experimental matrix:
  (Bangla, English) x (16-bit, 8-bit, 4-bit)
Supports:
  - Primary context-grounded mode (Passage + Question -> Answer)
  - Secondary closed-book mode (Question -> Answer)
  - Un-tuned base model comparison baseline (--base-baseline)

Usage:
  python scripts/evaluate_models.py --model-id google/gemma-3-1b-it --mode grounded
  python scripts/evaluate_models.py --adapter-dir models/gemma3-1b-bilingual-lora --base-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "google/gemma-3-1b-it"

EVAL_FILES = {
    "bangla": ROOT / "data" / "eval_bangla.jsonl",
    "english": ROOT / "data" / "eval_english.jsonl",
}

DEFAULT_OUTPUT_FILE = ROOT / "results" / "eval_predictions.jsonl"


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_eval_data(lang: str) -> list[dict]:
    path = EVAL_FILES.get(lang)
    if not path or not path.exists():
        raise FileNotFoundError(f"Eval file not found for lang={lang}: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run cross-lingual quantized hallucination evaluation"
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Base or merged model ID/path (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional LoRA adapter directory to attach",
    )
    p.add_argument(
        "--base-baseline",
        action="store_true",
        help="Also evaluate un-tuned base model alongside adapter model as comparison baseline",
    )
    p.add_argument(
        "--mode",
        choices=("grounded", "closed_book"),
        default="grounded",
        help="Primary context-grounded (default) or secondary closed-book mode",
    )
    p.add_argument(
        "--precisions",
        nargs="+",
        choices=("16bit", "8bit", "4bit"),
        default=["16bit", "8bit", "4bit"],
        help="Quantization levels to evaluate (default: 16bit 8bit 4bit)",
    )
    p.add_argument(
        "--languages",
        nargs="+",
        choices=("bangla", "english"),
        default=["bangla", "english"],
        help="Languages to evaluate (default: bangla english)",
    )
    p.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to save output JSONL (default: {DEFAULT_OUTPUT_FILE})",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (0.0 for deterministic greedy decoding)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max eval examples per language",
    )
    return p.parse_args(argv)


def load_model_for_precision(
    model_id: str,
    precision: str,
    adapter_dir: Path | None = None,
):
    print(f"\n[{precision.upper()}] Loading model '{model_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if precision == "16bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    elif precision == "8bit":
        if device != "cuda":
            raise RuntimeError("8-bit evaluation requires CUDA.")
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    elif precision == "4bit":
        if device != "cuda":
            raise RuntimeError("4-bit evaluation requires CUDA.")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown precision: {precision}")

    if adapter_dir is not None:
        print(f"[{precision.upper()}] Attaching LoRA adapter from {adapter_dir} ...")
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return tokenizer, model


def format_prompt(
    tokenizer, instruction: str, passage: str = "", mode: str = "grounded", lang: str = "english"
) -> str:
    """Format prompt for chat-tuned models in grounded or closed-book mode."""
    if mode == "grounded" and passage.strip():
        if lang == "bangla":
            content = (
                "অনুগ্রহ করে নিচের অনুচ্ছেদটি মনোযোগ দিয়ে পড়ুন এবং শুধুমাত্র এই অনুচ্ছেদের তথ্যের "
                "ভিত্তিতে প্রশ্নের সঠিক উত্তর দিন। অনুচ্ছেদে নেই এমন কোনো তথ্য বা অনুমান যোগ করবেন না।\n\n"
                f"অনুচ্ছেদ:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
                f"প্রশ্ন:\n{instruction.strip()}\n\n"
                "উত্তর:"
            )
        else:
            content = (
                "Read the following passage carefully and answer the question based ONLY on facts "
                "directly stated in the passage. Do not introduce outside knowledge or speculation.\n\n"
                f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
                f"Question:\n{instruction.strip()}\n\n"
                "Answer:"
            )
    else:
        content = instruction.strip()

    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"User: {content}\nAssistant: "


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> tuple[str, float, int]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        if temperature > 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    t1 = time.perf_counter()

    gen_tokens = outputs[0][prompt_len:]
    prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    latency = t1 - t0
    num_tokens = len(gen_tokens)
    return prediction, latency, num_tokens


def run_evaluation(args: argparse.Namespace) -> None:
    _configure_stdout()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing IDs to avoid re-evaluating completed runs
    completed: set[tuple[str, bool, str, str, str]] = set()
    if args.output_file.exists():
        with args.output_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed.add(
                            (
                                rec.get("eval_mode", "closed_book"),
                                bool(rec.get("is_base_baseline", False)),
                                rec["precision"],
                                rec["lang"],
                                rec["id"],
                            )
                        )
                    except Exception:
                        pass

    # Model evaluation conditions: adapter run vs un-tuned baseline run
    conditions: list[tuple[str, Path | None, bool]] = []
    conditions.append((args.model_id, args.adapter_dir, False))
    if args.base_baseline and args.adapter_dir is not None:
        conditions.append((args.model_id, None, True))

    total_runs = 0

    for model_id, adapter_path, is_baseline in conditions:
        tag = "BASE_BASELINE" if is_baseline else ("LORA" if adapter_path else "MERGED_BASE")
        print(f"\n{'=' * 64}")
        print(f"Running condition: {tag} (Mode: {args.mode.upper()})")
        print(f"{'=' * 64}")

        for precision in args.precisions:
            try:
                tokenizer, model = load_model_for_precision(
                    model_id=model_id,
                    precision=precision,
                    adapter_dir=adapter_path,
                )
            except Exception as e:
                print(f"Error loading model for {precision}: {e}")
                continue

            for lang in args.languages:
                data = load_eval_data(lang)
                if args.limit:
                    data = data[: args.limit]
                print(
                    f"\n--- Testing [{tag} | {precision.upper()} | {lang.upper()}] "
                    f"({len(data)} items, mode={args.mode}) ---"
                )

                for item in data:
                    key = (args.mode, is_baseline, precision, lang, item["id"])
                    if key in completed:
                        continue

                    passage = item.get("passage", "")
                    prompt = format_prompt(
                        tokenizer=tokenizer,
                        instruction=item["instruction"],
                        passage=passage,
                        mode=args.mode,
                        lang=lang,
                    )

                    prediction, latency, num_tokens = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )

                    tps = (num_tokens / latency) if latency > 0 else 0.0
                    record = {
                        "id": item["id"],
                        "domain": item.get("domain", "unknown"),
                        "lang": lang,
                        "eval_mode": args.mode,
                        "is_base_baseline": is_baseline,
                        "precision": precision,
                        "model_id": model_id,
                        "adapter_dir": str(adapter_path) if adapter_path else None,
                        "passage": passage,
                        "instruction": item["instruction"],
                        "ground_truth": item["response"],
                        "prediction": prediction,
                        "latency_sec": round(latency, 3),
                        "tokens_generated": num_tokens,
                        "tokens_per_sec": round(tps, 1),
                        "timestamp": time.time(),
                    }

                    with args.output_file.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed.add(key)
                    total_runs += 1

                    preview = prediction.replace("\n", " ")[:60]
                    print(f"  [{item['id']}] {lang.upper()} ({tps:.1f} t/s): {preview}...")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n✓ Evaluation complete! Added {total_runs} predictions -> {args.output_file}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
