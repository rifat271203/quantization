"""
Merge LoRA adapter into base google/gemma-3-1b-it model and export a unified FP16 checkpoint.

Features:
- Pinned to exact base model revision dcc83ea841ab6100d6b47a070329e1ba4cf78752.
- Merges LoRA weights into FP16 precision.
- Exports a complete standalone Hugging Face checkpoint (model + tokenizer) ready for GGUF quantization.

Usage:
  python merge_lora.py --adapter_dir models/gemma3-1b-bilingual-lora --output_dir models/gemma3-1b-bilingual-merged-fp16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "google/gemma-3-1b-it"
PINNED_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into FP16 base model")
    parser.add_argument(
        "--adapter_dir",
        "--adapter-dir",
        dest="adapter_dir",
        type=str,
        required=True,
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Path to save merged FP16 checkpoint",
    )
    parser.add_argument(
        "--base_model",
        "--base-model",
        dest="base_model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--revision",
        default=PINNED_REVISION,
        help=f"Base model HF commit revision (default: {PINNED_REVISION})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to perform merge ('auto', 'cpu', 'cuda')",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    adapter_path = Path(args.adapter_dir)
    output_path = Path(args.output_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    print("=" * 70)
    print("MERGE LORA ADAPTER INTO FP16 BASE MODEL")
    print("=" * 70)
    print(f"Base Model:        {args.base_model}")
    print(f"Pinned Revision:   {args.revision}")
    print(f"Adapter Dir:       {adapter_path}")
    print(f"Output Checkpoint: {output_path}")
    print(f"Target Precision:  torch.float16 (FP16)")
    print("=" * 70)

    t0 = time.perf_counter()

    # 1. Load Tokenizer
    print("\n1/4. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, revision=args.revision, trust_remote_code=True
        )

    # 2. Load Base Model in FP16
    print(f"2/4. Loading base model '{args.base_model}' in FP16 (device={args.device})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 3. Load Adapter and Merge
    print(f"3/4. Loading LoRA adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("     Merging adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    # 4. Save Merged Checkpoint
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"4/4. Saving merged FP16 model and tokenizer to {output_path}...")
    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print(f"✓ Merge complete in {elapsed:.1f} s ({elapsed / 60:.2f} min)")
    print(f"  Merged FP16 checkpoint saved to: {output_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
