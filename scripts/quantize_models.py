"""
Quantization Pipeline Utility for Cross-Lingual Study.

Converts Hugging Face checkpoints to GGUF and quantizes across the 7-level degradation curve:
  - 16-bit (FP16 / F16)
  - 8-bit (Q8_0)
  - 6-bit (Q6_K)
  - 5-bit (Q5_K_M)
  - 4-bit (Q4_K_M)
  - 3-bit (Q3_K_M)
  - 2-bit (Q2_K)

Usage:
  python scripts/quantize_models.py --model-dir models/gemma3-1b-bilingual-merged --output-dir models/gguf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The 7 canonical degradation levels specified in final_plan.md
DEFAULT_TIERS = [
    "f16",
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q4_k_m",
    "q3_k_m",
    "q2_k",
]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize Hugging Face model across 7 GGUF degradation tiers."
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to merged Hugging Face model directory",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "gguf",
        help="Directory to save GGUF files (default: models/gguf)",
    )
    p.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=None,
        help="Optional path to local llama.cpp clone/binaries",
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        default=DEFAULT_TIERS,
        help=f"Quantization tiers to generate (default: {' '.join(DEFAULT_TIERS)})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Display planned conversion commands without executing them.",
    )
    return p.parse_args(argv)


def find_converter(llama_cpp_dir: Path | None) -> Path | None:
    candidates = []
    if llama_cpp_dir:
        candidates.append(llama_cpp_dir / "convert_hf_to_gguf.py")
    candidates.extend([
        ROOT / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ])
    for c in candidates:
        if c.exists():
            return c
    return None


def find_quantize_binary(llama_cpp_dir: Path | None) -> Path | None:
    candidates = []
    if llama_cpp_dir:
        candidates.extend([
            llama_cpp_dir / "llama-quantize.exe",
            llama_cpp_dir / "llama-quantize",
            llama_cpp_dir / "quantize.exe",
            llama_cpp_dir / "quantize",
            llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
            llama_cpp_dir / "build" / "bin" / "llama-quantize",
        ])
    candidates.extend([
        ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize.exe",
        ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize",
        ROOT / "llama.cpp" / "llama-quantize.exe",
        ROOT / "llama.cpp" / "llama-quantize",
    ])
    for c in candidates:
        if c.exists():
            return c

    # Check system PATH
    for name in ("llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"):
        which_path = shutil.which(name)
        if which_path:
            return Path(which_path)

    return None


def run_quantization(args: argparse.Namespace) -> int:
    _configure_stdout()
    model_dir = args.model_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        return 1

    tiers = [t.lower() for t in args.tiers]

    print("=" * 64)
    print(f"Seven-Tier Quantization Pipeline for {model_dir.name}")
    print("=" * 64)
    print(f"Source Model:      {model_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Requested Tiers:   {', '.join(tiers)} ({len(tiers)} levels)")
    print("=" * 64)

    f16_gguf = output_dir / f"{model_dir.name}-f16.gguf"
    converter = find_converter(args.llama_cpp_dir)

    if not converter:
        print("\n[INFO] llama.cpp converter script (convert_hf_to_gguf.py) not detected in standard paths.")
        print("To convert to 7-tier GGUF format for llama.cpp:")
        print("  1. git clone https://github.com/ggerganov/llama.cpp")
        print("  2. pip install -r llama.cpp/requirements.txt")
        print(f"  3. python llama.cpp/convert_hf_to_gguf.py {model_dir} --outfile {f16_gguf} --outtype f16")
        for t in tiers:
            if t == "f16":
                continue
            out_file = output_dir / f"{model_dir.name}-{t}.gguf"
            print(f"  4. llama-quantize {f16_gguf} {out_file} {t.upper()}")
        print("\nNote: For PyTorch in-process experiments, scripts/evaluate_models.py supports 16-bit, 8-bit, and 4-bit via bitsandbytes.")
        return 0

    # Step 1: Convert to FP16 GGUF
    convert_cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(f16_gguf),
        "--outtype",
        "f16",
    ]
    print(f"\nStep 1: Converting HF checkpoint to F16 GGUF:\n  {' '.join(convert_cmd)}")
    if args.dry_run:
        print("  [DRY RUN] Skipping execution.")
    else:
        if "f16" in tiers or not f16_gguf.exists():
            res = subprocess.run(convert_cmd)
            if res.returncode != 0:
                print("Conversion to F16 GGUF failed.")
                return res.returncode
        else:
            print(f"  F16 GGUF already exists at {f16_gguf}.")

    quant_bin = find_quantize_binary(args.llama_cpp_dir)
    if not quant_bin:
        print(f"\nF16 GGUF ready at {f16_gguf}.")
        print("llama-quantize binary not found on PATH. Build llama.cpp with cmake to generate the remaining tiers.")
        return 0

    # Step 2: Quantize to requested sub-tiers
    print(f"\nStep 2: Quantizing to sub-tiers using {quant_bin.name}:")
    for tier in tiers:
        if tier == "f16":
            continue
        tier_file = output_dir / f"{model_dir.name}-{tier}.gguf"
        cmd = [str(quant_bin), str(f16_gguf), str(tier_file), tier.upper()]
        print(f"  -> Quantizing {tier.upper()}: {' '.join(cmd)}")
        if args.dry_run:
            print("     [DRY RUN] Skipped.")
        else:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"     Quantization to {tier} failed.")
            else:
                print(f"     ✓ Created {tier_file.name}")

    print("\n✓ Quantization process finished.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_quantization(args)


if __name__ == "__main__":
    sys.exit(main())
