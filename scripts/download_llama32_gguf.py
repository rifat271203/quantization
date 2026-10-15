# -*- coding: utf-8 -*-
"""Download Llama-3.2-1B-Instruct GGUF models across 7 quantization tiers from Hugging Face."""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ID = "unsloth/Llama-3.2-1B-Instruct-GGUF"

TIERS = [
    ("f16",    "Llama-3.2-1B-Instruct-F16.gguf",    2365),
    ("Q8_0",   "Llama-3.2-1B-Instruct-Q8_0.gguf",   1260),
    ("Q6_K",   "Llama-3.2-1B-Instruct-Q6_K.gguf",   975),
    ("Q5_K_M", "Llama-3.2-1B-Instruct-Q5_K_M.gguf", 869),
    ("Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 770),
    ("Q3_K_M", "Llama-3.2-1B-Instruct-Q3_K_M.gguf", 659),
    ("Q2_K",   "Llama-3.2-1B-Instruct-Q2_K.gguf",   554),
]

TARGET_DIR = Path("d:/Tokenizer/quant-hallucination-study/models/gguf")

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"================================================================================")
    print(f"Hugging Face GGUF Downloader: {REPO_ID}")
    print(f"Destination: {TARGET_DIR}")
    print(f"Total Tiers: {len(TIERS)} (Total approx: 7.4 GB)")
    print(f"================================================================================")

    for idx, (tier_name, filename, est_mb) in enumerate(TIERS, 1):
        target_path = TARGET_DIR / filename
        if target_path.exists():
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[{idx}/{len(TIERS)}] ALREADY DOWNLOADED: {filename} ({size_mb:.1f} MB)")
            continue

        print(f"\n[{idx}/{len(TIERS)}] DOWNLOADING {tier_name} ({filename}, ~{est_mb} MB)...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(TARGET_DIR),
            )
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[{idx}/{len(TIERS)}] SUCCESS: {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[{idx}/{len(TIERS)}] ERROR downloading {filename}: {e}", file=sys.stderr)

    print("\n================================================================================")
    print("Download cycle complete! Check models/gguf/ for downloaded weights.")
    print("================================================================================")

if __name__ == "__main__":
    main()
