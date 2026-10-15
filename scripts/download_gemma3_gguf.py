# -*- coding: utf-8 -*-
"""Download Gemma 3 1B GGUF models for the 7-tier quantization degradation study."""

import os
import sys
from huggingface_hub import hf_hub_download

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TIERS = [
    "gemma-3-1b-it-f16.gguf",
    "gemma-3-1b-it-Q8_0.gguf",
    "gemma-3-1b-it-Q6_K.gguf",
    "gemma-3-1b-it-Q5_K_M.gguf",
    "gemma-3-1b-it-Q4_K_M.gguf",
    "gemma-3-1b-it-Q3_K_M.gguf",
    "gemma-3-1b-it-Q2_K.gguf",
]

TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gguf")
os.makedirs(TARGET_DIR, exist_ok=True)

def main():
    repo_id = "second-state/gemma-3-1b-it-GGUF"
    print(f"Target directory: {TARGET_DIR}")
    for idx, filename in enumerate(TIERS, 1):
        target_path = os.path.join(TARGET_DIR, filename)
        if os.path.exists(target_path):
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"[{idx}/{len(TIERS)}] ALREADY EXISTS: {filename} ({size_mb:.1f} MB)")
            continue
        print(f"[{idx}/{len(TIERS)}] Downloading {filename} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=TARGET_DIR,
        )
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[{idx}/{len(TIERS)}] COMPLETED: {filename} ({size_mb:.1f} MB)")
    print("\nAll 7 tiers downloaded successfully!")

if __name__ == "__main__":
    main()
