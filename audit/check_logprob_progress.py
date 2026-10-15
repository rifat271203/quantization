# -*- coding: utf-8 -*-
"""Real-time Progress Tracker for 7-Tier Logprob Evaluation."""

import json
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"

TIERS = [
    ("f16", 16.0),
    ("Q8_0", 8.5),
    ("Q6_K", 6.5),
    ("Q5_K_M", 5.5),
    ("Q4_K_M", 4.5),
    ("Q3_K_M", 3.5),
    ("Q2_K", 2.5),
]

def check_progress():
    print("=" * 80)
    print("LOGPROB EVALUATION REAL-TIME PROGRESS TRACKER (ZERO PARSING ARTIFACTS)")
    print("=" * 80)

    total_all = 0
    for tier, bpw in TIERS:
        f = RESULTS_DIR / f"logprob_pred_{tier}.jsonl"
        bn_items = 0
        en_items = 0
        bn_corr = 0
        en_corr = 0
        if f.exists():
            with f.open(encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("language") == "bangla":
                            bn_items += 1
                            if obj.get("is_correct"):
                                bn_corr += 1
                        elif obj.get("language") == "english":
                            en_items += 1
                            if obj.get("is_correct"):
                                en_corr += 1
        
        tot = bn_items + en_items
        total_all += tot
        status = "[COMPLETED]" if tot >= 600 else ("[RUNNING]" if tot > 0 else "[PENDING]")
        bn_acc = (bn_corr / bn_items * 100) if bn_items else 0.0
        en_acc = (en_corr / en_items * 100) if en_items else 0.0

        print(f"{tier:<8}: {status:<11} {tot:>3}/600 items (BN: {bn_items:>3}/300 [Acc: {bn_acc:>4.1f}%] | EN: {en_items:>3}/300 [Acc: {en_acc:>4.1f}%])")

    print("=" * 80)
    print(f"Total evaluated items across all tiers: {total_all}/4200 ({(total_all/4200*100):.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    check_progress()
