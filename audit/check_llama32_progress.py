# -*- coding: utf-8 -*-
"""Real-time Progress Tracker for Llama 3.2 1B Belebele Logprob Evaluation."""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PRED_FILE = ROOT / "results" / "llama32_logprob_predictions.jsonl"

TIERS = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

def main():
    if not PRED_FILE.exists():
        print(f"File {PRED_FILE} does not exist yet. Run the evaluation script first.")
        return

    counts = {t: {"bangla": 0, "english": 0, "bn_corr": 0, "en_corr": 0} for t in TIERS}
    with open(PRED_FILE, encoding="utf-8") as in_f:
        for line in in_f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    t = rec["tier"]
                    l = rec["language"]
                    if t in counts and l in counts[t]:
                        counts[t][l] += 1
                        if rec.get("is_correct"):
                            if l == "bangla":
                                counts[t]["bn_corr"] += 1
                            else:
                                counts[t]["en_corr"] += 1
                except Exception:
                    pass

    print("=" * 80)
    print("LLAMA 3.2 1B INSTRUCT — LOGPROB EVALUATION PROGRESS TRACKER (12,600 TOTAL)")
    print("=" * 80)
    total = 0
    for t in TIERS:
        bn = counts[t]["bangla"]
        en = counts[t]["english"]
        tot = bn + en
        total += tot
        bn_acc = (counts[t]["bn_corr"] / bn * 100) if bn else 0.0
        en_acc = (counts[t]["en_corr"] / en * 100) if en else 0.0
        status = "[COMPLETED]" if tot == 1800 else ("[RUNNING]" if tot > 0 else "[PENDING]")
        print(f"{t:<8}: {status:<11} {tot:>4}/1800 items (BN: {bn:>3}/900 [Acc: {bn_acc:>4.1f}%] | EN: {en:>3}/900 [Acc: {en_acc:>4.1f}%])")

    print("=" * 80)
    print(f"Total evaluated predictions: {total}/12600 ({total/12600*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
