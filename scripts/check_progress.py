# -*- coding: utf-8 -*-
"""Check real-time progress of running quantization evaluation."""

import os
import sys
import json
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path("d:/Tokenizer/quant-hallucination-study/results")

TIERS = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

def main():
    print("=" * 70)
    print("REAL-TIME EVALUATION PROGRESS TRACKER")
    print("=" * 70)
    
    total_evaluated = 0
    for tier in TIERS:
        pred_file = RESULTS_DIR / f"pred_{tier}.jsonl"
        if not pred_file.exists():
            print(f"{tier:<8} : [PENDING] Waiting in queue...")
            continue
            
        bn_count = 0
        en_count = 0
        bn_correct = 0
        en_correct = 0
        bn_hall = 0
        en_hall = 0
        
        with pred_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    lang = obj.get("language")
                    correct = obj.get("is_correct", False)
                    hall = obj.get("is_distractor_hallucination", False)
                    if lang == "bangla":
                        bn_count += 1
                        if correct: bn_correct += 1
                        if hall: bn_hall += 1
                    else:
                        en_count += 1
                        if correct: en_correct += 1
                        if hall: en_hall += 1
                except Exception:
                    pass
                    
        total = bn_count + en_count
        total_evaluated += total
        
        status = "COMPLETED" if (bn_count >= 300 and en_count >= 300) else "RUNNING"
        bn_acc = (bn_correct / bn_count * 100) if bn_count else 0.0
        bn_h_pct = (bn_hall / bn_count * 100) if bn_count else 0.0
        en_acc = (en_correct / en_count * 100) if en_count else 0.0
        en_h_pct = (en_hall / en_count * 100) if en_count else 0.0
        
        print(f"{tier:<8} : [{status}] {total}/600 items (BN: {bn_count}/300 [Acc: {bn_acc:.1f}%, Halluc: {bn_h_pct:.1f}%] | EN: {en_count}/300 [Acc: {en_acc:.1f}%])")

    print("=" * 70)
    print(f"Total evaluated items across all tiers: {total_evaluated}")


if __name__ == "__main__":
    import time
    if "--watch" in sys.argv or "-w" in sys.argv:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                main()
                print("\nAuto-refreshing every 5 seconds... (Press Ctrl+C to stop)")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        main()
