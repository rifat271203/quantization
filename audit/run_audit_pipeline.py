# -*- coding: utf-8 -*-
"""Master Pipeline Runner for Full 900-Item Belebele Audit.

Orchestrates:
1. Waiting for evaluate_full900_logprobs.py to complete all 12,600 predictions.
2. Running run_perplexity_normalised.py (Task 5: 3-part spread on f16, Q4_K_M, Q3_K_M, Q2_K).
3. Running compute_full900_stats.py (Tasks 2, 3, 4: CI, McNemar, flips, entropy).
4. Generating final consolidated summary text and tables.
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
AUDIT_DIR = ROOT / "audit"
RESULTS_DIR = ROOT / "results"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"

PRED_FILE = RESULTS_DIR / "logprob_full900_predictions.jsonl"
EVAL_SCRIPT = AUDIT_DIR / "evaluate_full900_logprobs.py"
PPL_SCRIPT = AUDIT_DIR / "run_perplexity_normalised.py"
STATS_SCRIPT = AUDIT_DIR / "compute_full900_stats.py"

TOTAL_EXPECTED_RECORDS = 12600  # 7 tiers * 2 languages * 900 items


def get_current_record_count() -> int:
    if not PRED_FILE.exists():
        return 0
    cnt = 0
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt


def is_eval_running() -> bool:
    # Check if python running evaluate_full900_logprobs.py or llama-server is active
    try:
        res = subprocess.run(
            ["powershell", "-Command", "Get-Process | Where-Object { $_.ProcessName -match 'llama-server' } | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True, text=True
        )
        cnt = int(res.stdout.strip() or "0")
        return cnt > 0
    except Exception:
        return False


def wait_for_eval_completion():
    print(f"Waiting for logprob evaluation to finish ({TOTAL_EXPECTED_RECORDS} records expected) ...", flush=True)
    last_cnt = -1
    while True:
        cnt = get_current_record_count()
        if cnt != last_cnt:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Progress: {cnt}/{TOTAL_EXPECTED_RECORDS} records ({cnt/TOTAL_EXPECTED_RECORDS*100:.1f}%)", flush=True)
            last_cnt = cnt
            
        if cnt >= TOTAL_EXPECTED_RECORDS:
            print(f"\nAll {TOTAL_EXPECTED_RECORDS} predictions successfully generated!")
            break
            
        # Check if process died
        if not is_eval_running() and cnt < TOTAL_EXPECTED_RECORDS:
            # Re-check count after 5s in case it was transitioning between tiers
            time.sleep(5)
            cnt = get_current_record_count()
            if not is_eval_running() and cnt < TOTAL_EXPECTED_RECORDS:
                print(f"[WARNING] Evaluator seems to have stopped at {cnt} records. Resuming evaluate_full900_logprobs.py ...", flush=True)
                proc = subprocess.Popen([str(PYTHON_EXE), str(EVAL_SCRIPT)])
                proc.wait()
                continue
                
        time.sleep(30)


def run_pipeline():
    print("=" * 80)
    print("STARTING FULL AUDIT PIPELINE")
    print("=" * 80)

    # Step 1: Wait for evaluation
    wait_for_eval_completion()

    # Step 2: Run Task 5 (Perplexity normalisation and 3-part spread)
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING TASK 5 (PERPLEXITY NORMALISATION & 3-PART SPREAD)")
    print("=" * 80, flush=True)
    res_ppl = subprocess.run([str(PYTHON_EXE), str(PPL_SCRIPT)], capture_output=True, text=True, encoding="utf-8")
    print(res_ppl.stdout)
    if res_ppl.returncode != 0:
        print("[ERROR in Task 5]:", res_ppl.stderr)

    # Step 3: Run Tasks 2, 3, 4 (Statistical analysis, bootstrap CIs, flips, entropy)
    print("\n" + "=" * 80)
    print("STEP 3: RUNNING TASKS 2, 3, 4 (STATISTICAL ANALYSIS & BOOTSTRAP CIS)")
    print("=" * 80, flush=True)
    res_stats = subprocess.run([str(PYTHON_EXE), str(STATS_SCRIPT)], capture_output=True, text=True, encoding="utf-8")
    print(res_stats.stdout)
    if res_stats.returncode != 0:
        print("[ERROR in Stats]:", res_stats.stderr)

    print("\n" + "=" * 80)
    print("PIPELINE FULLY COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
