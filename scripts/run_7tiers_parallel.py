# -*- coding: utf-8 -*-
"""Master Parallel Runner for Gemma 3 1B 7-Tier Quantization Degradation Study.

Runs 7 tiers across dedicated ports concurrently or sequentially with automated
worker pool management, per-tier file isolation, and unified report consolidation.

Tiers:
  f16    (port 8081)
  Q8_0   (port 8082)
  Q6_K   (port 8083)
  Q5_K_M (port 8084)
  Q4_K_M (port 8085)
  Q3_K_M (port 8086)
  Q2_K   (port 8087)
"""

import os
import sys

# Ensure UTF-8 everywhere on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = ROOT / "audit" / "evaluate_gguf_7tiers.py"

TIERS_CONFIG = [
    {"name": "f16", "port": 8081, "bpw": 16.0, "file": "gemma-3-1b-it-f16.gguf"},
    {"name": "Q8_0", "port": 8082, "bpw": 8.5, "file": "gemma-3-1b-it-Q8_0.gguf"},
    {"name": "Q6_K", "port": 8083, "bpw": 6.5, "file": "gemma-3-1b-it-Q6_K.gguf"},
    {"name": "Q5_K_M", "port": 8084, "bpw": 5.5, "file": "gemma-3-1b-it-Q5_K_M.gguf"},
    {"name": "Q4_K_M", "port": 8085, "bpw": 4.5, "file": "gemma-3-1b-it-Q4_K_M.gguf"},
    {"name": "Q3_K_M", "port": 8086, "bpw": 3.5, "file": "gemma-3-1b-it-Q3_K_M.gguf"},
    {"name": "Q2_K", "port": 8087, "bpw": 2.5, "file": "gemma-3-1b-it-Q2_K.gguf"},
]


def parse_args():
    p = argparse.ArgumentParser(description="Parallel / Batch Runner for 7-Tier Gemma Degradation")
    p.add_argument(
        "--concurrency", "-c",
        type=int,
        default=2,
        help="Number of concurrent tiers to evaluate (default: 2, set 7 for all in parallel)",
    )
    p.add_argument(
        "--threads", "-t",
        type=int,
        default=None,
        help="Threads per llama-server instance (default: auto based on 16 cores / concurrency)",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit items per language (e.g. 5 for quick check, None for 300)")
    p.add_argument("--tiers", nargs="+", default=None, help="Filter specific tiers (e.g. Q4_K_M Q2_K)")
    return p.parse_args()


def run_tier_worker(cfg: dict, threads: int, limit: int | None) -> dict:
    tier_name = cfg["name"]
    port = cfg["port"]
    out_file = RESULTS_DIR / f"pred_{tier_name}.jsonl"
    sum_file = RESULTS_DIR / f"summary_{tier_name}.json"

    print(f"\n[JOB START] Launching tier: {tier_name} on port {port} (threads: {threads}) ...", flush=True)

    cmd = [
        str(PYTHON_EXE),
        str(EVAL_SCRIPT),
        "--tiers", tier_name,
        "--port", str(port),
        "--threads", str(threads),
        "--output-file", str(out_file),
        "--summary-file", str(sum_file),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"[JOB FAILED] Tier {tier_name} exited with code {res.returncode}:\n{res.stderr}", flush=True)
        return {"tier": tier_name, "status": "failed", "error": res.stderr}

    print(f"[JOB DONE] Tier {tier_name} completed in {elapsed:.1f}s!", flush=True)

    # Read summary if available
    summary = []
    if sum_file.exists():
        try:
            with sum_file.open(encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            pass

    return {"tier": tier_name, "status": "success", "elapsed_s": elapsed, "summary": summary}


def consolidate_results(active_configs: list[dict]):
    all_preds_file = RESULTS_DIR / "gguf_7tier_predictions.jsonl"
    all_summary_file = RESULTS_DIR / "gguf_7tier_summary.json"
    report_file = RESULTS_DIR / "gguf_7tier_degradation_report.md"

    # 1. Merge predictions
    merged_count = 0
    with all_preds_file.open("w", encoding="utf-8") as out_f:
        for cfg in active_configs:
            pred_f = RESULTS_DIR / f"pred_{cfg['name']}.jsonl"
            if pred_f.exists():
                with pred_f.open(encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line)
                            merged_count += 1

    # 2. Merge summaries
    all_summaries = []
    tier_stats = {}
    for cfg in active_configs:
        sum_f = RESULTS_DIR / f"summary_{cfg['name']}.json"
        if sum_f.exists():
            try:
                with sum_f.open(encoding="utf-8") as f:
                    data = json.load(f)
                    all_summaries.extend(data)
                    tier_stats[cfg["name"]] = {d["language"]: d for d in data}
            except Exception:
                pass

    with all_summary_file.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # 3. Build comparison report
    print("\n" + "=" * 90)
    print("7-TIER QUANTIZATION DEGRADATION STUDY: CONSOLIDATED RESULTS")
    print("=" * 90)
    header = f"{'Tier':<8} | {'BPW':<6} | {'BN Acc (%)':<11} | {'EN Acc (%)':<11} | {'BN Halluc (%)':<13} | {'EN Halluc (%)':<13} | {'Gap (Δ)':<8}"
    print(header)
    print("-" * 90)

    table_rows_md = []
    for cfg in TIERS_CONFIG:
        name = cfg["name"]
        if name not in tier_stats:
            continue
        st = tier_stats[name]
        bn = st.get("bangla", {})
        en = st.get("english", {})

        bn_acc = bn.get("accuracy_pct", 0.0)
        en_acc = en.get("accuracy_pct", 0.0)
        bn_hall = bn.get("hallucination_pct", 0.0)
        en_hall = en.get("hallucination_pct", 0.0)
        gap = round(bn_hall - en_hall, 2)

        line = f"{name:<8} | {cfg['bpw']:<6.1f} | {bn_acc:<11.1f} | {en_acc:<11.1f} | {bn_hall:<13.1f} | {en_hall:<13.1f} | {gap:<+8.1f}"
        print(line)

        table_rows_md.append(
            f"| **{name}** | {cfg['bpw']:.1f} | {bn_acc:.1f}% | {en_acc:.1f}% | {bn_hall:.1f}% | {en_hall:.1f}% | {gap:+.1f}% |"
        )

    print("=" * 90)

    # Save markdown report
    md_content = f"""# Gemma 3 1B IT — 7-Tier Quantization Degradation Report

**Dataset:** Meta AI Belebele Parallel Benchmark (300 items BN vs EN)  
**Tiers Evaluated:** {len(tier_stats)} tiers (`f16` down to `Q2_K`)  
**Predictions Log:** [`{all_preds_file.name}`]({all_preds_file.as_uri()})  
**Summary JSON:** [`{all_summary_file.name}`]({all_summary_file.as_uri()})

---

### Comparative Degradation Table

| Tier | Nominal BPW | Bangla Acc | English Acc | Bangla Hallucination | English Hallucination | Gap (BN - EN) |
|---|---|---|---|---|---|---|
""" + "\n".join(table_rows_md) + "\n\n"

    with report_file.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nExecutive Markdown Report saved to: {report_file}")
    print(f"Total merged predictions: {merged_count} rows in {all_preds_file}")


def main():
    args = parse_args()
    concurrency = max(1, min(args.concurrency, 7))

    # CPU thread allocation (system has 16 logical cores)
    threads_per_worker = args.threads if args.threads else max(2, 14 // concurrency)

    # Filter active configs
    if args.tiers:
        active_configs = [c for c in TIERS_CONFIG if c["name"] in args.tiers]
    else:
        active_configs = TIERS_CONFIG

    print("=" * 80)
    print("GEMMA 3 1B IT: 7-TIER MASTER BENCHMARK RUNNER")
    print(f"Selected Tiers:      {[c['name'] for c in active_configs]}")
    print(f"Concurrency Level:   {concurrency} workers")
    print(f"Threads per Worker:  {threads_per_worker}")
    print(f"Items per Language:  {args.limit if args.limit else 'All 300'}")
    print("=" * 80)

    t_start = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(run_tier_worker, cfg, threads_per_worker, args.limit): cfg
            for cfg in active_configs
        }
        for future in concurrent.futures.as_completed(futures):
            cfg = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                print(f"Tier {cfg['name']} raised exception: {exc}")

    total_time = time.time() - t_start
    print(f"\nAll worker tasks finished in {total_time:.1f}s ({total_time / 60:.2f} minutes).")

    consolidate_results(active_configs)


if __name__ == "__main__":
    main()
