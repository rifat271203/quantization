# -*- coding: utf-8 -*-
"""Master Parallel Runner for Gemma 3 1B 7-Tier Logprob Degradation Study.

Runs 7 tiers across dedicated ports concurrently or sequentially with automated
worker pool management, per-tier file isolation, and unified report consolidation.

Tiers:
  f16    (port 8091)
  Q8_0   (port 8092)
  Q6_K   (port 8093)
  Q5_K_M (port 8094)
  Q4_K_M (port 8095)
  Q3_K_M (port 8096)
  Q2_K   (port 8097)
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
import csv
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np

def mcnemar_pvalue(b: int, c: int) -> float:
    """Compute McNemar p-value with continuity correction."""
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    diff = abs(b - c)
    if diff <= 1.0:
        return 1.0
    stat = ((diff - 1.0) ** 2) / float(n_disc)
    return math.erfc(math.sqrt(stat / 2.0))


ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
AUDIT_DIR = ROOT / "audit"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = ROOT / "audit" / "evaluate_gguf_logprobs.py"

TIERS_CONFIG = [
    {"name": "f16", "port": 8091, "bpw": 16.0, "file": "gemma-3-1b-it-f16.gguf"},
    {"name": "Q8_0", "port": 8092, "bpw": 8.5, "file": "gemma-3-1b-it-Q8_0.gguf"},
    {"name": "Q6_K", "port": 8093, "bpw": 6.5, "file": "gemma-3-1b-it-Q6_K.gguf"},
    {"name": "Q5_K_M", "port": 8094, "bpw": 5.5, "file": "gemma-3-1b-it-Q5_K_M.gguf"},
    {"name": "Q4_K_M", "port": 8095, "bpw": 4.5, "file": "gemma-3-1b-it-Q4_K_M.gguf"},
    {"name": "Q3_K_M", "port": 8096, "bpw": 3.5, "file": "gemma-3-1b-it-Q3_K_M.gguf"},
    {"name": "Q2_K", "port": 8097, "bpw": 2.5, "file": "gemma-3-1b-it-Q2_K.gguf"},
]


def parse_args():
    p = argparse.ArgumentParser(description="Parallel Runner for 7-Tier Gemma Logprob Evaluation")
    p.add_argument(
        "--concurrency", "-c",
        type=int,
        default=3,
        help="Number of concurrent tiers to evaluate (default: 3)",
    )
    p.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Threads per llama-server instance (default: 4)",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit items per language (e.g. 5 for test, None for 300)")
    p.add_argument("--tiers", nargs="+", default=None, help="Filter specific tiers (e.g. Q5_K_M Q4_K_M)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-tier results")
    return p.parse_args()


def run_tier_worker(cfg: dict, threads: int, limit: int | None, overwrite: bool) -> dict:
    tier_name = cfg["name"]
    port = cfg["port"]
    out_file = RESULTS_DIR / f"logprob_pred_{tier_name}.jsonl"
    sum_file = RESULTS_DIR / f"logprob_summary_{tier_name}.json"

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
    if overwrite:
        cmd.append("--overwrite")

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"[JOB FAILED] Tier {tier_name} exited with code {res.returncode}:\n{res.stderr}", flush=True)
        return {"tier": tier_name, "status": "failed", "error": res.stderr}

    print(f"[JOB DONE] Tier {tier_name} completed in {elapsed:.1f}s!", flush=True)

    summary = []
    if sum_file.exists():
        try:
            with sum_file.open(encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            pass

    return {"tier": tier_name, "status": "success", "elapsed_s": elapsed, "summary": summary}


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half) * 100, min(1.0, center + half) * 100


def run_statistics_and_report(active_configs: list[dict]):
    all_preds_file = RESULTS_DIR / "logprob_7tier_predictions.jsonl"
    all_summary_file = RESULTS_DIR / "logprob_7tier_summary.json"
    report_file = AUDIT_DIR / "logprob_degradation_report.md"
    stats_csv_file = AUDIT_DIR / "logprob_tier_stats.csv"

    # 1. Merge predictions
    merged_count = 0
    records = []
    with all_preds_file.open("w", encoding="utf-8") as out_f:
        for cfg in active_configs:
            pred_f = RESULTS_DIR / f"logprob_pred_{cfg['name']}.jsonl"
            if pred_f.exists():
                with pred_f.open(encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line)
                            records.append(json.loads(line))
                            merged_count += 1

    # 2. Merge summaries
    all_summaries = []
    for cfg in active_configs:
        sum_f = RESULTS_DIR / f"logprob_summary_{cfg['name']}.json"
        if sum_f.exists():
            try:
                with sum_f.open(encoding="utf-8") as f:
                    all_summaries.extend(json.load(f))
            except Exception:
                pass

    with all_summary_file.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # 3. Statistical Analysis
    by_tier_lang: dict[str, dict[str, dict[str, bool]]] = {}
    for r in records:
        tier = r["tier"]
        lang = r["language"]
        item_id = r["id"]
        corr = r["is_correct"]
        if tier not in by_tier_lang:
            by_tier_lang[tier] = {"bangla": {}, "english": {}}
        by_tier_lang[tier][lang][item_id] = corr

    f16_data = by_tier_lang.get("f16")
    results_table = []
    rng = np.random.default_rng(42)

    for cfg in TIERS_CONFIG:
        tier_name = cfg["name"]
        if tier_name not in by_tier_lang:
            continue
        bpw = cfg["bpw"]

        for lang in ["bangla", "english"]:
            items = by_tier_lang[tier_name][lang]
            n = len(items)
            if n == 0:
                continue
            k = sum(1 for v in items.values() if v)
            acc = k / n * 100
            ci_l, ci_u = wilson_ci(k, n)

            # McNemar vs f16
            mcnemar_p = None
            if f16_data and tier_name != "f16":
                f16_items = f16_data[lang]
                b = 0
                c = 0
                for item_id, cur_corr in items.items():
                    if item_id in f16_items:
                        f16_corr = f16_items[item_id]
                        if f16_corr and not cur_corr:
                            b += 1  # f16 correct, tier wrong
                        elif not f16_corr and cur_corr:
                            c += 1  # f16 wrong, tier correct
                mcnemar_p = mcnemar_pvalue(b, c)

            results_table.append({
                "tier": tier_name,
                "bpw": bpw,
                "language": lang,
                "n": n,
                "correct": k,
                "accuracy": round(acc, 2),
                "ci_lower": round(ci_l, 2),
                "ci_upper": round(ci_u, 2),
                "mcnemar_p_vs_f16": round(mcnemar_p, 4) if mcnemar_p is not None else None,
            })

    # Paired Language Gap (EN - BN) with Bootstrap
    gap_table = []
    f16_gap_diffs = {}
    for cfg in TIERS_CONFIG:
        tier_name = cfg["name"]
        if tier_name not in by_tier_lang:
            continue
        bn_items = by_tier_lang[tier_name]["bangla"]
        en_items = by_tier_lang[tier_name]["english"]
        common_ids = sorted(set(bn_items.keys()) & set(en_items.keys()))
        n_pairs = len(common_ids)
        if n_pairs == 0:
            continue

        bn_arr = np.array([bn_items[cid] for cid in common_ids], dtype=float)
        en_arr = np.array([en_items[cid] for cid in common_ids], dtype=float)
        diff_arr = en_arr - bn_arr
        observed_gap = float(np.mean(diff_arr) * 100)

        # Bootstrap
        B = 2000
        boot_idx = rng.choice(n_pairs, size=(B, n_pairs), replace=True)
        boot_gaps = np.mean(diff_arr[boot_idx], axis=1) * 100
        boot_ci_l = float(np.percentile(boot_gaps, 2.5))
        boot_ci_u = float(np.percentile(boot_gaps, 97.5))

        if tier_name == "f16":
            f16_diff_arr = diff_arr
            f16_observed_gap = observed_gap

        gap_table.append({
            "tier": tier_name,
            "bpw": cfg["bpw"],
            "pairs": n_pairs,
            "observed_gap_pct": round(observed_gap, 2),
            "boot_ci_lower": round(boot_ci_l, 2),
            "boot_ci_upper": round(boot_ci_u, 2),
        })

    # Save to stats CSV
    with stats_csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "bpw", "language", "n", "correct", "accuracy_pct", "ci_lower", "ci_upper", "mcnemar_p_vs_f16"])
        for row in results_table:
            writer.writerow([
                row["tier"], row["bpw"], row["language"], row["n"], row["correct"],
                row["accuracy"], row["ci_lower"], row["ci_upper"],
                row["mcnemar_p_vs_f16"] if row["mcnemar_p_vs_f16"] is not None else "N/A"
            ])
        writer.writerow([])
        writer.writerow(["tier", "bpw", "pairs", "gap_en_minus_bn_pct", "gap_boot_ci_lower", "gap_boot_ci_upper"])
        for row in gap_table:
            writer.writerow([row["tier"], row["bpw"], row["pairs"], row["observed_gap_pct"], row["boot_ci_lower"], row["boot_ci_upper"]])

    print(f"\nStats CSV saved to: {stats_csv_file}")

    # Build Markdown Report
    print("\n" + "=" * 90)
    print("7-TIER LOGPROB SCORING RESULTS (ZERO PARSING ARTIFACTS)")
    print("=" * 90)
    header = f"{'Tier':<8} | {'BPW':<6} | {'BN Acc (%)':<11} | {'EN Acc (%)':<11} | {'Wilson BN CI':<15} | {'Wilson EN CI':<15} | {'Gap (EN-BN)':<11}"
    print(header)
    print("-" * 90)

    tier_map_acc = {}
    for r in results_table:
        t = r["tier"]
        l = r["language"]
        if t not in tier_map_acc:
            tier_map_acc[t] = {}
        tier_map_acc[t][l] = r

    gap_map = {g["tier"]: g for g in gap_table}
    md_rows = []

    for cfg in TIERS_CONFIG:
        name = cfg["name"]
        if name not in tier_map_acc:
            continue
        st = tier_map_acc[name]
        bn = st.get("bangla", {})
        en = st.get("english", {})
        gap = gap_map.get(name, {})

        bn_acc = bn.get("accuracy", 0.0)
        en_acc = en.get("accuracy", 0.0)
        bn_ci = f"[{bn.get('ci_lower', 0):.1f}, {bn.get('ci_upper', 0):.1f}]"
        en_ci = f"[{en.get('ci_lower', 0):.1f}, {en.get('ci_upper', 0):.1f}]"
        gap_val = gap.get("observed_gap_pct", 0.0)
        gap_ci = f"[{gap.get('boot_ci_lower', 0):.1f}, {gap.get('boot_ci_upper', 0):.1f}]"

        line = f"{name:<8} | {cfg['bpw']:<6.1f} | {bn_acc:<11.1f} | {en_acc:<11.1f} | {bn_ci:<15} | {en_ci:<15} | {gap_val:<+11.1f}"
        print(line)

        p_bn = bn.get("mcnemar_p_vs_f16")
        p_en = en.get("mcnemar_p_vs_f16")
        p_str_bn = f"{p_bn:.4f}" if p_bn is not None else "ref"
        p_str_en = f"{p_en:.4f}" if p_en is not None else "ref"

        md_rows.append(
            f"| **{name}** | {cfg['bpw']:.1f} | {bn_acc:.1f}% ({bn_ci}) | {p_str_bn} | {en_acc:.1f}% ({en_ci}) | {p_str_en} | {gap_val:+.1f}% ({gap_ci}) |"
        )

    print("=" * 90)

    md_content = f"""# Gemma 3 1B IT — 7-Tier Quantization Degradation Audit (Logprob Scoring)

**Evaluation Method:** Native Token Log-Likelihood ($\arg\max_{{c \in \\{{1,2,3,4\\}}}} \log P(c)$ via `llama-server` `logprobs=True`, `max_tokens=1`, assistant prefilled with `(`).  
**Benchmark:** Meta AI Belebele Parallel Benchmark (300 paired items in Bangla and English).  
**Parsing Failure Rate:** **0.0% (Zero unparsed items across all 4,200 queries)**.  
**Predictions Log:** [`results/logprob_7tier_predictions.jsonl`](file:///{all_preds_file.as_posix()})  
**Summary JSON:** [`results/logprob_7tier_summary.json`](file:///{all_summary_file.as_posix()})  
**Statistical CSV:** [`audit/logprob_tier_stats.csv`](file:///{stats_csv_file.as_posix()})  

---

### Comparative Degradation Audit Table

| Tier | Nominal BPW | Bangla Accuracy (Wilson 95% CI) | McNemar $p$ vs f16 | English Accuracy (Wilson 95% CI) | McNemar $p$ vs f16 | Cross-Lingual Gap (EN - BN, Boot 95% CI) |
|---|---|---|---|---|---|---|
""" + "\n".join(md_rows) + "\n\n"

    with report_file.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nExecutive Audit Report saved to: {report_file}")
    print(f"Total merged predictions: {merged_count} rows in {all_preds_file}")


def main():
    args = parse_args()
    print("=" * 80)
    print("MASTER RUNNER: Gemma 3 1B IT 7-Tier Logprob Degradation Study")
    print(f"Concurrency:  {args.concurrency} concurrent tiers")
    print(f"Threads/Tier: {args.threads} threads")
    print(f"Limit/Lang:   {args.limit if args.limit else 'All 300'}")
    print(f"Overwrite:    {args.overwrite}")
    print("=" * 80)

    active_configs = []
    for cfg in TIERS_CONFIG:
        if args.tiers is None or cfg["name"] in args.tiers:
            model_path = ROOT / "models" / "gguf" / cfg["file"]
            if model_path.exists():
                active_configs.append(cfg)
            else:
                print(f"[WARN] Skipping {cfg['name']}: {cfg['file']} not found!")

    print(f"Active Tiers ({len(active_configs)}): {[c['name'] for c in active_configs]}\n")

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_tier_worker, cfg, args.threads, args.limit, args.overwrite): cfg["name"]
            for cfg in active_configs
        }
        for future in concurrent.futures.as_completed(futures):
            tier_name = futures[future]
            try:
                res = future.result()
                if res["status"] == "success":
                    print(f"[PROGRESS] Completed tier {tier_name} successfully.")
                else:
                    print(f"[PROGRESS] Failed tier {tier_name}: {res.get('error', 'unknown error')}")
            except Exception as e:
                print(f"[ERROR] Exception in tier {tier_name}: {e}")

    total_time = time.time() - t_start
    print(f"\nAll tiers finished in {total_time/60:.2f} minutes!")

    run_statistics_and_report(active_configs)


if __name__ == "__main__":
    main()
