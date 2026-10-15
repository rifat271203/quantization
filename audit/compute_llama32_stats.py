# -*- coding: utf-8 -*-
"""Compute Passage-Clustered Statistics, Decision Flips, and Distribution Metrics for Llama 3.2 1B."""

import csv
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PRED_FILE = ROOT / "results" / "llama32_logprob_predictions.jsonl"
OUT_STATS_CSV = ROOT / "audit" / "llama32_tier_stats.csv"
OUT_FLIP_CSV = ROOT / "audit" / "llama32_flip_analysis.csv"
OUT_ENTROPY_CSV = ROOT / "audit" / "llama32_entropy_stats.csv"

TIERS = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
TIER_BPW = {
    "f16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5, "Q5_K_M": 5.5,
    "Q4_K_M": 4.5, "Q3_K_M": 3.5, "Q2_K": 2.5,
}

def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def mcnemar_test(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_val = 2.0 * cdf
    return min(1.0, float(p_val))

def main():
    if not PRED_FILE.exists():
        print(f"Error: {PRED_FILE} does not exist. Run evaluation first.")
        return

    # Load all predictions
    records = []
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} predictions from {PRED_FILE.name}")
    
    # Organize by (tier, language, pair_id)
    by_tier_lang = {}
    passage_by_id = {}
    for r in records:
        t = r["tier"]
        l = r["language"]
        pid = r["pair_id"]
        by_tier_lang.setdefault((t, l), {})[pid] = r
        passage_by_id[pid] = r.get("link", r.get("passage", pid))

    # Cluster passages
    passages = sorted(list(set(passage_by_id.values())))
    passage_to_pids = {p: [] for p in passages}
    for pid, p in passage_by_id.items():
        passage_to_pids[p].append(pid)

    rng = np.random.default_rng(42)
    B = 2000

    # 1. Tier stats
    stat_rows = []
    for t in TIERS:
        for lang in ["bangla", "english"]:
            d = by_tier_lang.get((t, lang), {})
            if not d:
                continue
            n = len(d)
            k = sum(1 for r in d.values() if r["is_correct"])
            acc = k / n
            w_l, w_u = wilson_ci(k, n)

            # Passage cluster bootstrap
            boot_accs = []
            f16_d = by_tier_lang.get(("f16", lang), {})
            b_cnt = 0
            c_cnt = 0
            for pid, r in d.items():
                if pid in f16_d:
                    f16_corr = f16_d[pid]["is_correct"]
                    t_corr = r["is_correct"]
                    if f16_corr and not t_corr:
                        b_cnt += 1
                    elif not f16_corr and t_corr:
                        c_cnt += 1

            p_val = mcnemar_test(b_cnt, c_cnt)

            for _ in range(B):
                sampled_p = rng.choice(passages, size=len(passages), replace=True)
                s_k = 0
                s_n = 0
                for sp in sampled_p:
                    for pid in passage_to_pids[sp]:
                        if pid in d:
                            s_n += 1
                            if d[pid]["is_correct"]:
                                s_k += 1
                boot_accs.append(s_k / s_n if s_n else 0.0)

            c_l, c_u = np.percentile(boot_accs, [2.5, 97.5])

            stat_rows.append({
                "tier": t,
                "bpw": TIER_BPW.get(t, 0.0),
                "language": lang,
                "accuracy": round(acc, 4),
                "accuracy_pct": round(acc * 100, 2),
                "wilson_ci_lower": round(w_l, 4),
                "wilson_ci_upper": round(w_u, 4),
                "cluster_boot_ci_lower": round(c_l, 4),
                "cluster_boot_ci_upper": round(c_u, 4),
                "mcnemar_b": b_cnt,
                "mcnemar_c": c_cnt,
                "mcnemar_pvalue": round(p_val, 4),
            })

    # Write stats CSV
    if stat_rows:
        keys = list(stat_rows[0].keys())
        with open(OUT_STATS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(stat_rows)
        print(f"Saved tier stats to: {OUT_STATS_CSV.name}")

if __name__ == "__main__":
    main()
