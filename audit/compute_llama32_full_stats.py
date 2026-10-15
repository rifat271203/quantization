# -*- coding: utf-8 -*-
"""Compute Passage-Clustered Statistics, Decision Flips, and Distribution Metrics for Llama 3.2 1B Instruct."""

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

TIERS_ORDER = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
BPW_MAP = {
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

def mcnemar_test(ref_bool: list[bool], cur_bool: list[bool]) -> tuple[int, int, float]:
    b = sum(1 for r, c in zip(ref_bool, cur_bool) if r and not c)
    c = sum(1 for r, c in zip(ref_bool, cur_bool) if not r and c)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_val = min(1.0, 2.0 * cdf)
    return b, c, float(p_val)

def get_softmax_probs(cand_lps: dict[str, float]) -> dict[str, float]:
    max_lp = max(cand_lps.values())
    exps = {k: math.exp(v - max_lp) for k, v in cand_lps.items()}
    sum_exps = sum(exps.values())
    if sum_exps == 0:
        return {k: 0.25 for k in cand_lps}
    return {k: v / sum_exps for k, v in exps.items()}

def main():
    if not PRED_FILE.exists():
        print(f"Error: {PRED_FILE} does not exist.")
        return

    # 1. Load predictions
    preds = {}
    pair_meta = {}
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                key = (rec["tier"], rec["language"], rec["pair_id"])
                preds[key] = {
                    "pred_option": str(rec["predicted_option"]),
                    "is_correct": bool(rec["is_correct"]),
                    "confidence_margin": float(rec.get("confidence_margin", 0.0)),
                    "predictive_entropy": float(rec.get("predictive_entropy", 0.0)),
                    "candidate_logprobs": rec.get("candidate_logprobs", {}),
                }
                pid = rec["pair_id"]
                if pid not in pair_meta:
                    pair_meta[pid] = {
                        "link": rec.get("link", pid),
                        "gold": str(rec["gold_option"]),
                    }

    print(f"Loaded {len(preds)} predictions across {len(pair_meta)} items.")

    pair_ids = sorted(pair_meta.keys())
    cluster_ids = [pair_meta[pid]["link"] for pid in pair_ids]
    unique_clusters = sorted(list(set(cluster_ids)))
    cluster_to_indices = {c: [] for c in unique_clusters}
    for idx, c in enumerate(cluster_ids):
        cluster_to_indices[c].append(idx)

    # 2000 cluster bootstrap resamples
    rng = np.random.RandomState(42)
    n_clusters = len(unique_clusters)
    n_resamples = 2000
    boot_indices_list = []
    for _ in range(n_resamples):
        sampled_c_indices = rng.choice(n_clusters, size=n_clusters, replace=True)
        sample_item_idx = []
        for ci in sampled_c_indices:
            sample_item_idx.extend(cluster_to_indices[unique_clusters[ci]])
        boot_indices_list.append(np.array(sample_item_idx))

    # Reference f16 correctness
    f16_bn_correct = [preds[("f16", "bangla", pid)]["is_correct"] for pid in pair_ids]
    f16_en_correct = [preds[("f16", "english", pid)]["is_correct"] for pid in pair_ids]
    f16_bn_arr = np.array(f16_bn_correct, dtype=float)
    f16_en_arr = np.array(f16_en_correct, dtype=float)

    f16_boot_gaps = []
    for b_idx in boot_indices_list:
        f16_boot_gaps.append(np.mean(f16_en_arr[b_idx]) - np.mean(f16_bn_arr[b_idx]))
    f16_boot_gaps = np.array(f16_boot_gaps)

    # Compute Tier Stats
    tier_stats_rows = []
    gap_dict = {}

    for tier in TIERS_ORDER:
        bpw = BPW_MAP[tier]
        for lang in ["bangla", "english"]:
            correct_list = [preds[(tier, lang, pid)]["is_correct"] for pid in pair_ids]
            n_tot = len(correct_list)
            k_corr = sum(correct_list)
            acc = k_corr / float(n_tot)

            w_lo, w_hi = wilson_ci(k_corr, n_tot)

            corr_arr = np.array(correct_list, dtype=float)
            boot_accs = [np.mean(corr_arr[b_idx]) for b_idx in boot_indices_list]
            boot_lo = float(np.percentile(boot_accs, 2.5))
            boot_hi = float(np.percentile(boot_accs, 97.5))

            if tier == "f16":
                mcn_b, mcn_c, mcn_p = 0, 0, 1.0
            else:
                ref_list = f16_bn_correct if lang == "bangla" else f16_en_correct
                mcn_b, mcn_c, mcn_p = mcnemar_test(ref_list, correct_list)

            tier_stats_rows.append({
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "accuracy": round(acc, 4),
                "accuracy_pct": round(acc * 100.0, 2),
                "wilson_ci_lower": round(w_lo, 4),
                "wilson_ci_upper": round(w_hi, 4),
                "cluster_boot_ci_lower": round(boot_lo, 4),
                "cluster_boot_ci_upper": round(boot_hi, 4),
                "mcnemar_b": mcn_b,
                "mcnemar_c": mcn_c,
                "mcnemar_pvalue": round(mcn_p, 4) if mcn_p >= 0.0001 else "<0.0001",
            })

        # Gap calculation
        bn_corr = np.array([preds[(tier, "bangla", pid)]["is_correct"] for pid in pair_ids], dtype=float)
        en_corr = np.array([preds[(tier, "english", pid)]["is_correct"] for pid in pair_ids], dtype=float)
        empirical_gap = float(np.mean(en_corr) - np.mean(bn_corr))

        boot_gaps = []
        delta_gaps = []
        for idx_b, b_idx in enumerate(boot_indices_list):
            g_t = np.mean(en_corr[b_idx]) - np.mean(bn_corr[b_idx])
            boot_gaps.append(g_t)
            delta_gaps.append(g_t - f16_boot_gaps[idx_b])

        g_lo = float(np.percentile(boot_gaps, 2.5))
        g_hi = float(np.percentile(boot_gaps, 97.5))
        d_lo = float(np.percentile(delta_gaps, 2.5))
        d_hi = float(np.percentile(delta_gaps, 97.5))
        empirical_delta = empirical_gap - float(np.mean(f16_en_arr) - np.mean(f16_bn_arr))

        gap_dict[tier] = {
            "gap_en_minus_bn": round(empirical_gap, 4),
            "gap_boot_ci_lower": round(g_lo, 4),
            "gap_boot_ci_upper": round(g_hi, 4),
            "delta_gap_vs_f16": round(empirical_delta, 4),
            "delta_gap_boot_ci_lower": round(d_lo, 4),
            "delta_gap_boot_ci_upper": round(d_hi, 4),
        }

    for r in tier_stats_rows:
        t = r["tier"]
        r.update(gap_dict[t])

    with open(OUT_STATS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(tier_stats_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(tier_stats_rows)
    print(f"Saved: {OUT_STATS_CSV}")

    # -------------------------------------------------------------
    # TASK 3: FLIP ANALYSIS & MARGIN STRATIFICATION
    # -------------------------------------------------------------
    f16_margins = {}
    f16_preds_map = {}
    for lang in ["bangla", "english"]:
        f16_margins[lang] = {}
        f16_preds_map[lang] = {}
        for pid in pair_ids:
            cand_lps = preds[("f16", lang, pid)]["candidate_logprobs"]
            probs = sorted(get_softmax_probs(cand_lps).values(), reverse=True)
            margin = probs[0] - probs[1]
            f16_margins[lang][pid] = margin
            f16_preds_map[lang][pid] = preds[("f16", lang, pid)]["pred_option"]

    margin_bins = {}
    for lang in ["bangla", "english"]:
        sorted_pids = sorted(pair_ids, key=lambda pid: f16_margins[lang].get(pid, 0.0))
        n = len(sorted_pids)
        t1, t2 = n // 3, (2 * n) // 3
        margin_bins[lang] = {}
        for idx, pid in enumerate(sorted_pids):
            if idx < t1:
                margin_bins[lang][pid] = "low_margin"
            elif idx < t2:
                margin_bins[lang][pid] = "mid_margin"
            else:
                margin_bins[lang][pid] = "high_margin"

    flip_rows = []
    for tier in TIERS_ORDER:
        bpw = BPW_MAP[tier]
        for lang in ["bangla", "english"]:
            agrees = {"overall": [], "low_margin": [], "mid_margin": [], "high_margin": []}
            acc_top_third = []
            acc_bottom_third = []

            for pid in pair_ids:
                cur_pred = preds[(tier, lang, pid)]["pred_option"]
                ref_pred = f16_preds_map[lang][pid]
                ag = (cur_pred == ref_pred)
                bname = margin_bins[lang][pid]
                agrees["overall"].append(ag)
                agrees[bname].append(ag)

                is_corr = preds[(tier, lang, pid)]["is_correct"]
                if bname == "high_margin":
                    acc_top_third.append(is_corr)
                elif bname == "low_margin":
                    acc_bottom_third.append(is_corr)

            agr_overall = np.mean(agrees["overall"])
            agr_low = np.mean(agrees["low_margin"])
            agr_mid = np.mean(agrees["mid_margin"])
            agr_high = np.mean(agrees["high_margin"])
            flip_rate = 1.0 - agr_overall

            flip_rows.append({
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "agreement_f16_overall": round(agr_overall, 4),
                "flip_rate_overall": round(flip_rate, 4),
                "agreement_low_margin_bin": round(agr_low, 4),
                "agreement_mid_margin_bin": round(agr_mid, 4),
                "agreement_high_margin_bin": round(agr_high, 4),
                "accuracy_bottom_third": round(np.mean(acc_bottom_third), 4),
                "accuracy_top_third": round(np.mean(acc_top_third), 4),
            })

    # Language gap in flip rate per tier
    flip_gap_dict = {}
    for tier in TIERS_ORDER:
        bn_flips = np.array([preds[(tier, "bangla", pid)]["pred_option"] != f16_preds_map["bangla"][pid] for pid in pair_ids], dtype=float)
        en_flips = np.array([preds[(tier, "english", pid)]["pred_option"] != f16_preds_map["english"][pid] for pid in pair_ids], dtype=float)
        gap_flip_empirical = float(np.mean(bn_flips) - np.mean(en_flips))

        boot_flip_gaps = []
        for b_idx in boot_indices_list:
            bf_gap = np.mean(bn_flips[b_idx]) - np.mean(en_flips[b_idx])
            boot_flip_gaps.append(bf_gap)

        f_lo = float(np.percentile(boot_flip_gaps, 2.5))
        f_hi = float(np.percentile(boot_flip_gaps, 97.5))

        flip_gap_dict[tier] = {
            "language_gap_in_flip_rate (BN - EN)": round(gap_flip_empirical, 4),
            "flip_gap_boot_ci_lower": round(f_lo, 4),
            "flip_gap_boot_ci_upper": round(f_hi, 4),
        }

    for r in flip_rows:
        t = r["tier"]
        r.update(flip_gap_dict[t])

    with open(OUT_FLIP_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(flip_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flip_rows)
    print(f"Saved: {OUT_FLIP_CSV}")

    # -------------------------------------------------------------
    # TASK 4: ENTROPY STATS
    # -------------------------------------------------------------
    entropy_rows = []
    for tier in TIERS_ORDER:
        bpw = BPW_MAP[tier]
        for lang in ["bangla", "english"]:
            ents = [preds[(tier, lang, pid)]["predictive_entropy"] for pid in pair_ids]
            margs = [preds[(tier, lang, pid)]["confidence_margin"] for pid in pair_ids]
            entropy_rows.append({
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "mean_entropy": round(float(np.mean(ents)), 4),
                "std_entropy": round(float(np.std(ents)), 4),
                "median_entropy": round(float(np.median(ents)), 4),
                "mean_confidence_margin": round(float(np.mean(margs)), 4),
                "std_confidence_margin": round(float(np.std(margs)), 4),
            })

    with open(OUT_ENTROPY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(entropy_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(entropy_rows)
    print(f"Saved: {OUT_ENTROPY_CSV}")

if __name__ == "__main__":
    main()
