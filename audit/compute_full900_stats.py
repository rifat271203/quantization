# -*- coding: utf-8 -*-
"""Statistical Analysis for Belebele 900-Item Logprob Evaluations (Tasks 2, 3, 4).

Generates:
1. audit/full900_tier_stats.csv:
   - Accuracy with Wilson CI and passage-cluster bootstrap CI (2000 resamples, seed 42)
   - McNemar test versus f16 per language
   - EN minus BN gap per tier with passage-cluster bootstrap CI
   - Change in gap versus f16 with passage-cluster bootstrap CI

2. audit/flip_analysis.csv:
   - Agreement with f16 predicted option, stratified by f16 confidence margin
     (top-1 minus top-2 softmax probability) in three equal-count bins (low, mid, high)
   - Fallback for missing statsmodels: passage-cluster bootstrap of language gap
     in flip rate per tier (2000 resamples, seed 42)
   - Accuracy among top third and bottom third of items by f16 margin

3. audit/entropy_stats.csv:
   - Mean entropy of 4-way softmax (nats and bits)
   - Mean max probability
   - Mean ln P(gold) after clipping probabilities at 1e-4
"""

import csv
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PRED_FILE = ROOT / "results" / "logprob_full900_predictions.jsonl"
PARALLEL_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"

TIER_STATS_CSV = ROOT / "audit" / "full900_tier_stats.csv"
FLIP_ANALYSIS_CSV = ROOT / "audit" / "flip_analysis.csv"
ENTROPY_STATS_CSV = ROOT / "audit" / "entropy_stats.csv"

TIERS_ORDER = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
BPW_MAP = {
    "f16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5,
    "Q5_K_M": 5.5,
    "Q4_K_M": 4.5,
    "Q3_K_M": 3.5,
    "Q2_K": 2.5,
}


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95%
    p = float(k) / float(n)
    denom = 1.0 + (z ** 2) / float(n)
    center = (p + (z ** 2) / (2.0 * float(n))) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / float(n)) + ((z ** 2) / (4.0 * (float(n) ** 2))))
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return lower, upper


def mcnemar_test(ref_correct: list[bool], cur_correct: list[bool]) -> tuple[int, int, float]:
    """Compute McNemar test with continuity correction comparing cur against ref."""
    b = 0  # ref=1, cur=0
    c = 0  # ref=0, cur=1
    for r, cur in zip(ref_correct, cur_correct):
        if r and not cur:
            b += 1
        elif not r and cur:
            c += 1
    n_disc = b + c
    if n_disc == 0:
        return b, c, 1.0
    diff = abs(b - c)
    if diff <= 1.0:
        return b, c, 1.0
    stat = ((diff - 1.0) ** 2) / float(n_disc)
    pval = math.erfc(math.sqrt(stat / 2.0))
    return b, c, pval


def get_softmax_probs(cand_lps: dict[str, float]) -> dict[str, float]:
    """Compute normalized 4-way softmax probabilities from candidate logprobs."""
    max_lp = max(cand_lps.values())
    sum_exp = sum(math.exp(lp - max_lp) for lp in cand_lps.values())
    return {k: math.exp(lp - max_lp) / sum_exp for k, lp in cand_lps.items()}


def compute_distribution_metrics(cand_lps: dict[str, float], gold: str) -> tuple[float, float, float, float]:
    """Return (entropy_nats, entropy_bits, max_prob, ln_p_gold_clipped)."""
    probs = get_softmax_probs(cand_lps)
    p_vals = list(probs.values())
    
    ent_nats = 0.0
    ent_bits = 0.0
    for p in p_vals:
        if p > 1e-12:
            ent_nats -= p * math.log(p)
            ent_bits -= p * math.log2(p)
            
    max_p = max(p_vals)
    gold_p = probs.get(gold, 0.0)
    gold_p_clipped = max(gold_p, 1e-4)
    ln_p_gold = math.log(gold_p_clipped)
    
    return ent_nats, ent_bits, max_p, ln_p_gold


def main():
    print("=" * 80)
    print("RUNNING STATISTICAL ANALYSIS ON FULL 900-ITEM DATASET")
    print("=" * 80)

    # 1. Load item metadata (passage mapping) from parallel jsonl
    pair_meta = {}
    with open(PARALLEL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pair_meta[item["pair_id"]] = {
                    "link": item["link"],
                    "bn_passage": item["bn"]["passage"].strip(),
                    "en_passage": item["en"]["passage"].strip(),
                    "gold": str(item["correct_answer_num"]).strip(),
                }
    print(f"Loaded metadata for {len(pair_meta)} parallel item pairs.")

    # 2. Load all prediction records
    preds = {}  # (tier, lang, id) -> record
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[(rec["tier"], rec["language"], rec["id"])] = rec
    print(f"Loaded {len(preds)} total prediction records from {PRED_FILE.name}.")

    # Check available tiers and item counts
    tier_counts = {}
    for (tier, lang, iid) in preds:
        tier_counts[(tier, lang)] = tier_counts.get((tier, lang), 0) + 1
    for tier in TIERS_ORDER:
        bn_cnt = tier_counts.get((tier, "bangla"), 0)
        en_cnt = tier_counts.get((tier, "english"), 0)
        print(f"  Tier {tier:<8}: BN = {bn_cnt}/900, EN = {en_cnt}/900")

    # Group item pairs and cluster IDs
    # Passage clusters: link uniquely identifies passage
    pair_ids = sorted(pair_meta.keys())
    cluster_ids = [pair_meta[pid]["link"] for pid in pair_ids]
    unique_clusters = sorted(list(set(cluster_ids)))
    print(f"Total unique passage clusters: {len(unique_clusters)}")

    cluster_to_indices = {c: [] for c in unique_clusters}
    for idx, c in enumerate(cluster_ids):
        cluster_to_indices[c].append(idx)

    # -------------------------------------------------------------------------
    # TASK 2: TIER STATS WITH WILSON & PASSAGE-CLUSTER BOOTSTRAP CI
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING TASK 2: ACCURACY, WILSON CI, PASSAGE-CLUSTER BOOTSTRAP CI, MCNEMAR, GAP")
    print("=" * 80)

    # Pre-generate 2000 bootstrap resamples of passage clusters (seed 42)
    rng = np.random.RandomState(42)
    n_clusters = len(unique_clusters)
    n_resamples = 2000
    boot_indices_list = []
    for _ in range(n_resamples):
        sampled_c_indices = rng.choice(n_clusters, size=n_clusters, replace=True)
        sample_item_idx = []
        for ci in sampled_c_indices:
            c_name = unique_clusters[ci]
            sample_item_idx.extend(cluster_to_indices[c_name])
        boot_indices_list.append(np.array(sample_item_idx))

    tier_stats_rows = []

    # Get f16 reference predictions for McNemar and delta gap
    f16_bn_correct = [preds[("f16", "bangla", pid)]["is_correct"] for pid in pair_ids if ("f16", "bangla", pid) in preds]
    f16_en_correct = [preds[("f16", "english", pid)]["is_correct"] for pid in pair_ids if ("f16", "english", pid) in preds]
    has_full_f16 = (len(f16_bn_correct) == len(pair_ids) and len(f16_en_correct) == len(pair_ids))

    # Pre-calculate f16 bootstrap gaps if f16 is complete
    if has_full_f16:
        f16_bn_arr = np.array(f16_bn_correct, dtype=float)
        f16_en_arr = np.array(f16_en_correct, dtype=float)
        f16_boot_gaps = []
        for b_idx in boot_indices_list:
            gap_f16 = np.mean(f16_en_arr[b_idx]) - np.mean(f16_bn_arr[b_idx])
            f16_boot_gaps.append(gap_f16)
        f16_boot_gaps = np.array(f16_boot_gaps)
    else:
        f16_boot_gaps = None

    for tier in TIERS_ORDER:
        bpw = BPW_MAP[tier]
        for lang in ["bangla", "english"]:
            # Check availability
            avail_pids = [pid for pid in pair_ids if (tier, lang, pid) in preds]
            if len(avail_pids) < len(pair_ids):
                print(f"Skipping {tier} {lang}: only {len(avail_pids)}/900 available.")
                continue

            correct_list = [preds[(tier, lang, pid)]["is_correct"] for pid in pair_ids]
            n_tot = len(correct_list)
            k_corr = sum(correct_list)
            acc = k_corr / float(n_tot)

            # Wilson CI
            w_lo, w_hi = wilson_ci(k_corr, n_tot)

            # Passage-cluster bootstrap CI for accuracy
            corr_arr = np.array(correct_list, dtype=float)
            boot_accs = [np.mean(corr_arr[b_idx]) for b_idx in boot_indices_list]
            boot_lo = float(np.percentile(boot_accs, 2.5))
            boot_hi = float(np.percentile(boot_accs, 97.5))

            # McNemar vs f16
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

    # Add Gap and Delta Gap for tiers with both languages complete
    gap_dict = {}
    for tier in TIERS_ORDER:
        if all((tier, "bangla", pid) in preds for pid in pair_ids) and all((tier, "english", pid) in preds for pid in pair_ids):
            bn_corr = np.array([preds[(tier, "bangla", pid)]["is_correct"] for pid in pair_ids], dtype=float)
            en_corr = np.array([preds[(tier, "english", pid)]["is_correct"] for pid in pair_ids], dtype=float)
            
            gap_empirical = float(np.mean(en_corr) - np.mean(bn_corr))
            
            # Bootstrap gap and delta gap
            boot_gaps = []
            boot_delta_gaps = []
            for b_idx_i, b_idx in enumerate(boot_indices_list):
                b_gap = np.mean(en_corr[b_idx]) - np.mean(bn_corr[b_idx])
                boot_gaps.append(b_gap)
                if has_full_f16:
                    boot_delta_gaps.append(b_gap - f16_boot_gaps[b_idx_i])
            
            gap_lo = float(np.percentile(boot_gaps, 2.5))
            gap_hi = float(np.percentile(boot_gaps, 97.5))
            
            if tier == "f16":
                delta_gap = 0.0
                d_lo, d_hi = 0.0, 0.0
            else:
                delta_gap = gap_empirical - (np.mean(f16_en_arr) - np.mean(f16_bn_arr))
                d_lo = float(np.percentile(boot_delta_gaps, 2.5))
                d_hi = float(np.percentile(boot_delta_gaps, 97.5))
                
            gap_dict[tier] = {
                "gap_en_minus_bn": round(gap_empirical, 4),
                "gap_boot_ci_lower": round(gap_lo, 4),
                "gap_boot_ci_upper": round(gap_hi, 4),
                "delta_gap_vs_f16": round(delta_gap, 4),
                "delta_gap_boot_ci_lower": round(d_lo, 4),
                "delta_gap_boot_ci_upper": round(d_hi, 4),
            }

    # Merge gap metrics into tier_stats_rows
    for r in tier_stats_rows:
        t = r["tier"]
        if t in gap_dict:
            r.update(gap_dict[t])
        else:
            r.update({
                "gap_en_minus_bn": "",
                "gap_boot_ci_lower": "",
                "gap_boot_ci_upper": "",
                "delta_gap_vs_f16": "",
                "delta_gap_boot_ci_lower": "",
                "delta_gap_boot_ci_upper": "",
            })

    # Save full900_tier_stats.csv
    with open(TIER_STATS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tier", "bpw", "language", "accuracy", "accuracy_pct",
            "wilson_ci_lower", "wilson_ci_upper", "cluster_boot_ci_lower", "cluster_boot_ci_upper",
            "mcnemar_b", "mcnemar_c", "mcnemar_pvalue",
            "gap_en_minus_bn", "gap_boot_ci_lower", "gap_boot_ci_upper",
            "delta_gap_vs_f16", "delta_gap_boot_ci_lower", "delta_gap_boot_ci_upper"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in tier_stats_rows:
            writer.writerow(r)
    print(f"Saved: {TIER_STATS_CSV}")

    # -------------------------------------------------------------------------
    # TASK 3: DECISION FLIPS AND STRATIFICATION BY F16 CONFIDENCE MARGIN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING TASK 3: DECISION FLIPS & F16 CONFIDENCE MARGIN STRATIFICATION")
    print("=" * 80)

    # Check statsmodels
    has_statsmodels = False
    try:
        import statsmodels.api as sm
        has_statsmodels = True
    except ImportError:
        print("[NOTICE] statsmodels is missing; using passage-cluster bootstrap of language gap in flip rate per tier instead.")

    # Compute f16 confidence margins for all 900 items per language
    # Margin = top-1 minus top-2 softmax probability
    f16_margins = {}
    f16_preds = {}
    for lang in ["bangla", "english"]:
        f16_margins[lang] = {}
        f16_preds[lang] = {}
        for pid in pair_ids:
            if ("f16", lang, pid) in preds:
                cand_lps = preds[("f16", lang, pid)]["candidate_logprobs"]
                probs = sorted(get_softmax_probs(cand_lps).values(), reverse=True)
                margin = probs[0] - probs[1]
                f16_margins[lang][pid] = margin
                f16_preds[lang][pid] = preds[("f16", lang, pid)]["pred_option"]

    # Divide 900 items into 3 equal-count bins per language (tertiles: 300 items each)
    margin_bins = {}  # lang -> pid -> bin_name ("low_margin", "mid_margin", "high_margin")
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
            avail_pids = [pid for pid in pair_ids if (tier, lang, pid) in preds]
            if len(avail_pids) < len(pair_ids):
                continue

            # Agreement with f16 overall and by margin bin
            agrees = {
                "overall": [],
                "low_margin": [],
                "mid_margin": [],
                "high_margin": [],
            }
            # Accuracy among top third and bottom third of items by f16 margin
            acc_top_third = []
            acc_bottom_third = []

            for pid in pair_ids:
                cur_pred = preds[(tier, lang, pid)]["pred_option"]
                ref_pred = f16_preds[lang][pid]
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

            acc_top = np.mean(acc_top_third)
            acc_bot = np.mean(acc_bottom_third)

            flip_rows.append({
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "agreement_f16_overall": round(agr_overall, 4),
                "flip_rate_overall": round(flip_rate, 4),
                "agreement_low_margin_bin": round(agr_low, 4),
                "agreement_mid_margin_bin": round(agr_mid, 4),
                "agreement_high_margin_bin": round(agr_high, 4),
                "accuracy_bottom_third": round(acc_bot, 4),
                "accuracy_top_third": round(acc_top, 4),
            })

    # Language gap in flip rate per tier with passage-cluster bootstrap
    flip_gap_dict = {}
    for tier in TIERS_ORDER:
        if all((tier, "bangla", pid) in preds for pid in pair_ids) and all((tier, "english", pid) in preds for pid in pair_ids):
            bn_flips = np.array([preds[(tier, "bangla", pid)]["pred_option"] != f16_preds["bangla"][pid] for pid in pair_ids], dtype=float)
            en_flips = np.array([preds[(tier, "english", pid)]["pred_option"] != f16_preds["english"][pid] for pid in pair_ids], dtype=float)

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
        if t in flip_gap_dict:
            r.update(flip_gap_dict[t])

    with open(FLIP_ANALYSIS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tier", "bpw", "language",
            "agreement_f16_overall", "flip_rate_overall",
            "agreement_low_margin_bin", "agreement_mid_margin_bin", "agreement_high_margin_bin",
            "accuracy_bottom_third", "accuracy_top_third",
            "language_gap_in_flip_rate (BN - EN)", "flip_gap_boot_ci_lower", "flip_gap_boot_ci_upper",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in flip_rows:
            writer.writerow(r)
    print(f"Saved: {FLIP_ANALYSIS_CSV}")

    # -------------------------------------------------------------------------
    # TASK 4: DISTRIBUTION SHAPE (ENTROPY, MAX PROB, LN P(GOLD))
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING TASK 4: DISTRIBUTION SHAPE (ENTROPY, MAX PROB, LN P(GOLD))")
    print("=" * 80)

    entropy_rows = []
    for tier in TIERS_ORDER:
        bpw = BPW_MAP[tier]
        for lang in ["bangla", "english"]:
            avail_pids = [pid for pid in pair_ids if (tier, lang, pid) in preds]
            if len(avail_pids) < len(pair_ids):
                continue

            ent_nats_list = []
            ent_bits_list = []
            max_p_list = []
            ln_gold_list = []

            for pid in pair_ids:
                cand_lps = preds[(tier, lang, pid)]["candidate_logprobs"]
                gold = pair_meta[pid]["gold"]
                e_nats, e_bits, mx_p, ln_g = compute_distribution_metrics(cand_lps, gold)
                ent_nats_list.append(e_nats)
                ent_bits_list.append(e_bits)
                max_p_list.append(mx_p)
                ln_gold_list.append(ln_g)

            entropy_rows.append({
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "mean_entropy_nats": round(float(np.mean(ent_nats_list)), 4),
                "mean_entropy_bits": round(float(np.mean(ent_bits_list)), 4),
                "mean_max_probability": round(float(np.mean(max_p_list)), 4),
                "mean_ln_p_gold_clipped": round(float(np.mean(ln_gold_list)), 4),
            })

    with open(ENTROPY_STATS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tier", "bpw", "language",
            "mean_entropy_nats", "mean_entropy_bits",
            "mean_max_probability", "mean_ln_p_gold_clipped"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in entropy_rows:
            writer.writerow(r)
    print(f"Saved: {ENTROPY_STATS_CSV}")

    print("\n" + "=" * 80)
    print("ALL STATISTICAL ANALYSES SUCCESSFULLY COMPUTED AND SAVED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
