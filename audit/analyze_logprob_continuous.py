# -*- coding: utf-8 -*-
"""Compute continuous metrics on 7-tier logprob predictions.

Metrics:
(a) Mean log-probability of gold option and paired diff vs f16 (with 95% bootstrap CI)
(b) Agreement rate with f16 predicted option
(c) Mean Total Variation Distance (TVD) and KL divergence vs f16
(d) Language x Tier interaction on (a): (EN_tier - EN_f16) - (BN_tier - BN_f16) with 95% bootstrap CI
"""

import csv
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PRED_FILE = ROOT / "results" / "logprob_7tier_predictions.jsonl"
OUT_CSV = ROOT / "audit" / "continuous_stats.csv"

TIERS = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
LANGUAGES = ["bangla", "english"]

def softmax_lps(cand_lps: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Compute softmax probabilities and normalized log-probabilities."""
    opts = ["1", "2", "3", "4"]
    vals = np.array([cand_lps.get(o, -99.0) for o in opts], dtype=float)
    max_val = np.max(vals)
    exps = np.exp(vals - max_val)
    denom = np.sum(exps)
    probs = exps / denom
    norm_lps = (vals - max_val) - np.log(denom)
    
    p_dict = {o: float(probs[i]) for i, o in enumerate(opts)}
    log_p_dict = {o: float(norm_lps[i]) for i, o in enumerate(opts)}
    return p_dict, log_p_dict

def kl_div(p: dict[str, float], q: dict[str, float], eps=1e-12) -> float:
    """KL(p || q)."""
    kl = 0.0
    for o in ["1", "2", "3", "4"]:
        pi = max(p[o], eps)
        qi = max(q[o], eps)
        kl += pi * math.log(pi / qi)
    return max(0.0, kl)

def tvd(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance = 0.5 * sum |p_i - q_i|."""
    return 0.5 * sum(abs(p[o] - q[o]) for o in ["1", "2", "3", "4"])

def main():
    print(f"Reading predictions from: {PRED_FILE}")
    records_by_tier_lang = {t: {l: {} for l in LANGUAGES} for t in TIERS}
    
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = r["tier"]
            l = r["language"]
            item_id = r["id"]
            gold = str(r["gold_option"]).strip()
            pred = str(r["pred_option"]).strip()
            cand_lps = r["candidate_logprobs"]
            p_dict, log_p_dict = softmax_lps(cand_lps)
            gold_logp = log_p_dict[gold]
            
            records_by_tier_lang[t][l][item_id] = {
                "gold": gold,
                "pred": pred,
                "p_dict": p_dict,
                "log_p_dict": log_p_dict,
                "gold_logp": gold_logp,
            }

    # Ensure items are matched across tiers
    f16_bn = records_by_tier_lang["f16"]["bangla"]
    f16_en = records_by_tier_lang["f16"]["english"]
    common_ids = sorted(list(set(f16_bn.keys()) & set(f16_en.keys())))
    n_items = len(common_ids)
    print(f"Loaded {n_items} paired items per language across all 7 tiers.\n")

    # RNG for bootstrap
    rng = np.random.default_rng(42)
    B = 2000
    boot_idx = rng.choice(n_items, size=(B, n_items), replace=True)

    # 1. Per Tier and Language metrics
    summary_rows = []
    print("=" * 110)
    print("TASK 3: CONTINUOUS METRICS (SOFTMAX OVER 4 OPTIONS)")
    print("=" * 110)
    header = f"{'Tier':<8} | {'Lang':<7} | {'Mean Gold LogP':<15} | {'Diff vs f16':<12} | {'Diff 95% Boot CI':<20} | {'Agreement':<10} | {'Mean TVD':<10} | {'Mean KL':<10}"
    print(header)
    print("-" * 110)

    # Pre-extract arrays
    gold_logp_arrays = {t: {l: np.array([records_by_tier_lang[t][l][cid]["gold_logp"] for cid in common_ids]) for l in LANGUAGES} for t in TIERS}

    for t in TIERS:
        for l in LANGUAGES:
            arr_t = gold_logp_arrays[t][l]
            arr_f16 = gold_logp_arrays["f16"][l]
            mean_logp = float(np.mean(arr_t))
            
            if t == "f16":
                diff_mean = 0.0
                ci_l, ci_u = 0.0, 0.0
                agree_rate = 1.0
                mean_tvd_val = 0.0
                mean_kl_val = 0.0
            else:
                diff_arr = arr_t - arr_f16
                diff_mean = float(np.mean(diff_arr))
                boot_diffs = np.mean(diff_arr[boot_idx], axis=1)
                ci_l = float(np.percentile(boot_diffs, 2.5))
                ci_u = float(np.percentile(boot_diffs, 97.5))
                
                # Agreement
                agrees = [records_by_tier_lang[t][l][cid]["pred"] == records_by_tier_lang["f16"][l][cid]["pred"] for cid in common_ids]
                agree_rate = float(np.mean(agrees))
                
                # TVD and KL
                tvds = [tvd(records_by_tier_lang["f16"][l][cid]["p_dict"], records_by_tier_lang[t][l][cid]["p_dict"]) for cid in common_ids]
                kls = [kl_div(records_by_tier_lang["f16"][l][cid]["p_dict"], records_by_tier_lang[t][l][cid]["p_dict"]) for cid in common_ids]
                mean_tvd_val = float(np.mean(tvds))
                mean_kl_val = float(np.mean(kls))

            summary_rows.append({
                "tier": t,
                "language": l,
                "mean_gold_logp": round(mean_logp, 4),
                "paired_diff_vs_f16": round(diff_mean, 4) if t != "f16" else "ref",
                "diff_boot_ci_lower": round(ci_l, 4) if t != "f16" else "ref",
                "diff_boot_ci_upper": round(ci_u, 4) if t != "f16" else "ref",
                "agreement_rate_vs_f16": round(agree_rate, 4),
                "mean_tvd_vs_f16": round(mean_tvd_val, 4),
                "mean_kl_vs_f16": round(mean_kl_val, 4),
            })
            
            diff_str = f"{diff_mean:+.4f}" if t != "f16" else "ref"
            ci_str = f"[{ci_l:+.4f}, {ci_u:+.4f}]" if t != "f16" else "ref"
            print(f"{t:<8} | {l:<7} | {mean_logp:<15.4f} | {diff_str:<12} | {ci_str:<20} | {agree_rate:<10.4f} | {mean_tvd_val:<10.4f} | {mean_kl_val:<10.4f}")

    # 2. Language x Tier Interaction on Mean Gold Log-Prob:
    # Interaction = (EN_tier - EN_f16) - (BN_tier - BN_f16)
    print("\n" + "=" * 95)
    print("LANGUAGE x TIER INTERACTION ON GOLD LOG-PROBABILITY")
    print("Formula: (EN_tier - EN_f16) - (BN_tier - BN_f16)")
    print("=" * 95)
    print(f"{'Tier':<8} | {'Interaction (ΔΔ)':<18} | {'Bootstrap 95% CI':<24} | {'Significant? (p < 0.05)':<22}")
    print("-" * 95)

    interaction_rows = []
    for t in TIERS:
        if t == "f16":
            print(f"{t:<8} | {'0.0000 (ref)':<18} | {'[0.0000, 0.0000]':<24} | {'ref':<22}")
            interaction_rows.append({
                "tier": t,
                "interaction_delta_delta": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "significant": "ref"
            })
            continue

        d_en = gold_logp_arrays[t]["english"] - gold_logp_arrays["f16"]["english"]
        d_bn = gold_logp_arrays[t]["bangla"] - gold_logp_arrays["f16"]["bangla"]
        inter_item = d_en - d_bn
        inter_mean = float(np.mean(inter_item))

        boot_inters = np.mean(inter_item[boot_idx], axis=1)
        ci_l = float(np.percentile(boot_inters, 2.5))
        ci_u = float(np.percentile(boot_inters, 97.5))
        is_sig = (ci_l > 0.0 or ci_u < 0.0)
        sig_str = "YES (p < 0.05)" if is_sig else "NO (Not Sig)"

        print(f"{t:<8} | {inter_mean:<+18.4f} | [{ci_l:+.4f}, {ci_u:+.4f}]" + f"{'':<6} | {sig_str:<22}")
        interaction_rows.append({
            "tier": t,
            "interaction_delta_delta": round(inter_mean, 4),
            "ci_lower": round(ci_l, 4),
            "ci_upper": round(ci_u, 4),
            "significant": sig_str
        })

    # Save to CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "language", "mean_gold_logp", "paired_diff_vs_f16", "diff_boot_ci_lower", "diff_boot_ci_upper", "agreement_rate_vs_f16", "mean_tvd_vs_f16", "mean_kl_vs_f16"])
        for r in summary_rows:
            writer.writerow([r["tier"], r["language"], r["mean_gold_logp"], r["paired_diff_vs_f16"], r["diff_boot_ci_lower"], r["diff_boot_ci_upper"], r["agreement_rate_vs_f16"], r["mean_tvd_vs_f16"], r["mean_kl_vs_f16"]])
        
        writer.writerow([])
        writer.writerow(["tier", "interaction_delta_delta", "ci_lower", "ci_upper", "significant"])
        for r in interaction_rows:
            writer.writerow([r["tier"], r["interaction_delta_delta"], r["ci_lower"], r["ci_upper"], r["significant"]])

    print(f"\nSaved full continuous statistics to: {OUT_CSV}")

if __name__ == "__main__":
    main()
