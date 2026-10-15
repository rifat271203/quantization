# -*- coding: utf-8 -*-
"""Option distribution check across models, tiers, and languages.

Strict evaluation:
- Every number from data files.
- Raw counts next to percentages.
- Shannon entropy in bits (base 2).
- Ground truth option distribution reference.
- Detailed collapse check for Llama Q2_K and Q3_K_M (BN and EN).
- Expected flip rate under uniform random guessing (75%) vs marginal random guessing vs observed flip rate.
- High-margin and low-margin agreement recomputed with exact n reported per bin.
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
AUDIT_DIR = ROOT / "audit"

GEMMA_FILE = RESULTS_DIR / "logprob_full900_predictions.jsonl"
LLAMA_FILE = RESULTS_DIR / "llama32_logprob_predictions.jsonl"
PARALLEL_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"

TIERS_ORDER = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
BPW_MAP = {
    "f16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5,
    "Q5_K_M": 5.5,
    "Q4_K_M": 4.5,
    "Q3_K_M": 3.5,
    "Q2_K": 2.5
}

def compute_entropy_bits(counts: dict[str, int], total: int) -> float:
    """Shannon entropy in bits (base 2). Max entropy for 4 options is log2(4) = 2.0 bits."""
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return round(h, 4)

def softmax_probs(cand_lps: dict[str, float]) -> dict[str, float]:
    vals = list(cand_lps.values())
    max_v = max(vals)
    exps = {k: math.exp(v - max_v) for k, v in cand_lps.items()}
    sum_exps = sum(exps.values())
    return {k: v / sum_exps for k, v in exps.items()}

def load_gold_distribution():
    gold_counts = {"1": 0, "2": 0, "3": 0, "4": 0}
    items = []
    with open(PARALLEL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                items.append(rec)
                gold_counts[str(rec["correct_answer_num"]).strip()] += 1
    return gold_counts, len(items)

def load_model_predictions():
    # Load Gemma 3 predictions
    gemma_preds = {} # (tier, lang, pid) -> dict
    with open(GEMMA_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                pid = rec.get("id") or rec.get("pair_id")
                pred_opt = str(rec.get("pred_option") or rec.get("predicted_option")).strip()
                gemma_preds[(rec["tier"], rec["language"], pid)] = {
                    "pred_option": pred_opt,
                    "gold_option": str(rec["gold_option"]).strip(),
                    "candidate_logprobs": {str(k): float(v) for k, v in rec["candidate_logprobs"].items()},
                    "confidence_margin": float(rec.get("confidence_margin", 0.0)),
                    "is_correct": bool(rec.get("is_correct", pred_opt == str(rec["gold_option"]).strip()))
                }

    # Load Llama 3.2 predictions
    llama_preds = {} # (tier, lang, pid) -> dict
    with open(LLAMA_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                pid = rec.get("pair_id") or rec.get("id")
                pred_opt = str(rec.get("predicted_option") or rec.get("pred_option")).strip()
                llama_preds[(rec["tier"], rec["language"], pid)] = {
                    "pred_option": pred_opt,
                    "gold_option": str(rec["gold_option"]).strip(),
                    "candidate_logprobs": {str(k): float(v) for k, v in rec["candidate_logprobs"].items()},
                    "confidence_margin": float(rec.get("confidence_margin", 0.0)),
                    "is_correct": bool(rec.get("is_correct", pred_opt == str(rec["gold_option"]).strip()))
                }

    return gemma_preds, llama_preds

def main():
    gold_counts, total_gold = load_gold_distribution()
    gold_entropy = compute_entropy_bits(gold_counts, total_gold)
    max_gold_opt = max(gold_counts, key=gold_counts.get)
    max_gold_share = gold_counts[max_gold_opt] / total_gold

    print("=" * 100)
    print("TASK 1: OPTION DISTRIBUTION, PREDICTION ENTROPY & COLLAPSE AUDIT")
    print("=" * 100)
    print(f"Reference Ground Truth Option Distribution (N={total_gold} items):")
    for opt in ["1", "2", "3", "4"]:
        c = gold_counts[opt]
        print(f"  Option {opt}: {c:3d} ({c/total_gold*100:5.2f}%)")
    print(f"  Entropy: {gold_entropy:.4f} bits (Uniform ideal: 2.0000 bits)")
    print(f"  Max Option Share: {max_gold_share*100:.2f}% (Option {max_gold_opt}: {gold_counts[max_gold_opt]}/{total_gold})")
    print("-" * 100)

    gemma_preds, llama_preds = load_model_predictions()
    models = [
        ("Gemma-3-1B-IT", gemma_preds),
        ("Llama-3.2-1B-Instruct", llama_preds)
    ]

    csv_rows = []
    
    # Also add gold row to CSV
    csv_rows.append({
        "model": "Ground Truth Reference",
        "tier": "Reference",
        "bpw": 0.0,
        "language": "all",
        "total_items": total_gold,
        "count_opt1": gold_counts["1"],
        "pct_opt1": round(gold_counts["1"] / total_gold * 100, 2),
        "count_opt2": gold_counts["2"],
        "pct_opt2": round(gold_counts["2"] / total_gold * 100, 2),
        "count_opt3": gold_counts["3"],
        "pct_opt3": round(gold_counts["3"] / total_gold * 100, 2),
        "count_opt4": gold_counts["4"],
        "pct_opt4": round(gold_counts["4"] / total_gold * 100, 2),
        "entropy_bits": gold_entropy,
        "most_pred_option": max_gold_opt,
        "most_pred_count": gold_counts[max_gold_opt],
        "most_pred_share_pct": round(max_gold_share * 100, 2),
    })

    for model_name, preds_dict in models:
        for tier in TIERS_ORDER:
            bpw = BPW_MAP[tier]
            for lang in ["bangla", "english"]:
                counts = {"1": 0, "2": 0, "3": 0, "4": 0}
                total = 0
                for (t, l, pid), data in preds_dict.items():
                    if t == tier and l == lang:
                        opt = data["pred_option"]
                        if opt in counts:
                            counts[opt] += 1
                        total += 1
                
                entropy = compute_entropy_bits(counts, total)
                most_opt = max(counts, key=counts.get) if total > 0 else "N/A"
                most_count = counts[most_opt] if total > 0 else 0
                most_share = (most_count / total * 100) if total > 0 else 0.0

                csv_rows.append({
                    "model": model_name,
                    "tier": tier,
                    "bpw": bpw,
                    "language": lang,
                    "total_items": total,
                    "count_opt1": counts["1"],
                    "pct_opt1": round(counts["1"] / total * 100, 2) if total else 0.0,
                    "count_opt2": counts["2"],
                    "pct_opt2": round(counts["2"] / total * 100, 2) if total else 0.0,
                    "count_opt3": counts["3"],
                    "pct_opt3": round(counts["3"] / total * 100, 2) if total else 0.0,
                    "count_opt4": counts["4"],
                    "pct_opt4": round(counts["4"] / total * 100, 2) if total else 0.0,
                    "entropy_bits": entropy,
                    "most_pred_option": most_opt,
                    "most_pred_count": most_count,
                    "most_pred_share_pct": round(most_share, 2),
                })

    df = pd.DataFrame(csv_rows)
    out_csv = AUDIT_DIR / "option_distribution.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved option distribution CSV to: {out_csv}\n")

    # Print nicely formatted summary table
    print(f"{'Model':<22} | {'Tier':<6} | {'Lang':<7} | {'Opt 1':<12} | {'Opt 2':<12} | {'Opt 3':<12} | {'Opt 4':<12} | {'Entropy':<7} | {'Top Opt':<14}")
    print("-" * 125)
    for r in csv_rows:
        if r["model"] == "Ground Truth Reference":
            continue
        c1 = f"{r['count_opt1']} ({r['pct_opt1']:4.1f}%)"
        c2 = f"{r['count_opt2']} ({r['pct_opt2']:4.1f}%)"
        c3 = f"{r['count_opt3']} ({r['pct_opt3']:4.1f}%)"
        c4 = f"{r['count_opt4']} ({r['pct_opt4']:4.1f}%)"
        top = f"Opt {r['most_pred_option']}: {r['most_pred_share_pct']:4.1f}%"
        print(f"{r['model']:<22} | {r['tier']:<6} | {r['language']:<7} | {c1:<12} | {c2:<12} | {c3:<12} | {c4:<12} | {r['entropy_bits']:<7.4f} | {top:<14}")

    # -------------------------------------------------------------
    # SPECIFIC AUDIT: Llama Q2_K and Q3_K_M (BN and EN)
    # -------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DETAILED COLLAPSE & FLIP ANALYSIS: LLAMA 3.2 1B (Q2_K and Q3_K_M)")
    print("=" * 100)

    # Get all pair_ids present in Llama
    pair_ids = sorted(list({pid for (t, l, pid) in llama_preds.keys() if t == "f16" and l == "bangla"}))
    total_n = len(pair_ids)

    # Precompute f16 predictions and margins for Llama
    f16_preds = {"bangla": {}, "english": {}}
    f16_prob_margins = {"bangla": {}, "english": {}}
    for lang in ["bangla", "english"]:
        for pid in pair_ids:
            item = llama_preds[("f16", lang, pid)]
            f16_preds[lang][pid] = item["pred_option"]
            probs = sorted(softmax_probs(item["candidate_logprobs"]).values(), reverse=True)
            f16_prob_margins[lang][pid] = probs[0] - probs[1]

    # Precompute tercile bins based on f16 probability margin
    margin_tercile_bins = {"bangla": {}, "english": {}}
    tercile_ranges = {"bangla": {}, "english": {}}
    for lang in ["bangla", "english"]:
        sorted_pids = sorted(pair_ids, key=lambda pid: f16_prob_margins[lang][pid])
        t1, t2 = total_n // 3, (2 * total_n) // 3
        margin_tercile_bins[lang]["low"] = sorted_pids[:t1]
        margin_tercile_bins[lang]["mid"] = sorted_pids[t1:t2]
        margin_tercile_bins[lang]["high"] = sorted_pids[t2:]
        
        low_m = [f16_prob_margins[lang][p] for p in margin_tercile_bins[lang]["low"]]
        mid_m = [f16_prob_margins[lang][p] for p in margin_tercile_bins[lang]["mid"]]
        high_m = [f16_prob_margins[lang][p] for p in margin_tercile_bins[lang]["high"]]
        
        tercile_ranges[lang]["low"] = (min(low_m), max(low_m))
        tercile_ranges[lang]["mid"] = (min(mid_m), max(mid_m))
        tercile_ranges[lang]["high"] = (min(high_m), max(high_m))

    # Precompute fixed threshold bins
    # Low: < 0.20, Mid: [0.20, 0.50), High: >= 0.50
    fixed_bins = {"bangla": {"low": [], "mid": [], "high": []}, "english": {"low": [], "mid": [], "high": []}}
    for lang in ["bangla", "english"]:
        for pid in pair_ids:
            m = f16_prob_margins[lang][pid]
            if m < 0.20:
                fixed_bins[lang]["low"].append(pid)
            elif m < 0.50:
                fixed_bins[lang]["mid"].append(pid)
            else:
                fixed_bins[lang]["high"].append(pid)

    for target_tier in ["Q3_K_M", "Q2_K"]:
        for lang in ["bangla", "english"]:
            print(f"\n>>> Model: Llama 3.2 1B | Tier: {target_tier} | Language: {lang.upper()} <<<")
            
            # 1. Option breakdown
            opts_tier = [llama_preds[(target_tier, lang, pid)]["pred_option"] for pid in pair_ids]
            opts_f16 = [f16_preds[lang][pid] for pid in pair_ids]
            counts_tier = {opt: opts_tier.count(opt) for opt in ["1", "2", "3", "4"]}
            counts_f16 = {opt: opts_f16.count(opt) for opt in ["1", "2", "3", "4"]}
            
            print(f"  Distribution at {target_tier}:")
            for opt in ["1", "2", "3", "4"]:
                c = counts_tier[opt]
                print(f"    Option {opt}: {c:3d} ({c/total_n*100:5.2f}%) [f16 reference: {counts_f16[opt]:3d} ({counts_f16[opt]/total_n*100:5.2f}%)]")
            
            ent = compute_entropy_bits(counts_tier, total_n)
            print(f"  Shannon Entropy: {ent:.4f} bits (Max uniform: 2.0000 bits)")
            max_opt = max(counts_tier, key=counts_tier.get)
            print(f"  Most-Predicted Option: Option {max_opt} with {counts_tier[max_opt]}/{total_n} ({counts_tier[max_opt]/total_n*100:.2f}%)")
            
            # Collapse diagnostic
            is_collapsed = counts_tier[max_opt] / total_n > 0.60
            if is_collapsed:
                print(f"  -> Positional Collapse: YES (Option {max_opt} dominates >60% of predictions)")
            else:
                print(f"  -> Positional Collapse: NO (Predictions remain distributed across options, entropy {ent:.4f} bits)")

            # 2. Random guessing expected flip rate vs observed
            # Theoretical uniform random guessing: P(pred_Q == pred_f16) = 1/4 = 0.25 => Flip rate = 75.0%
            expected_flip_uniform = 0.7500
            expected_flips_uniform_n = round(expected_flip_uniform * total_n)
            
            # Marginal-matched random guessing: sum_{i} p_f16(i) * p_tier(i)
            p_f16 = {opt: counts_f16[opt] / total_n for opt in ["1", "2", "3", "4"]}
            p_tier = {opt: counts_tier[opt] / total_n for opt in ["1", "2", "3", "4"]}
            expected_agree_marginal = sum(p_f16[opt] * p_tier[opt] for opt in ["1", "2", "3", "4"])
            expected_flip_marginal = 1.0 - expected_agree_marginal
            expected_flips_marginal_n = round(expected_flip_marginal * total_n)

            # Observed flips
            observed_flips = sum(1 for pid in pair_ids if llama_preds[(target_tier, lang, pid)]["pred_option"] != f16_preds[lang][pid])
            observed_flip_rate = observed_flips / total_n

            print("\n  Flip Rate Comparison:")
            print(f"    Observed Flip Rate:            {observed_flip_rate*100:5.2f}% ({observed_flips}/{total_n})")
            print(f"    Uniform Random Guessing (75%): {expected_flip_uniform*100:5.2f}% ({expected_flips_uniform_n}/{total_n})")
            print(f"    Marginal-Matched Guessing:     {expected_flip_marginal*100:5.2f}% ({expected_flips_marginal_n}/{total_n})")
            diff_from_uniform = (observed_flip_rate - expected_flip_uniform) * 100
            diff_from_marginal = (observed_flip_rate - expected_flip_marginal) * 100
            print(f"    Observed vs Uniform Delta:     {diff_from_uniform:+5.2f}% points")
            print(f"    Observed vs Marginal Delta:    {diff_from_marginal:+5.2f}% points")

            # 3. High-margin and low-margin agreement recomputation
            print("\n  Margin Stratification 1: Equal Terciles (n=300 each):")
            for bname in ["low", "mid", "high"]:
                pids_bin = margin_tercile_bins[lang][bname]
                n_b = len(pids_bin)
                r_min, r_max = tercile_ranges[lang][bname]
                agrees = sum(1 for pid in pids_bin if llama_preds[(target_tier, lang, pid)]["pred_option"] == f16_preds[lang][pid])
                agr_rate = agrees / n_b if n_b > 0 else 0.0
                accs = sum(1 for pid in pids_bin if llama_preds[(target_tier, lang, pid)]["is_correct"])
                acc_rate = accs / n_b if n_b > 0 else 0.0
                print(f"    {bname.capitalize():<4} Tercile (n={n_b:3d}, margin range [{r_min:.4f}, {r_max:.4f}]): "
                      f"Agreement={agr_rate*100:5.2f}% ({agrees}/{n_b}) | Accuracy={acc_rate*100:5.2f}% ({accs}/{n_b})")

            print("\n  Margin Stratification 2: Fixed Absolute Margin Thresholds (p(1) - p(2)):")
            for bname, lbl in [("low", "Low Margin (< 0.20)         "), 
                               ("mid", "Mid Margin (0.20 <= m < 0.50)"), 
                               ("high", "High Margin (>= 0.50)        ")]:
                pids_bin = fixed_bins[lang][bname]
                n_b = len(pids_bin)
                if n_b > 0:
                    agrees = sum(1 for pid in pids_bin if llama_preds[(target_tier, lang, pid)]["pred_option"] == f16_preds[lang][pid])
                    agr_rate = agrees / n_b
                    accs = sum(1 for pid in pids_bin if llama_preds[(target_tier, lang, pid)]["is_correct"])
                    acc_rate = accs / n_b
                    print(f"    {lbl} (n={n_b:3d}, {n_b/total_n*100:5.2f}% of items): "
                          f"Agreement={agr_rate*100:5.2f}% ({agrees}/{n_b}) | Accuracy={acc_rate*100:5.2f}% ({accs}/{n_b})")
                else:
                    print(f"    {lbl} (n=  0,   0.00% of items): N/A")

if __name__ == "__main__":
    main()
