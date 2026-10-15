# -*- coding: utf-8 -*-
"""Position-Bias Calibration and Flip Decomposition Analysis.

Strict rules:
- Uses existing prediction logs only (no new server runs).
- Raw counts next to percentages.
- Exact calibration formulas defined and evaluated.
- 95% passage-clustered bootstrap CIs (488 unique clusters, 2000 resamples, seed 42).
- Flips decomposed into:
  (a) Flips where quantized prediction equals the quantized run's dominant option.
  (b) Other flips.
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
AUDIT_DIR = ROOT / "audit"
PARALLEL_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"
GEMMA_FILE = RESULTS_DIR / "logprob_full900_predictions.jsonl"
LLAMA_FILE = RESULTS_DIR / "llama32_logprob_predictions.jsonl"

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
CANDIDATE_DIGITS = ["1", "2", "3", "4"]

def softmax_probs(cand_lps: dict[str, float]) -> dict[str, float]:
    vals = list(cand_lps.values())
    max_v = max(vals)
    exps = {k: math.exp(v - max_v) for k, v in cand_lps.items()}
    sum_exps = sum(exps.values())
    return {k: v / sum_exps for k, v in exps.items()}

def load_data():
    items = []
    with open(PARALLEL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    pair_ids = sorted([it["pair_id"] for it in items])
    item_meta = {it["pair_id"]: {"link": it.get("link", it["pair_id"])} for it in items}

    def load_preds(path):
        preds = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pid = r.get("id") or r.get("pair_id")
                    opt = str(r.get("pred_option") or r.get("predicted_option")).strip()
                    cand_lps = {str(k): float(v) for k, v in r["candidate_logprobs"].items()}
                    gold = str(r["gold_option"]).strip()
                    preds[(r["tier"], r["language"], pid)] = {
                        "pred_option": opt,
                        "gold_option": gold,
                        "candidate_logprobs": cand_lps,
                        "is_correct": (opt == gold)
                    }
        return preds

    gemma_preds = load_preds(GEMMA_FILE)
    llama_preds = load_preds(LLAMA_FILE)

    return items, pair_ids, item_meta, gemma_preds, llama_preds

def main():
    items, pair_ids, item_meta, gemma_preds, llama_preds = load_data()
    n_total = len(pair_ids)

    # Passage clusters
    cluster_ids = [item_meta[pid]["link"] for pid in pair_ids]
    unique_clusters = np.array(sorted(list(set(cluster_ids))))
    n_clusters = len(unique_clusters)
    cluster_to_indices = {c: [] for c in unique_clusters}
    for idx, pid in enumerate(pair_ids):
        cluster_to_indices[item_meta[pid]["link"]].append(idx)

    # 2000 cluster bootstrap resamples
    rng = np.random.RandomState(42)
    n_resamples = 2000
    boot_indices_list = []
    for _ in range(n_resamples):
        sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sample_idx = []
        for c in sampled_c:
            sample_idx.extend(cluster_to_indices[c])
        boot_indices_list.append(np.array(sample_idx))

    models = [
        ("Gemma-3-1B-IT", gemma_preds),
        ("Llama-3.2-1B-Instruct", llama_preds)
    ]

    calib_rows = []
    decomp_rows = []

    print("=" * 115)
    print("PART 1: POSITION-BIAS CALIBRATION (ACCURACY BEFORE & AFTER WITH 95% CLUSTERED BOOTSTRAP CIs)")
    print("=" * 115)

    for model_name, preds_dict in models:
        for tier in TIERS_ORDER:
            bpw = BPW_MAP[tier]
            for lang in ["bangla", "english"]:
                # 1. Collect candidate logprobs and probabilities across all N=900 items
                run_items = [preds_dict[(tier, lang, pid)] for pid in pair_ids]
                gold_list = [it["gold_option"] for it in run_items]
                orig_pred_list = [it["pred_option"] for it in run_items]

                # Method A: Log-prior calibration via average softmax probabilities
                # p_bar_k = mean_i p_{i, k}
                # score_{i, k} = log p_{i, k} - log p_bar_k
                probs_matrix = np.array([[softmax_probs(it["candidate_logprobs"])[d] for d in CANDIDATE_DIGITS] for it in run_items]) # shape (900, 4)
                p_bar = np.mean(probs_matrix, axis=0) # shape (4,)
                log_p_bar = np.log(np.clip(p_bar, 1e-12, 1.0))
                
                # Calibrated scores under Method A
                scores_A = np.log(np.clip(probs_matrix, 1e-12, 1.0)) - log_p_bar # shape (900, 4)
                calib_A_preds = [CANDIDATE_DIGITS[idx] for idx in np.argmax(scores_A, axis=1)]

                # Method B: Raw logprob mean subtraction
                # l_bar_k = mean_i l_{i, k}
                # score_{i, k} = l_{i, k} - l_bar_k
                lps_matrix = np.array([[it["candidate_logprobs"][d] for d in CANDIDATE_DIGITS] for it in run_items])
                # Cap extreme -99 values at -15.0 for realistic mean subtraction without negative overflow
                lps_capped = np.clip(lps_matrix, -15.0, 0.0)
                l_bar = np.mean(lps_capped, axis=0)
                scores_B = lps_capped - l_bar
                calib_B_preds = [CANDIDATE_DIGITS[idx] for idx in np.argmax(scores_B, axis=1)]

                # Compute accuracies
                orig_correct = np.array([orig_pred_list[i] == gold_list[i] for i in range(n_total)], dtype=float)
                calib_A_correct = np.array([calib_A_preds[i] == gold_list[i] for i in range(n_total)], dtype=float)
                calib_B_correct = np.array([calib_B_preds[i] == gold_list[i] for i in range(n_total)], dtype=float)

                orig_acc = np.mean(orig_correct)
                calib_A_acc = np.mean(calib_A_correct)
                calib_B_acc = np.mean(calib_B_correct)

                # Cluster bootstrap CIs (488 clusters)
                orig_boot_accs = [np.mean(orig_correct[b_idx]) for b_idx in boot_indices_list]
                calib_A_boot_accs = [np.mean(calib_A_correct[b_idx]) for b_idx in boot_indices_list]
                
                orig_ci = (np.percentile(orig_boot_accs, 2.5), np.percentile(orig_boot_accs, 97.5))
                calib_A_ci = (np.percentile(calib_A_boot_accs, 2.5), np.percentile(calib_A_boot_accs, 97.5))

                delta_A = calib_A_acc - orig_acc

                calib_rows.append({
                    "model": model_name,
                    "tier": tier,
                    "bpw": bpw,
                    "language": lang,
                    "total_items": n_total,
                    "raw_correct_count": int(np.sum(orig_correct)),
                    "raw_accuracy_pct": round(orig_acc * 100, 2),
                    "raw_ci_95": f"[{orig_ci[0]*100:.2f}%, {orig_ci[1]*100:.2f}%]",
                    "calibrated_correct_count": int(np.sum(calib_A_correct)),
                    "calibrated_accuracy_pct": round(calib_A_acc * 100, 2),
                    "calibrated_ci_95": f"[{calib_A_ci[0]*100:.2f}%, {calib_A_ci[1]*100:.2f}%]",
                    "delta_percentage_points": round(delta_A * 100, 2),
                    "prior_opt1": round(p_bar[0], 4),
                    "prior_opt2": round(p_bar[1], 4),
                    "prior_opt3": round(p_bar[2], 4),
                    "prior_opt4": round(p_bar[3], 4),
                })

    df_calib = pd.DataFrame(calib_rows)
    calib_csv = AUDIT_DIR / "position_bias_calibration.csv"
    df_calib.to_csv(calib_csv, index=False)
    print(f"Saved Position-Bias Calibration CSV to: {calib_csv}\n")

    # Print Part 1 Table
    header_calib = f"{'Model':<22} | {'Tier':<6} | {'Lang':<7} | {'Raw Acc (Count)':<18} | {'Raw 95% CI':<19} | {'Calib Acc (Count)':<18} | {'Calib 95% CI':<19} | {'Delta':<8}"
    print(header_calib)
    print("-" * 125)
    for r in calib_rows:
        raw_str = f"{r['raw_accuracy_pct']:5.2f}% ({r['raw_correct_count']}/{r['total_items']})"
        cal_str = f"{r['calibrated_accuracy_pct']:5.2f}% ({r['calibrated_correct_count']}/{r['total_items']})"
        delta_str = f"{r['delta_percentage_points']:+5.2f}%"
        print(f"{r['model']:<22} | {r['tier']:<6} | {r['language']:<7} | {raw_str:<18} | {r['raw_ci_95']:<19} | {cal_str:<18} | {r['calibrated_ci_95']:<19} | {delta_str:<8}")

    # -------------------------------------------------------------
    # PART 2: FLIP DECOMPOSITION VS OWN F16
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PART 2: DECOMPOSITION OF DECISION FLIPS RELATIVE TO OWN F16")
    print("=" * 115)
    print("Decomposition categories for flips relative to own f16:")
    print("  (a) Dominant-Option Flips: pred_Q != pred_f16 AND pred_Q == Dominant_Option_Q")
    print("  (b) Other Flips:           pred_Q != pred_f16 AND pred_Q != Dominant_Option_Q\n")

    for model_name, preds_dict in models:
        # Preload f16 predictions
        f16_preds = {"bangla": {}, "english": {}}
        for lang in ["bangla", "english"]:
            for pid in pair_ids:
                f16_preds[lang][pid] = preds_dict[("f16", lang, pid)]["pred_option"]

        for tier in TIERS_ORDER:
            if tier == "f16":
                continue # Flips relative to own f16 are 0 by definition
            bpw = BPW_MAP[tier]
            for lang in ["bangla", "english"]:
                # 1. Identify dominant option for this quantized run
                preds_run = [preds_dict[(tier, lang, pid)]["pred_option"] for pid in pair_ids]
                counts = {opt: preds_run.count(opt) for opt in CANDIDATE_DIGITS}
                dominant_opt = max(counts, key=counts.get)
                dominant_count = counts[dominant_opt]

                # 2. Decompose flips
                total_flips = 0
                dominant_flips = 0
                other_flips = 0

                for pid in pair_ids:
                    p_q = preds_dict[(tier, lang, pid)]["pred_option"]
                    p_f16 = f16_preds[lang][pid]
                    if p_q != p_f16:
                        total_flips += 1
                        if p_q == dominant_opt:
                            dominant_flips += 1
                        else:
                            other_flips += 1

                flip_rate_pct = total_flips / n_total * 100
                dom_pct_of_all = dominant_flips / n_total * 100
                dom_pct_of_flips = (dominant_flips / total_flips * 100) if total_flips > 0 else 0.0
                oth_pct_of_all = other_flips / n_total * 100
                oth_pct_of_flips = (other_flips / total_flips * 100) if total_flips > 0 else 0.0

                decomp_rows.append({
                    "model": model_name,
                    "tier": tier,
                    "bpw": bpw,
                    "language": lang,
                    "total_items": n_total,
                    "dominant_option": dominant_opt,
                    "dominant_option_total_count": dominant_count,
                    "dominant_option_share_pct": round(dominant_count / n_total * 100, 2),
                    "total_flips_count": total_flips,
                    "total_flip_rate_pct": round(flip_rate_pct, 2),
                    "dominant_flips_count": dominant_flips,
                    "dominant_flips_pct_of_items": round(dom_pct_of_all, 2),
                    "dominant_flips_share_of_flips_pct": round(dom_pct_of_flips, 2),
                    "other_flips_count": other_flips,
                    "other_flips_pct_of_items": round(oth_pct_of_all, 2),
                    "other_flips_share_of_flips_pct": round(oth_pct_of_flips, 2),
                })

    df_decomp = pd.DataFrame(decomp_rows)
    decomp_csv = AUDIT_DIR / "flip_decomposition.csv"
    df_decomp.to_csv(decomp_csv, index=False)
    print(f"Saved Flip Decomposition CSV to: {decomp_csv}\n")

    # Print Part 2 Table
    header_decomp = f"{'Model':<22} | {'Tier':<6} | {'Lang':<7} | {'Dom Opt':<8} | {'Total Flips':<16} | {'(a) Dominant Flips':<22} | {'(b) Other Flips':<22}"
    print(header_decomp)
    print("-" * 120)
    for r in decomp_rows:
        tot_str = f"{r['total_flips_count']}/{r['total_items']} ({r['total_flip_rate_pct']:4.1f}%)"
        dom_str = f"{r['dominant_flips_count']}/{r['total_items']} ({r['dominant_flips_pct_of_items']:4.1f}% | {r['dominant_flips_share_of_flips_pct']:4.1f}% of flips)"
        oth_str = f"{r['other_flips_count']}/{r['total_items']} ({r['other_flips_pct_of_items']:4.1f}% | {r['other_flips_share_of_flips_pct']:4.1f}% of flips)"
        dom_label = f"Opt {r['dominant_option']} ({r['dominant_option_share_pct']:4.1f}%)"
        print(f"{r['model']:<22} | {r['tier']:<6} | {r['language']:<7} | {dom_label:<12} | {tot_str:<16} | {dom_str:<22} | {oth_str:<22}")

if __name__ == "__main__":
    main()
