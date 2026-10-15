# -*- coding: utf-8 -*-
"""Within-Model Analysis: Does Token Count / Fertility Drive Decision Flips and Degradation?"""

import json
import numpy as np
from pathlib import Path
def calc_point_biserial(binary, continuous):
    binary = np.array(binary, dtype=float)
    continuous = np.array(continuous, dtype=float)
    n = len(binary)
    n1 = np.sum(binary)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 1.0
    mean1 = np.mean(continuous[binary == 1])
    mean0 = np.mean(continuous[binary == 0])
    s_x = np.std(continuous, ddof=1)
    if s_x == 0:
        return 0.0, 1.0
    r = ((mean1 - mean0) / s_x) * np.sqrt((n1 * n0) / (n * (n - 1)))
    # t-test for p-value
    df = n - 2
    t = r * np.sqrt(df / (1.0 - r**2 + 1e-12))
    # approximate p-value
    from math import erf
    pval = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / np.sqrt(2.0))))
    return float(r), float(pval)

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"
GEMMA_PRED = ROOT / "results" / "logprob_full900_predictions.jsonl"
LLAMA_PRED = ROOT / "results" / "llama32_logprob_predictions.jsonl"

def main():
    # 1. Load predictions
    def load_preds(path):
        d = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pid = r.get("pair_id") or r.get("id")
                    d[(r["tier"], r["language"], pid)] = r
        return d

    print("Loading predictions...")
    g_preds = load_preds(GEMMA_PRED)
    l_preds = load_preds(LLAMA_PRED)

    # 2. Load dataset and compute char lengths
    items = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} parallel items.")

    # Approximate token counts or character lengths
    # Bengali passages vary in length and character count
    # Let's inspect length distribution
    bn_lengths = {item["pair_id"]: len(item["bn"]["passage"]) for item in items}
    en_lengths = {item["pair_id"]: len(item["en"]["passage"]) for item in items}

    # For both models at Q4_K_M and Q2_K, check if longer passages flip more frequently
    pair_ids = sorted(bn_lengths.keys())

    for model_name, p_dict in [("Gemma 3 1B", g_preds), ("Llama 3.2 1B", l_preds)]:
        print(f"\n{'='*70}\nWITHIN-MODEL ANALYSIS: {model_name}\n{'='*70}")
        for tier in ["Q4_K_M", "Q3_K_M", "Q2_K"]:
            for lang in ["bangla", "english"]:
                lens = [bn_lengths[pid] if lang == "bangla" else en_lengths[pid] for pid in pair_ids]
                flips = []
                for pid in pair_ids:
                    f16_p = p_dict.get(("f16", lang, pid), {}).get("predicted_option") or p_dict.get(("f16", lang, pid), {}).get("pred_option")
                    tier_p = p_dict.get((tier, lang, pid), {}).get("predicted_option") or p_dict.get((tier, lang, pid), {}).get("pred_option")
                    flips.append(1 if f16_p != tier_p else 0)

                # Divide into length tertiles
                t1, t2 = np.percentile(lens, [33.3, 66.7])
                short_flips = [flips[i] for i in range(len(lens)) if lens[i] <= t1]
                mid_flips = [flips[i] for i in range(len(lens)) if t1 < lens[i] <= t2]
                long_flips = [flips[i] for i in range(len(lens)) if lens[i] > t2]

                s_rate = np.mean(short_flips) * 100
                m_rate = np.mean(mid_flips) * 100
                l_rate = np.mean(long_flips) * 100

                corr, pval = calc_point_biserial(flips, lens)

                print(f"Tier: {tier:<6} | Lang: {lang:<7} | Short: {s_rate:5.1f}% | Mid: {m_rate:5.1f}% | Long: {l_rate:5.1f}% | r_pb = {corr:+.3f} (p = {pval:.4f})")

if __name__ == "__main__":
    main()
