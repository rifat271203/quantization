# -*- coding: utf-8 -*-
"""Compute Task 2 stats: discordant pairs, gap change vs f16 with bootstrap CI."""

import json
import math
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PRED_FILE = ROOT / "results" / "logprob_7tier_predictions.jsonl"
TIERS = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

def main():
    # Load all predictions
    records = []
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Organize by tier, language, id
    by_tier_lang = {t: {"bangla": {}, "english": {}} for t in TIERS}
    for r in records:
        t = r["tier"]
        l = r["language"]
        item_id = r["id"]
        by_tier_lang[t][l][item_id] = r["is_correct"]

    f16_bn = by_tier_lang["f16"]["bangla"]
    f16_en = by_tier_lang["f16"]["english"]
    common_ids = sorted(list(set(f16_bn.keys()) & set(f16_en.keys())))
    n_pairs = len(common_ids)

    # Base f16 gap
    f16_bn_arr = np.array([f16_bn[cid] for cid in common_ids], dtype=float)
    f16_en_arr = np.array([f16_en[cid] for cid in common_ids], dtype=float)
    f16_gap_item = f16_en_arr - f16_bn_arr
    f16_gap_mean = float(np.mean(f16_gap_item) * 100)

    print("=== TASK 2: McNemar Discordant Counts and Gap Changes vs f16 ===")
    print(f"Total paired items: {n_pairs}")
    print(f"f16 Baseline: BN Acc = {np.mean(f16_bn_arr)*100:.2f}%, EN Acc = {np.mean(f16_en_arr)*100:.2f}%, Gap = {f16_gap_mean:+.2f}%\n")

    # Discordant counts
    print(f"{'Tier':<8} | {'Lang':<7} | {'Total':<5} | {'Acc (%)':<8} | {'b (f16=1, T=0)':<15} | {'c (f16=0, T=1)':<15} | {'McNemar p':<10}")
    print("-" * 75)
    for t in TIERS:
        for l in ["bangla", "english"]:
            items = by_tier_lang[t][l]
            k = sum(items.values())
            acc = k / len(items) * 100
            if t == "f16":
                print(f"{t:<8} | {l:<7} | {len(items):<5} | {acc:<8.2f} | {'ref':<15} | {'ref':<15} | {'ref':<10}")
            else:
                f16_ref = f16_bn if l == "bangla" else f16_en
                b = sum(1 for cid in common_ids if f16_ref[cid] and not items[cid])
                c = sum(1 for cid in common_ids if not f16_ref[cid] and items[cid])
                # McNemar with continuity correction
                diff = abs(b - c)
                disc = b + c
                if disc == 0 or diff <= 1:
                    p = 1.0
                else:
                    stat = ((diff - 1.0) ** 2) / float(disc)
                    p = math.erfc(math.sqrt(stat / 2.0))
                print(f"{t:<8} | {l:<7} | {len(items):<5} | {acc:<8.2f} | {b:<15} | {c:<15} | {p:<10.4f}")

    print("\n" + "=" * 90)
    print("EN-BN GAP CHANGE VERSUS f16 (Bootstrap 95% CI, 2000 resamples, seed 42)")
    print("=" * 90)
    print(f"{'Tier':<8} | {'Tier Gap (%)':<13} | {'ΔGap vs f16 (%)':<16} | {'Bootstrap 95% CI':<22} | {'Significant? (p < 0.05)':<22}")
    print("-" * 90)

    rng = np.random.default_rng(42)
    B = 2000
    boot_indices = rng.choice(n_pairs, size=(B, n_pairs), replace=True)

    for t in TIERS:
        t_bn_arr = np.array([by_tier_lang[t]["bangla"][cid] for cid in common_ids], dtype=float)
        t_en_arr = np.array([by_tier_lang[t]["english"][cid] for cid in common_ids], dtype=float)
        t_gap_item = t_en_arr - t_bn_arr
        t_gap_mean = float(np.mean(t_gap_item) * 100)

        delta_gap_item = t_gap_item - f16_gap_item
        delta_gap_mean = float(np.mean(delta_gap_item) * 100)

        # Bootstrap delta gap
        boot_delta = np.mean(delta_gap_item[boot_indices], axis=1) * 100
        ci_lower = float(np.percentile(boot_delta, 2.5))
        ci_upper = float(np.percentile(boot_delta, 97.5))

        sig = (ci_lower > 0.0 or ci_upper < 0.0)
        sig_str = "YES (p < 0.05)" if sig else "NO (Not Sig)"

        if t == "f16":
            print(f"{t:<8} | {t_gap_mean:<+13.2f} | {'0.00 (ref)':<16} | {'[0.00, 0.00]':<22} | {'ref':<22}")
        else:
            print(f"{t:<8} | {t_gap_mean:<+13.2f} | {delta_gap_mean:<+16.2f} | [{ci_lower:+.2f}, {ci_upper:+.2f}]" + f"{'':<6} | {sig_str:<22}")

if __name__ == "__main__":
    main()
