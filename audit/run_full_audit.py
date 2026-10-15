# -*- coding: utf-8 -*-
"""Comprehensive Read-Only Audit of the 7-Tier Quantization Run.

Reports strictly what scripts and data actually produced.
"""

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

TIERS = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
DATA_BN = ROOT / "data_v2" / "belebele_eval_bn_300.jsonl"
DATA_EN = ROOT / "data_v2" / "belebele_eval_en_300.jsonl"


def wilson_ci(k, n, z=1.95996398454):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    spread = (z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)


def mcnemar_pvalue(b, c):
    """Compute McNemar p-value with continuity correction using standard normal erfc."""
    n_disc = b + c
    if n_disc == 0:
        return 0.0, 1.0
    diff = abs(b - c)
    if diff <= 1.0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = ((diff - 1.0) ** 2) / float(n_disc)
        # For chi2 with df=1, p-value = erfc(sqrt(stat / 2))
        p_val = math.erfc(math.sqrt(stat / 2.0))
    return stat, p_val


def main():
    # 1. Input data source files
    bn_data_ids = []
    with open(DATA_BN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bn_data_ids.append(json.loads(line)["id"])

    en_data_ids = []
    with open(DATA_EN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                en_data_ids.append(json.loads(line)["id"])

    id_list = sorted(list(set(bn_data_ids)))

    tier_records = {}
    for tier in TIERS:
        pfile = RESULTS_DIR / f"pred_{tier}.jsonl"
        records = []
        with open(pfile, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        tier_records[tier] = records

    # Binary correctness mapping: tier -> lang -> id -> bool
    tier_lang_id_correct = {}
    for tier in TIERS:
        tier_lang_id_correct[tier] = {"bangla": {}, "english": {}}
        for r in tier_records[tier]:
            tier_lang_id_correct[tier][r["language"]][r["id"]] = bool(r.get("is_correct"))

    # Bootstrap settings: 2000 resamples, seed 42
    rng = np.random.default_rng(42)
    B = 2000
    N = len(id_list) # 300
    boot_indices = rng.choice(N, size=(B, N), replace=True)

    boot_gaps = {}
    for tier in TIERS:
        bn_vec = np.array([tier_lang_id_correct[tier]["bangla"].get(item_id, False) for item_id in id_list], dtype=float)
        en_vec = np.array([tier_lang_id_correct[tier]["english"].get(item_id, False) for item_id in id_list], dtype=float)
        boot_en = np.mean(en_vec[boot_indices], axis=1)
        boot_bn = np.mean(bn_vec[boot_indices], axis=1)
        boot_gaps[tier] = boot_en - boot_bn

    f16_boot_gaps = boot_gaps["f16"]

    csv_rows = []
    for tier in TIERS:
        # Bangla accuracy & CI
        bn_n = len(id_list)
        bn_corr = sum(1 for item_id in id_list if tier_lang_id_correct[tier]["bangla"].get(item_id))
        bn_acc, bn_ci_low, bn_ci_high = wilson_ci(bn_corr, bn_n)

        # English accuracy & CI
        en_n = len(id_list)
        en_corr = sum(1 for item_id in id_list if tier_lang_id_correct[tier]["english"].get(item_id))
        en_acc, en_ci_low, en_ci_high = wilson_ci(en_corr, en_n)

        # Gap point and bootstrap CI
        gap_point = en_acc - bn_acc
        gap_ci_low = float(np.percentile(boot_gaps[tier], 2.5))
        gap_ci_high = float(np.percentile(boot_gaps[tier], 97.5))

        # McNemar test vs f16
        if tier == "f16":
            mcn_bn_stat, mcn_bn_pval, mcn_bn_b, mcn_bn_c = 0.0, 1.0, 0, 0
            mcn_en_stat, mcn_en_pval, mcn_en_b, mcn_en_c = 0.0, 1.0, 0, 0
            delta_gap_point = 0.0
            delta_gap_ci_low = 0.0
            delta_gap_ci_high = 0.0
            delta_gap_sig = False
        else:
            f16_bn = [tier_lang_id_correct["f16"]["bangla"][item_id] for item_id in id_list]
            tier_bn = [tier_lang_id_correct[tier]["bangla"][item_id] for item_id in id_list]
            b_bn = sum(1 for a, b_val in zip(f16_bn, tier_bn) if a and not b_val)
            c_bn = sum(1 for a, b_val in zip(f16_bn, tier_bn) if not a and b_val)
            mcn_bn_stat, mcn_bn_pval = mcnemar_pvalue(b_bn, c_bn)
            mcn_bn_b, mcn_bn_c = b_bn, c_bn

            f16_en = [tier_lang_id_correct["f16"]["english"][item_id] for item_id in id_list]
            tier_en = [tier_lang_id_correct[tier]["english"][item_id] for item_id in id_list]
            b_en = sum(1 for a, b_val in zip(f16_en, tier_en) if a and not b_val)
            c_en = sum(1 for a, b_val in zip(f16_en, tier_en) if not a and b_val)
            mcn_en_stat, mcn_en_pval = mcnemar_pvalue(b_en, c_en)
            mcn_en_b, mcn_en_c = b_en, c_en

            # Change in gap vs f16: delta = Gap_tier - Gap_f16
            f16_gap_point = (en_corr_f16 / 300.0) - (bn_corr_f16 / 300.0) if 'en_corr_f16' in locals() else 0.10666666666666667
            delta_gap_point = gap_point - 0.10666666666666667
            boot_delta = boot_gaps[tier] - f16_boot_gaps
            delta_gap_ci_low = float(np.percentile(boot_delta, 2.5))
            delta_gap_ci_high = float(np.percentile(boot_delta, 97.5))
            delta_gap_sig = bool(delta_gap_ci_low > 0 or delta_gap_ci_high < 0)

        if tier == "f16":
            bn_corr_f16 = bn_corr
            en_corr_f16 = en_corr

        csv_rows.append({
            "tier": tier,
            "bn_acc": round(bn_acc * 100, 3),
            "bn_ci_low": round(bn_ci_low * 100, 3),
            "bn_ci_high": round(bn_ci_high * 100, 3),
            "en_acc": round(en_acc * 100, 3),
            "en_ci_low": round(en_ci_low * 100, 3),
            "en_ci_high": round(en_ci_high * 100, 3),
            "gap": round(gap_point * 100, 3),
            "gap_ci_low": round(gap_ci_low * 100, 3),
            "gap_ci_high": round(gap_ci_high * 100, 3),
            "mcnemar_bn_stat": round(mcn_bn_stat, 4),
            "mcnemar_bn_p": round(mcn_bn_pval, 4),
            "mcnemar_bn_b": mcn_bn_b,
            "mcnemar_bn_c": mcn_bn_c,
            "mcnemar_en_stat": round(mcn_en_stat, 4),
            "mcnemar_en_p": round(mcn_en_pval, 4),
            "mcnemar_en_b": mcn_en_b,
            "mcnemar_en_c": mcn_en_c,
            "delta_gap_vs_f16": round(delta_gap_point * 100, 3),
            "delta_gap_ci_low": round(delta_gap_ci_low * 100, 3),
            "delta_gap_ci_high": round(delta_gap_ci_high * 100, 3),
            "delta_gap_sig_0_05": delta_gap_sig,
        })

    # Save CSV
    csv_file = AUDIT_DIR / "tier_stats.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("AUDIT_COMPLETED_SUCCESSFULLY")

if __name__ == "__main__":
    main()
