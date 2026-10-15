# -*- coding: utf-8 -*-
"""Plot 7-Tier Quantization Degradation Curves for Logprob Scoring Run."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
STATS_CSV = ROOT / "audit" / "logprob_tier_stats.csv"
OUT_PLOT = RESULTS_DIR / "gemma3_7tier_logprob_degradation_curve.png"

TIERS = [
    ("f16", 16.0),
    ("Q8_0", 8.5),
    ("Q6_K", 6.5),
    ("Q5_K_M", 5.5),
    ("Q4_K_M", 4.5),
    ("Q3_K_M", 3.5),
    ("Q2_K", 2.5),
]

# Exact stats from audit/logprob_tier_stats.csv
data = {
    "f16":    {"bn_acc": 32.33, "bn_ci": (27.29, 37.82), "en_acc": 47.33, "en_ci": (41.75, 52.98), "gap": 15.00, "gap_ci": (8.33, 21.67)},
    "Q8_0":   {"bn_acc": 33.00, "bn_ci": (27.92, 38.51), "en_acc": 45.67, "en_ci": (40.12, 51.32), "gap": 12.67, "gap_ci": (6.33, 19.00)},
    "Q6_K":   {"bn_acc": 32.00, "bn_ci": (26.98, 37.48), "en_acc": 46.67, "en_ci": (41.10, 52.32), "gap": 14.67, "gap_ci": (8.33, 20.67)},
    "Q5_K_M": {"bn_acc": 34.33, "bn_ci": (29.19, 39.87), "en_acc": 47.33, "en_ci": (41.75, 52.98), "gap": 13.00, "gap_ci": (6.67, 19.67)},
    "Q4_K_M": {"bn_acc": 32.67, "bn_ci": (27.61, 38.16), "en_acc": 44.67, "en_ci": (39.14, 50.32), "gap": 12.00, "gap_ci": (5.66, 18.33)},
    "Q3_K_M": {"bn_acc": 30.33, "bn_ci": (25.41, 35.76), "en_acc": 48.00, "en_ci": (42.41, 53.64), "gap": 17.67, "gap_ci": (11.33, 24.00)},
    "Q2_K":   {"bn_acc": 33.67, "bn_ci": (28.56, 39.19), "en_acc": 45.33, "en_ci": (39.79, 50.99), "gap": 11.67, "gap_ci": (5.00, 18.34)},
}

def main():
    names = [t[0] for t in TIERS]
    bpws = [t[1] for t in TIERS]
    
    bn_accs = [data[n]["bn_acc"] for n in names]
    bn_err_l = [data[n]["bn_acc"] - data[n]["bn_ci"][0] for n in names]
    bn_err_u = [data[n]["bn_ci"][1] - data[n]["bn_acc"] for n in names]
    
    en_accs = [data[n]["en_acc"] for n in names]
    en_err_l = [data[n]["en_acc"] - data[n]["en_ci"][0] for n in names]
    en_err_u = [data[n]["en_ci"][1] - data[n]["en_acc"] for n in names]
    
    gaps = [data[n]["gap"] for n in names]
    gap_err_l = [data[n]["gap"] - data[n]["gap_ci"][0] for n in names]
    gap_err_u = [data[n]["gap_ci"][1] - data[n]["gap"] for n in names]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Accuracy by Tier
    x = np.arange(len(names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, en_accs, width, yerr=[en_err_l, en_err_u], label="English (EN)", color="#2563eb", capsize=4, alpha=0.9)
    rects2 = ax1.bar(x + width/2, bn_accs, width, yerr=[bn_err_l, bn_err_u], label="Bangla (BN)", color="#dc2626", capsize=4, alpha=0.9)

    ax1.axhline(25.0, color="#64748b", linestyle="--", linewidth=1.2, label="Random Guess (25%)")
    ax1.set_title("Gemma 3 1B IT: Factual Grounding Accuracy (Logprob Scoring)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Quantization Tier (Decreasing Precision →)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Accuracy (%) [Wilson 95% CI]", fontsize=11, fontweight="semibold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n({b:.1f} bpw)" for n, b in zip(names, bpws)], fontsize=9.5)
    ax1.set_ylim(15, 60)
    ax1.legend(frameon=True, facecolor="white", loc="upper left")

    # Values on bars
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h + 3.8), xytext=(0, 0),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h + 3.8), xytext=(0, 0),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991b1b")

    # Panel 2: Cross-Lingual Disparity Gap (EN - BN)
    ax2.errorbar(x, gaps, yerr=[gap_err_l, gap_err_u], fmt="-o", color="#7c3aed", ecolor="#a78bfa",
                 elinewidth=2, capsize=5, capthick=1.5, markersize=8, linewidth=2.5, label="Observed Gap (EN - BN)")
    ax2.axhline(0.0, color="#000000", linestyle="-", linewidth=1)
    ax2.axhspan(min(gaps)-2, max(gaps)+2, color="#f5f3ff", alpha=0.6, label="Disparity Band (+11.7% to +17.7%)")

    ax2.set_title("Cross-Lingual Gap: English Advantage (Bootstrap 95% CI)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Quantization Tier (Decreasing Precision →)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Gap: English Acc - Bangla Acc (Δ %)", fontsize=11, fontweight="semibold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n}\n({b:.1f} bpw)" for n, b in zip(names, bpws)], fontsize=9.5)
    ax2.set_ylim(0, 30)
    ax2.legend(frameon=True, facecolor="white", loc="upper left")

    for i, g in enumerate(gaps):
        ax2.annotate(f"+{g:.1f}%", xy=(x[i], g + 1.2), ha="center", fontsize=9.5, fontweight="bold", color="#5b21b6")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    print(f"Plot saved successfully to: {OUT_PLOT}")

if __name__ == "__main__":
    main()
