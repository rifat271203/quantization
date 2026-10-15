# -*- coding: utf-8 -*-
"""Generate Publication-Ready 4-Panel Research Dashboard for Full 900-Item Evaluation."""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
AUDIT_DIR = ROOT / "audit"
OUT_FIG = AUDIT_DIR / "gemma3_full900_empirical_dashboard.png"

TIER_ORDER = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
TIER_BPW = [16.0, 8.5, 6.5, 5.5, 4.5, 3.5, 2.5]

def main():
    # 1. Load data
    df_acc = pd.read_csv(AUDIT_DIR / "full900_tier_stats.csv")
    df_flip = pd.read_csv(AUDIT_DIR / "flip_analysis.csv")
    df_ppl = pd.read_csv(AUDIT_DIR / "perplexity_normalised.csv")
    df_ent = pd.read_csv(AUDIT_DIR / "entropy_stats.csv")

    # Set publication styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    x = np.arange(len(TIER_ORDER))
    width = 0.35

    # Color tokens
    c_en = "#1d4ed8"  # Royal blue
    c_bn = "#dc2626"  # Crimson red
    c_gap = "#7c3aed" # Deep purple
    c_accent = "#d97706"

    # -------------------------------------------------------------
    # Panel (a): 900-Item Full Accuracy with Cluster Bootstrap CIs
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    bn_acc_map = {row["tier"]: row for _, row in df_acc[df_acc["language"] == "bangla"].iterrows()}
    en_acc_map = {row["tier"]: row for _, row in df_acc[df_acc["language"] == "english"].iterrows()}

    bn_accs = [bn_acc_map[t]["accuracy_pct"] for t in TIER_ORDER]
    en_accs = [en_acc_map[t]["accuracy_pct"] for t in TIER_ORDER]

    bn_err_l = [bn_acc_map[t]["accuracy_pct"] - bn_acc_map[t]["cluster_boot_ci_lower"]*100 for t in TIER_ORDER]
    bn_err_u = [bn_acc_map[t]["cluster_boot_ci_upper"]*100 - bn_acc_map[t]["accuracy_pct"] for t in TIER_ORDER]

    en_err_l = [en_acc_map[t]["accuracy_pct"] - en_acc_map[t]["cluster_boot_ci_lower"]*100 for t in TIER_ORDER]
    en_err_u = [en_acc_map[t]["cluster_boot_ci_upper"]*100 - en_acc_map[t]["accuracy_pct"] for t in TIER_ORDER]

    bars_en = ax_a.bar(x - width/2, en_accs, width, yerr=[en_err_l, en_err_u], label="English (EN)",
                       color=c_en, capsize=4, alpha=0.9, edgecolor="black", linewidth=0.8)
    bars_bn = ax_a.bar(x + width/2, bn_accs, width, yerr=[bn_err_l, bn_err_u], label="Bangla (BN)",
                       color=c_bn, capsize=4, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax_a.axhline(25.0, color="gray", linestyle="--", linewidth=1.2, label="Random Guess (25%)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{t}\n({bpw}b)" for t, bpw in zip(TIER_ORDER, TIER_BPW)])
    ax_a.set_ylabel("Factual Accuracy (%)")
    ax_a.set_ylim(20, 60)
    ax_a.set_title("(a) Full 900-Item Factual Accuracy (488-Passage Clustered 95% CI)")
    ax_a.legend(loc="upper right", frameon=True)

    # Annotate values
    for i, (en_v, bn_v) in enumerate(zip(en_accs, bn_accs)):
        ax_a.text(i - width/2, en_v + en_err_u[i] + 1.0, f"{en_v:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#1e3a8a")
        ax_a.text(i + width/2, bn_v + bn_err_u[i] + 1.0, f"{bn_v:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#991b1b")

    # -------------------------------------------------------------
    # Panel (b): Decision Flip Rate from f16 Baseline (%)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    bn_flip_map = {row["tier"]: row for _, row in df_flip[df_flip["language"] == "bangla"].iterrows()}
    en_flip_map = {row["tier"]: row for _, row in df_flip[df_flip["language"] == "english"].iterrows()}

    bn_flips = [bn_flip_map[t]["flip_rate_overall"] * 100 for t in TIER_ORDER]
    en_flips = [en_flip_map[t]["flip_rate_overall"] * 100 for t in TIER_ORDER]
    flip_gaps = [bn_flips[i] - en_flips[i] for i in range(len(TIER_ORDER))]

    ax_b.plot(x, bn_flips, marker="o", markersize=8, linewidth=2.5, color=c_bn, label="Bangla Flip Rate")
    ax_b.plot(x, en_flips, marker="s", markersize=8, linewidth=2.5, color=c_en, label="English Flip Rate")
    ax_b.fill_between(x, en_flips, bn_flips, color=c_gap, alpha=0.18, label="Language Instability Gap (BN > EN)")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{t}\n({bpw}b)" for t, bpw in zip(TIER_ORDER, TIER_BPW)])
    ax_b.set_ylabel("Decision Flip Rate vs f16 (%)")
    ax_b.set_ylim(-2, 55)
    ax_b.set_title("(b) Latent Prediction Instability: Decision Flips from f16 Baseline")
    ax_b.legend(loc="upper left", frameon=True)

    for i, (bn_f, en_f, gap) in enumerate(zip(bn_flips, en_flips, flip_gaps)):
        if i > 0:
            ax_b.annotate(f"+{gap:.1f}% gap", xy=(i, (bn_f + en_f)/2), xytext=(i, (bn_f + en_f)/2 + 3),
                          ha="center", fontsize=8.5, weight="bold", color=c_gap,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=c_gap, alpha=0.85))

    # -------------------------------------------------------------
    # Panel (c): Normalized Perplexity Relative Increase vs f16 (%)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    full_ppl = df_ppl[df_ppl["type"] == "full_corpus"]
    bn_ppl_map = {row["tier"]: row for _, row in full_ppl[full_ppl["language"] == "bangla"].iterrows()}
    en_ppl_map = {row["tier"]: row for _, row in full_ppl[full_ppl["language"] == "english"].iterrows()}

    bn_ppl_inc = [bn_ppl_map[t]["rel_ppl_inc_vs_f16_pct"] for t in TIER_ORDER]
    en_ppl_inc = [en_ppl_map[t]["rel_ppl_inc_vs_f16_pct"] for t in TIER_ORDER]

    # Plot lines with markers
    ax_c.plot(x, bn_ppl_inc, marker="^", markersize=9, linewidth=2.8, color=c_bn, label="Bangla PPL Increase (%)")
    ax_c.plot(x, en_ppl_inc, marker="v", markersize=9, linewidth=2.8, color=c_en, label="English PPL Increase (%)")

    # Add error bar/spread range from 3-part spread for Q4, Q3, Q2
    part_ppl = df_ppl[df_ppl["type"] == "part_spread"]
    for t_idx, t in enumerate(["Q4_K_M", "Q3_K_M", "Q2_K"]):
        x_pos = TIER_ORDER.index(t)
        bn_parts = part_ppl[(part_ppl["tier"] == t) & (part_ppl["language"] == "bangla")]["rel_ppl_inc_vs_f16_pct"].values
        en_parts = part_ppl[(part_ppl["tier"] == t) & (part_ppl["language"] == "english")]["rel_ppl_inc_vs_f16_pct"].values
        if len(bn_parts) > 0:
            ax_c.vlines(x_pos, min(bn_parts), max(bn_parts), color=c_bn, linestyle=":", linewidth=2, alpha=0.7)
            ax_c.scatter([x_pos]*len(bn_parts), bn_parts, color=c_bn, s=25, alpha=0.6, zorder=4)
        if len(en_parts) > 0:
            ax_c.vlines(x_pos, min(en_parts), max(en_parts), color=c_en, linestyle=":", linewidth=2, alpha=0.7)
            ax_c.scatter([x_pos]*len(en_parts), en_parts, color=c_en, s=25, alpha=0.6, zorder=4)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{t}\n({bpw}b)" for t, bpw in zip(TIER_ORDER, TIER_BPW)])
    ax_c.set_ylabel("Perplexity Increase vs f16 (%)")
    ax_c.set_ylim(-5, 85)
    ax_c.set_title("(c) Language Modeling Degradation: Perplexity Increase vs f16")
    ax_c.legend(loc="upper left", frameon=True)

    # Highlight ratio at Q4 and Q3
    ax_c.text(4, bn_ppl_inc[4] + 4, f"BN: +{bn_ppl_inc[4]:.1f}%\n(4.3x of EN)", ha="center", fontsize=8.5, weight="bold", color="#991b1b")
    ax_c.text(5, bn_ppl_inc[5] + 4, f"BN: +{bn_ppl_inc[5]:.1f}%\n(2.5x of EN)", ha="center", fontsize=8.5, weight="bold", color="#991b1b")
    ax_c.text(6, bn_ppl_inc[6] + 4, f"BN: +{bn_ppl_inc[6]:.1f}%\n(2.2x of EN)", ha="center", fontsize=8.5, weight="bold", color="#991b1b")

    # -------------------------------------------------------------
    # Panel (d): 4-Way Softmax Distribution Entropy (Bits)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    bn_ent_map = {row["tier"]: row for _, row in df_ent[df_ent["language"] == "bangla"].iterrows()}
    en_ent_map = {row["tier"]: row for _, row in df_ent[df_ent["language"] == "english"].iterrows()}

    bn_ents = [bn_ent_map[t]["mean_entropy_bits"] for t in TIER_ORDER]
    en_ents = [en_ent_map[t]["mean_entropy_bits"] for t in TIER_ORDER]

    ax_d.plot(x, bn_ents, marker="D", markersize=8, linewidth=2.5, color=c_bn, label="Bangla Entropy (bits)")
    ax_d.plot(x, en_ents, marker="P", markersize=8, linewidth=2.5, color=c_en, label="English Entropy (bits)")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{t}\n({bpw}b)" for t, bpw in zip(TIER_ORDER, TIER_BPW)])
    ax_d.set_ylabel("Mean Softmax Entropy (bits)")
    ax_d.set_ylim(0.04, 0.30)
    ax_d.set_title("(d) Predictive Uncertainty: 4-Way Candidate Softmax Entropy")
    ax_d.legend(loc="upper left", frameon=True)

    for i in range(len(TIER_ORDER)):
        if i >= 4:
            ax_d.text(i, bn_ents[i] + 0.012, f"{bn_ents[i]:.3f}", ha="center", fontsize=8.5, weight="bold", color="#991b1b")
            ax_d.text(i, en_ents[i] - 0.016, f"{en_ents[i]:.3f}", ha="center", fontsize=8.5, weight="bold", color="#1e3a8a")

    plt.suptitle("Gemma 3 1B IT — Multi-Metric Empirical Quantization Degradation Audit (N = 900)", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Successfully saved 4-panel empirical research dashboard to: {OUT_FIG}")

if __name__ == "__main__":
    main()
