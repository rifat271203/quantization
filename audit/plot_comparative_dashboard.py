# -*- coding: utf-8 -*-
"""Generate Publication-Ready 4-Panel Research Dashboard: Gemma 3 1B vs Llama 3.2 1B."""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
AUDIT_DIR = ROOT / "audit"
OUT_FIG = AUDIT_DIR / "gemma3_vs_llama32_comparative_dashboard.png"

TIERS_ORDER = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
TIER_BPW = [16.0, 8.5, 6.5, 5.5, 4.5, 3.5, 2.5]

def main():
    # 1. Load Data
    gemma_acc = pd.read_csv(AUDIT_DIR / "full900_tier_stats.csv")
    gemma_flip = pd.read_csv(AUDIT_DIR / "flip_analysis.csv")
    
    llama_acc = pd.read_csv(AUDIT_DIR / "llama32_tier_stats.csv")
    llama_flip = pd.read_csv(AUDIT_DIR / "llama32_flip_analysis.csv")
    llama_ppl = pd.read_csv(AUDIT_DIR / "llama32_perplexity_normalised.csv")

    # Set publication aesthetic
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)
    x = np.arange(len(TIERS_ORDER))

    # Color Palette
    c_gemma_en = "#1d4ed8"  # Royal blue
    c_gemma_bn = "#059669"  # Emerald green
    c_llama_en = "#b45309"  # Amber brown
    c_llama_bn = "#dc2626"  # Crimson red

    # -------------------------------------------------------------------------
    # Panel (a): Multi-Model Factual Accuracy Degradation Curves
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    g_bn_map = {r["tier"]: r["accuracy_pct"] for _, r in gemma_acc[gemma_acc["language"] == "bangla"].iterrows()}
    g_en_map = {r["tier"]: r["accuracy_pct"] for _, r in gemma_acc[gemma_acc["language"] == "english"].iterrows()}
    l_bn_map = {r["tier"]: r["accuracy_pct"] for _, r in llama_acc[llama_acc["language"] == "bangla"].iterrows()}
    l_en_map = {r["tier"]: r["accuracy_pct"] for _, r in llama_acc[llama_acc["language"] == "english"].iterrows()}

    g_bn_acc = [g_bn_map[t] for t in TIERS_ORDER]
    g_en_acc = [g_en_map[t] for t in TIERS_ORDER]
    l_bn_acc = [l_bn_map[t] for t in TIERS_ORDER]
    l_en_acc = [l_en_map[t] for t in TIERS_ORDER]

    ax_a.plot(x, g_en_acc, marker="o", markersize=8, linewidth=2.5, color=c_gemma_en, label="Gemma 3 1B — English (256k vocab)")
    ax_a.plot(x, g_bn_acc, marker="^", markersize=8, linewidth=2.5, color=c_gemma_bn, label="Gemma 3 1B — Bangla (256k vocab)")
    ax_a.plot(x, l_en_acc, marker="s", markersize=8, linewidth=2.5, linestyle="--", color=c_llama_en, label="Llama 3.2 1B — English (128k vocab)")
    ax_a.plot(x, l_bn_acc, marker="D", markersize=8, linewidth=2.5, linestyle="--", color=c_llama_bn, label="Llama 3.2 1B — Bangla (128k vocab)")

    ax_a.axhline(25.0, color="#6b7280", linestyle=":", linewidth=1.5, label="Random Guess (25.0%)")
    
    # Highlight Llama 2-bit collapse
    ax_a.annotate("Llama 3.2 Collapse\n(BN: 23.2% | EN: 24.6%)",
                  xy=(6, 23.9), xytext=(4.2, 21.0),
                  arrowprops=dict(facecolor=c_llama_bn, shrink=0.08, width=1.5, headwidth=7),
                  fontsize=10, weight="bold", color=c_llama_bn,
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec=c_llama_bn, lw=1))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{t}\n({bpw}b)" for t, bpw in zip(TIERS_ORDER, TIER_BPW)])
    ax_a.set_ylabel("Factual Accuracy (%)")
    ax_a.set_ylim(18, 55)
    ax_a.set_title("(a) Factual Grounding Accuracy Across 7 Quantization Tiers (N=900 Pairs)")
    ax_a.legend(loc="upper right", frameon=True, fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel (b): Cross-Lingual Factual Gap (EN Acc - BN Acc)
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    g_gap = [g_en_acc[i] - g_bn_acc[i] for i in range(len(TIERS_ORDER))]
    l_gap = [l_en_acc[i] - l_bn_acc[i] for i in range(len(TIERS_ORDER))]

    width = 0.35
    b1 = ax_b.bar(x - width/2, g_gap, width, label="Gemma 3 1B Cross-Lingual Gap", color=c_gemma_en, alpha=0.85, edgecolor="black", linewidth=0.8)
    b2 = ax_b.bar(x + width/2, l_gap, width, label="Llama 3.2 1B Cross-Lingual Gap", color=c_llama_bn, alpha=0.85, edgecolor="black", linewidth=0.8)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{t}\n({bpw}b)" for t, bpw in zip(TIERS_ORDER, TIER_BPW)])
    ax_b.set_ylabel("Accuracy Gap: EN - BN (%)")
    ax_b.set_ylim(0, 18)
    ax_b.set_title("(b) Cross-Lingual Grounding Disparity Gap (EN minus BN)")
    ax_b.legend(loc="upper right", frameon=True)

    for i, (gv, lv) in enumerate(zip(g_gap, l_gap)):
        ax_b.text(i - width/2, gv + 0.3, f"{gv:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")
        ax_b.text(i + width/2, lv + 0.3, f"{lv:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")

    # -------------------------------------------------------------------------
    # Panel (c): Perplexity Explosion Comparison (Language Modeling Degradation)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    # PPL Tiers: f16, Q4_K_M, Q3_K_M, Q2_K
    ppl_tiers = ["f16", "Q4_K_M", "Q3_K_M", "Q2_K"]
    ppl_x = np.arange(len(ppl_tiers))

    # Gemma 3 empirical relative PPL increase (%) on Belebele Part 1
    # f16: 0%, Q4: +10.88%, Q3: +22.93%, Q2: +82.61%
    g_ppl_inc_bn = [0.0, 10.88, 22.93, 82.61]
    g_ppl_inc_en = [0.0, 2.59, 8.07, 32.54]

    # Llama 3.2 empirical relative PPL increase (%) on Belebele Part 1
    # f16: 0%, Q4: +2.50%, Q3: +10.41%, Q2: +150.99%
    l_ppl_inc_bn = [0.0, 2.50, 10.41, 150.99]
    l_ppl_inc_en = [0.0, 1.61, 6.19, 53.95]

    ax_c.plot(ppl_x, l_ppl_inc_bn, marker="D", markersize=9, linewidth=3.0, linestyle="--", color=c_llama_bn, label="Llama 3.2 1B — Bangla (+151.0% at Q2)")
    ax_c.plot(ppl_x, g_ppl_inc_bn, marker="^", markersize=9, linewidth=3.0, color=c_gemma_bn, label="Gemma 3 1B — Bangla (+82.6% at Q2)")
    ax_c.plot(ppl_x, l_ppl_inc_en, marker="s", markersize=7, linewidth=2.0, linestyle="--", color=c_llama_en, label="Llama 3.2 1B — English (+54.0% at Q2)")
    ax_c.plot(ppl_x, g_ppl_inc_en, marker="o", markersize=7, linewidth=2.0, color=c_gemma_en, label="Gemma 3 1B — English (+32.5% at Q2)")

    ax_c.annotate("Llama 3.2 Catastrophic PPL Explosion\n(+151.0% vs Gemma's +82.6%)",
                  xy=(3, 150.99), xytext=(1.3, 115.0),
                  arrowprops=dict(facecolor=c_llama_bn, shrink=0.08, width=1.5, headwidth=7),
                  fontsize=10, weight="bold", color=c_llama_bn,
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec=c_llama_bn, lw=1))

    ax_c.set_xticks(ppl_x)
    bpw_dict = dict(zip(TIERS_ORDER, TIER_BPW))
    ax_c.set_xticklabels([f"{t}\n({bpw_dict[t]}b)" for t in ppl_tiers])
    ax_c.set_ylabel("Perplexity Increase vs f16 (%)")
    ax_c.set_ylim(-5, 175)
    ax_c.set_title("(c) Language Modeling Perplexity Degradation (Belebele Benchmark)")
    ax_c.legend(loc="upper left", frameon=True, fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel (d): Latent Decision Instability (Decision Flip Rate from f16 Baseline)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    g_bn_flip = [r["flip_rate_overall"] * 100 for _, r in gemma_flip[gemma_flip["language"] == "bangla"].iterrows()]
    g_en_flip = [r["flip_rate_overall"] * 100 for _, r in gemma_flip[gemma_flip["language"] == "english"].iterrows()]
    l_bn_flip = [r["flip_rate_overall"] * 100 for _, r in llama_flip[llama_flip["language"] == "bangla"].iterrows()]
    l_en_flip = [r["flip_rate_overall"] * 100 for _, r in llama_flip[llama_flip["language"] == "english"].iterrows()]

    ax_d.plot(x, l_bn_flip, marker="D", markersize=8, linewidth=2.5, linestyle="--", color=c_llama_bn, label="Llama 3.2 1B — Bangla Flips (69.7% at Q2)")
    ax_d.plot(x, l_en_flip, marker="s", markersize=8, linewidth=2.5, linestyle="--", color=c_llama_en, label="Llama 3.2 1B — English Flips (51.6% at Q2)")
    ax_d.plot(x, g_bn_flip, marker="^", markersize=8, linewidth=2.5, color=c_gemma_bn, label="Gemma 3 1B — Bangla Flips (46.9% at Q2)")
    ax_d.plot(x, g_en_flip, marker="o", markersize=8, linewidth=2.5, color=c_gemma_en, label="Gemma 3 1B — English Flips (37.1% at Q2)")

    ax_d.annotate("70% of Decisions Flipped\nfrom f16 in Llama Bangla",
                  xy=(6, 69.67), xytext=(4.3, 76.0),
                  arrowprops=dict(facecolor=c_llama_bn, shrink=0.08, width=1.5, headwidth=7),
                  fontsize=10, weight="bold", color=c_llama_bn,
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec=c_llama_bn, lw=1))

    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{t}\n({bpw}b)" for t, bpw in zip(TIERS_ORDER, TIER_BPW)])
    ax_d.set_ylabel("Decision Flip Rate vs f16 (%)")
    ax_d.set_ylim(-2, 85)
    ax_d.set_title("(d) Latent Prediction Instability: Instance-Level Decision Flips vs f16")
    ax_d.legend(loc="upper left", frameon=True, fontsize=9.5)

    plt.suptitle("Empirical Cross-Architectural Audit: Tokenizer Shielding vs Quantization Collapse\nGoogle Gemma 3 1B (256k Morpheme Vocab) vs Meta Llama 3.2 1B (128k Subword Vocab)", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Publication-ready dashboard saved to: {OUT_FIG}")

if __name__ == "__main__":
    main()
