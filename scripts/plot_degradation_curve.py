# -*- coding: utf-8 -*-
"""Generate publication-ready 7-tier degradation curve plot for Gemma 3 1B IT."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
SUMMARY_FILE = RESULTS_DIR / "gguf_7tier_summary.json"
PLOT_FILE = RESULTS_DIR / "gemma3_7tier_degradation_curve.png"

with open(SUMMARY_FILE, encoding="utf-8") as f:
    data = json.load(f)

tiers = [d["tier"] for d in data]
bpw = [d["bpw"] for d in data]
bn_acc = [d["bangla"]["accuracy_pct"] for d in data]
en_acc = [d["english"]["accuracy_pct"] for d in data]
bn_hall = [d["bangla"]["hallucination_pct"] for d in data]
en_hall = [d["english"]["hallucination_pct"] for d in data]
gaps = [d["gap_pct"] for d in data]

# Modern academic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

x = np.arange(len(tiers))

# --- Plot 1: Factual Accuracy ---
ax1.plot(x, en_acc, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="English Accuracy")
ax1.plot(x, bn_acc, marker="s", color="#d62728", linewidth=2.5, markersize=8, label="Bangla Accuracy")
ax1.axhline(25.0, color="gray", linestyle="--", alpha=0.7, label="Random Guess (25%)")

# Fill gap between languages
ax1.fill_between(x, bn_acc, en_acc, color="#1f77b4", alpha=0.12, label="Cross-Lingual Disparity")

# Annotate peak gap at Q3_K_M
q3_idx = tiers.index("Q3_K_M")
ax1.annotate(
    f"Peak Gap: +{gaps[q3_idx]:.1f}%\n(EN 46.3% vs BN 32.3%)",
    xy=(q3_idx, (en_acc[q3_idx] + bn_acc[q3_idx]) / 2),
    xytext=(q3_idx - 1.2, 38),
    arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
    fontweight="bold",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#2ca02c", lw=1.5)
)

ax1.set_title("Factual Accuracy across Quantization Tiers\n(Gemma 3 1B IT on Belebele Parallel 300)", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Quantization Tier (Decreasing Precision →)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{t}\n({b} bpw)" for t, b in zip(tiers, bpw)], fontsize=9)
ax1.set_ylim(20, 55)
ax1.legend(loc="lower left", frameon=True)
ax1.grid(True, linestyle=":", alpha=0.6)

# --- Plot 2: Distractor Hallucination Rate ---
ax2.plot(x, bn_hall, marker="^", color="#d62728", linewidth=2.5, markersize=8, label="Bangla Hallucination Rate")
ax2.plot(x, en_hall, marker="v", color="#1f77b4", linewidth=2.5, markersize=8, label="English Hallucination Rate")

# Highlight 4-bit / 3-bit cliff
ax2.axvspan(3.5, 5.5, color="#ffecb3", alpha=0.35, label="Edge Deployment Zone (Q4–Q3)")

ax2.set_title("Ungrounded Distractor Hallucination Rate\n(Higher is Worse)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Quantization Tier (Decreasing Precision →)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Hallucination Rate (%)", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{t}\n({b} bpw)" for t, b in zip(tiers, bpw)], fontsize=9)
ax2.set_ylim(40, 75)
ax2.legend(loc="upper left", frameon=True)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
print(f"Publication-ready chart saved to: {PLOT_FILE}")
