"""
Degradation Analysis and Publication Figure Generator.

Analyzes cross-lingual degradation across 16-bit, 8-bit, and 4-bit precisions.
Tests for degradation thresholds/cliffs and plots:
  1. Hallucination Rate vs Bit-Width (Bangla vs English comparison)
  2. Hallucination Taxonomy Distribution (Stacked bar chart)

Usage:
  python scripts/plot_degradation_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONSENSUS = ROOT / "results" / "human_eval_consensus.csv"
DEFAULT_AUTOMATED = ROOT / "results" / "automated_metrics.json"
DEFAULT_PREDICTIONS = ROOT / "results" / "eval_predictions.jsonl"
DEFAULT_FIGURE_DIR = ROOT / "results" / "figures"
DEFAULT_SUMMARY_JSON = ROOT / "results" / "degradation_summary.json"


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_data(
    consensus_path: Path,
    predictions_path: Path,
    automated_path: Path,
) -> tuple[dict[tuple[str, str], dict], str]:
    """Loads evaluation data, preferring human consensus with automated fallback."""
    stats = defaultdict(lambda: {
        "total": 0,
        "hallucinations": 0,
        "score_sum": 0.0,
        "error_cats": defaultdict(int),
    })

    if consensus_path.exists():
        with consensus_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                for r in rows:
                    lang = r["language"].lower()
                    prec = r["precision"].lower()
                    key = (lang, prec)
                    score = float(r.get("consensus_score", 0.0))
                    is_hal = r.get("consensus_hallucination", "NO") == "YES"
                    cat = r.get("consensus_error_category", "NONE")

                    stats[key]["total"] += 1
                    if is_hal:
                        stats[key]["hallucinations"] += 1
                    stats[key]["score_sum"] += score
                    stats[key]["error_cats"][cat] += 1
                return stats, "human_consensus"

    # Fallback to predictions file if human consensus is not yet completed
    if predictions_path.exists():
        from scripts.score_hallucinations import score_predictions

        scored_records, auto_summary = score_predictions(predictions_path)
        for r in scored_records:
            lang = r["lang"].lower()
            prec = r["precision"].lower()
            key = (lang, prec)
            f1 = r.get("token_f1", 0.0)
            purity = r.get("script_purity", 1.0)
            rep = r.get("is_repetitive", False)

            # Heuristic for automated hallucination detection
            is_hal = f1 < 0.35 or purity < 0.7 or rep
            cat = "NONE"
            if is_hal:
                if rep:
                    cat = "INCOHERENT"
                elif purity < 0.7:
                    cat = "CODE_MIX_TOK"
                else:
                    cat = "FACT_SUB"

            stats[key]["total"] += 1
            if is_hal:
                stats[key]["hallucinations"] += 1
            stats[key]["score_sum"] += f1 * 2.0  # scale to 0-2
            stats[key]["error_cats"][cat] += 1
        return stats, "automated_heuristic"

    return {}, "none"


def compute_metrics(stats: dict[tuple[str, str], dict]) -> dict:
    summary = {}
    precisions = ["16bit", "8bit", "4bit"]
    languages = ["bangla", "english"]

    for lang in languages:
        summary[lang] = {}
        for p in precisions:
            cell = stats.get((lang, p), {"total": 0, "hallucinations": 0, "score_sum": 0.0, "error_cats": {}})
            total = cell["total"]
            hals = cell["hallucinations"]
            hal_rate = (hals / total) if total > 0 else 0.0
            avg_score = (cell["score_sum"] / total) if total > 0 else 0.0

            summary[lang][p] = {
                "total": total,
                "hallucinations": hals,
                "hallucination_rate": round(hal_rate, 4),
                "avg_score": round(avg_score, 4),
                "error_categories": dict(cell["error_cats"]),
            }

        # Calculate degradation slopes & check for threshold cliff
        r_16 = summary[lang]["16bit"]["hallucination_rate"]
        r_8 = summary[lang]["8bit"]["hallucination_rate"]
        r_4 = summary[lang]["4bit"]["hallucination_rate"]

        delta_16_to_8 = r_8 - r_16
        delta_8_to_4 = r_4 - r_8

        cliff = delta_8_to_4 > (1.5 * max(delta_16_to_8, 0.05))
        summary[lang]["degradation_analysis"] = {
            "delta_16_to_8": round(delta_16_to_8, 4),
            "delta_8_to_4": round(delta_8_to_4, 4),
            "cliff_at_4bit_detected": cliff,
        }

    return summary


def plot_charts(summary: dict, output_dir: Path, data_source: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping graphical plot generation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    precisions = ["16bit", "8bit", "4bit"]
    x = [16, 8, 4]

    # Chart 1: Degradation Curves (Hallucination Rate vs Precision)
    plt.figure(figsize=(8, 5))
    bn_rates = [summary["bangla"][p]["hallucination_rate"] * 100 for p in precisions]
    en_rates = [summary["english"][p]["hallucination_rate"] * 100 for p in precisions]

    plt.plot(x, bn_rates, marker="o", linewidth=2.5, color="#D9381E", label="Bangla (Low-Resource)")
    plt.plot(x, en_rates, marker="s", linewidth=2.5, color="#1E88E5", label="English (High-Resource)")

    plt.gca().invert_xaxis()
    plt.xticks(x, ["16-bit (FP16)", "8-bit (INT8)", "4-bit (INT4)"], fontsize=11)
    plt.xlabel("Model Precision (Bit-Width)", fontsize=12, fontweight="bold")
    plt.ylabel("Hallucination Rate (%)", fontsize=12, fontweight="bold")
    plt.title(f"Quantization Degradation Curves: Bangla vs English\n(Source: {data_source.replace('_', ' ').title()})", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    curve_path = output_dir / "degradation_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"✓ Saved degradation curves chart -> {curve_path}")

    # Chart 2: Error Taxonomy Breakdown (Grouped stacked bars)
    cats = ["FACT_SUB", "FAB_ENT", "CODE_MIX_TOK", "INCOHERENT", "UNGROUNDED_REFUSAL"]
    cat_labels = {
        "FACT_SUB": "Factual Sub.",
        "FAB_ENT": "Fabricated Ent.",
        "CODE_MIX_TOK": "Code-Mix / Token",
        "INCOHERENT": "Incoherent",
        "UNGROUNDED_REFUSAL": "Refusal",
    }
    cat_colors = ["#F4511E", "#FB8C00", "#7CB342", "#8E24AA", "#546E7A"]

    plt.figure(figsize=(10, 5))
    bar_positions = []
    labels = []
    pos = 0
    for lang in ["bangla", "english"]:
        for p in precisions:
            bar_positions.append(pos)
            labels.append(f"{lang[:2].upper()}-{p}")
            pos += 1
        pos += 0.5  # gap between languages

    bottoms = [0] * len(bar_positions)
    for cat_idx, cat in enumerate(cats):
        values = []
        idx = 0
        for lang in ["bangla", "english"]:
            for p in precisions:
                errs = summary[lang][p]["error_categories"]
                values.append(errs.get(cat, 0))
                idx += 1
        plt.bar(bar_positions, values, bottom=bottoms, color=cat_colors[cat_idx], label=cat_labels[cat], width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    plt.xticks(bar_positions, labels, fontsize=10)
    plt.xlabel("Language & Precision Level", fontsize=12, fontweight="bold")
    plt.ylabel("Hallucination Count", fontsize=12, fontweight="bold")
    plt.title("Hallucination Taxonomy Breakdown Across Precisions", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    tax_path = output_dir / "error_taxonomy.png"
    plt.savefig(tax_path, dpi=300)
    plt.close()
    print(f"✓ Saved error taxonomy chart -> {tax_path}")


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    p = argparse.ArgumentParser(description="Generate degradation analysis and figures")
    p.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--automated", type=Path, default=DEFAULT_AUTOMATED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    p.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = p.parse_args(argv)

    stats, source = load_data(args.consensus, args.predictions, args.automated)
    if source == "none" or not stats:
        print("No evaluation data found. Run evaluate_models.py and score_hallucinations.py first.")
        return 1

    print(f"Analyzing degradation patterns using data source: {source} ...")
    summary = compute_metrics(stats)

    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved statistical summary -> {args.summary_file}")

    print("\nDegradation Analysis Summary:")
    for lang in ["bangla", "english"]:
        deg = summary[lang]["degradation_analysis"]
        print(f"\n[{lang.upper()}]:")
        for p in ["16bit", "8bit", "4bit"]:
            r = summary[lang][p]["hallucination_rate"] * 100
            score = summary[lang][p]["avg_score"]
            print(f"  {p:6s}: Hallucination Rate = {r:5.1f}% | Avg Score = {score:.2f}")
        print(f"  Drop 16->8: {deg['delta_16_to_8']*100:+.1f}% | Drop 8->4: {deg['delta_8_to_4']*100:+.1f}%")
        print(f"  Degradation Cliff Detected at 4-bit: {'YES' if deg['cliff_at_4bit_detected'] else 'NO'}")

    plot_charts(summary, args.output_dir, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
