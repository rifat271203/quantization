"""
Inter-Rater Reliability and Human Annotation Consensus Calculator.

Ingests human evaluation spreadsheets, unblinds model identities using blind_key.json,
computes inter-rater agreement (Cohen's Kappa & Percent Agreement),
and resolves consensus ratings for the final study.

Usage:
  python scripts/calculate_inter_rater_agreement.py --input review/human_eval_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHEET = ROOT / "review" / "human_eval_sheet.csv"
DEFAULT_BLIND_KEY = ROOT / "review" / "blind_key.json"
DEFAULT_OUTPUT = ROOT / "results" / "human_eval_consensus.csv"


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def compute_cohens_kappa(ratings1: list[str], ratings2: list[str], categories: list[str]) -> float:
    """Calculates Cohen's Kappa for categorical agreement between two raters."""
    if len(ratings1) != len(ratings2) or not ratings1:
        return 0.0

    n = len(ratings1)
    # Observed agreement
    agree = sum(1 for r1, r2 in zip(ratings1, ratings2) if r1 == r2)
    po = agree / n

    # Expected chance agreement
    c1 = Counter(ratings1)
    c2 = Counter(ratings2)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in categories)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def process_evaluations(
    sheet_paths: list[Path],
    blind_key_path: Path,
    output_path: Path,
) -> None:
    _configure_stdout()

    blind_key = {}
    if blind_key_path.exists():
        with blind_key_path.open(encoding="utf-8") as f:
            blind_key = json.load(f)

    # Key: (eval_id, blinded_model) -> list of rater entries
    samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    raters: set[str] = set()

    for path in sheet_paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = (row.get("correctness_score_0_to_2") or "").strip()
                if not score:
                    continue
                rater = (row.get("evaluator_name") or "rater_1").strip()
                raters.add(rater)
                key = (row["eval_id"], row["blinded_model"])
                samples[key].append({
                    "rater": rater,
                    "score": score,
                    "error_cat": (row.get("error_category") or "NONE").strip().upper(),
                    "row": row,
                })

    if not samples:
        print("No completed ratings found in provided sheets.")
        print(f"Please fill out {DEFAULT_SHEET} with scores (0, 1, 2) and error categories.")
        return

    print(f"Found {len(samples)} evaluated sample instances from {len(raters)} rater(s): {', '.join(sorted(raters))}")

    # If multiple raters exist, compute pairwise Kappa
    rater_list = sorted(raters)
    if len(rater_list) >= 2:
        print("\n--- Inter-Rater Reliability (Pairwise Cohen's Kappa) ---")
        for i in range(len(rater_list)):
            for j in range(i + 1, len(rater_list)):
                r1, r2 = rater_list[i], rater_list[j]
                paired_scores1 = []
                paired_scores2 = []
                for sample_entries in samples.values():
                    e1 = next((e for e in sample_entries if e["rater"] == r1), None)
                    e2 = next((e for e in sample_entries if e["rater"] == r2), None)
                    if e1 and e2:
                        paired_scores1.append(e1["score"])
                        paired_scores2.append(e2["score"])
                if paired_scores1:
                    kappa = compute_cohens_kappa(paired_scores1, paired_scores2, ["0", "1", "2"])
                    agree_pct = sum(1 for a, b in zip(paired_scores1, paired_scores2) if a == b) / len(paired_scores1) * 100
                    print(f"  {r1} vs {r2}: N={len(paired_scores1)} | Agreement={agree_pct:.1f}% | Kappa={kappa:.3f}")

    # Build consensus dataset
    consensus_rows = []
    for (eval_id, blind_code), entries in sorted(samples.items()):
        base_info = entries[0]["row"]
        scores = [int(e["score"]) for e in entries if e["score"].isdigit()]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Majority error category
        cat_counts = Counter(e["error_cat"] for e in entries if e["error_cat"] and e["error_cat"] != "NONE")
        consensus_cat = cat_counts.most_common(1)[0][0] if cat_counts else "NONE"

        # Unblind metadata
        meta = blind_key.get(blind_code, {})
        precision = meta.get("precision", "unknown")
        adapter_dir = meta.get("adapter_dir", "")

        consensus_rows.append({
            "eval_id": eval_id,
            "language": base_info.get("language"),
            "domain": base_info.get("domain"),
            "precision": precision,
            "adapter_dir": adapter_dir,
            "blinded_model": blind_code,
            "question": base_info.get("question"),
            "ground_truth_answer": base_info.get("ground_truth_answer"),
            "model_generated_answer": base_info.get("model_generated_answer"),
            "consensus_score": round(avg_score, 2),
            "consensus_hallucination": "YES" if avg_score < 1.5 else "NO",
            "consensus_error_category": consensus_cat,
            "num_ratings": len(entries),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(consensus_rows[0].keys()))
        writer.writeheader()
        writer.writerows(consensus_rows)

    print(f"\n✓ Consensus evaluation dataset generated ({len(consensus_rows)} rows) -> {output_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Calculate inter-rater agreement and compile consensus")
    p.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[DEFAULT_SHEET],
        help=f"Human evaluation sheets (default: {DEFAULT_SHEET})",
    )
    p.add_argument(
        "--blind-key",
        type=Path,
        default=DEFAULT_BLIND_KEY,
        help=f"Blind key JSON (default: {DEFAULT_BLIND_KEY})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output consensus CSV (default: {DEFAULT_OUTPUT})",
    )
    args = p.parse_args(argv)
    process_evaluations(args.input, args.blind_key, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
