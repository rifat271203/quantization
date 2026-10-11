"""
Automatic Scoring and Human Evaluation Sheet Generator.

Computes automated similarity metrics on model predictions and exports
a blinded evaluation spreadsheet for human annotators.

Usage:
  python scripts/score_hallucinations.py --predictions results/eval_predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_FILE = ROOT / "results" / "eval_predictions.jsonl"
DEFAULT_METRICS_FILE = ROOT / "results" / "automated_metrics.json"
DEFAULT_HUMAN_SHEET = ROOT / "review" / "human_eval_sheet.csv"
DEFAULT_BLIND_KEY = ROOT / "review" / "blind_key.json"


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def tokenize_words(text: str) -> list[str]:
    # Handles both Bangla script and English words/numbers
    tokens = re.findall(r"[\u0980-\u09FF]+|[a-zA-Z0-9]+", text.lower())
    return tokens


def compute_token_f1(pred: str, target: str) -> tuple[float, float, float]:
    pred_tokens = tokenize_words(pred)
    target_tokens = tokenize_words(target)
    if not pred_tokens or not target_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def check_script_purity(text: str, expected_lang: str) -> float:
    """Returns ratio of characters matching the expected language script."""
    if expected_lang == "bangla":
        matches = re.findall(r"[\u0980-\u09FF]", text)
    else:
        matches = re.findall(r"[a-zA-Z]", text)
    letters_only = re.findall(r"[\u0980-\u09FFa-zA-Z]", text)
    if not letters_only:
        return 1.0
    return len(matches) / len(letters_only)


def detect_repetition(tokens: list[str], n: int = 3) -> bool:
    """Detects degenerate repetition loops."""
    if len(tokens) < n * 2:
        return False
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    if any(c >= 4 for c in counts.values()):
        return True
    return False


def score_predictions(pred_path: Path) -> tuple[list[dict], dict]:
    records = []
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    scored_records = []
    # Group by (lang, precision)
    group_stats: dict[str, dict] = {}

    for r in records:
        precision = r["precision"]
        lang = r["lang"]
        group_key = f"{lang}_{precision}"

        if group_key not in group_stats:
            group_stats[group_key] = {
                "count": 0,
                "f1_sum": 0.0,
                "precision_sum": 0.0,
                "recall_sum": 0.0,
                "purity_sum": 0.0,
                "repetition_count": 0,
            }

        p, rec, f1 = compute_token_f1(r["prediction"], r["ground_truth"])
        purity = check_script_purity(r["prediction"], lang)
        tokens = tokenize_words(r["prediction"])
        is_repetitive = detect_repetition(tokens)

        r_scored = dict(r)
        r_scored["token_precision"] = round(p, 4)
        r_scored["token_recall"] = round(rec, 4)
        r_scored["token_f1"] = round(f1, 4)
        r_scored["script_purity"] = round(purity, 4)
        r_scored["is_repetitive"] = is_repetitive
        scored_records.append(r_scored)

        stats = group_stats[group_key]
        stats["count"] += 1
        stats["f1_sum"] += f1
        stats["precision_sum"] += p
        stats["recall_sum"] += rec
        stats["purity_sum"] += purity
        if is_repetitive:
            stats["repetition_count"] += 1

    summary = {}
    for g, s in group_stats.items():
        n = max(s["count"], 1)
        summary[g] = {
            "num_samples": s["count"],
            "avg_f1": round(s["f1_sum"] / n, 4),
            "avg_token_precision": round(s["precision_sum"] / n, 4),
            "avg_token_recall": round(s["recall_sum"] / n, 4),
            "avg_script_purity": round(s["purity_sum"] / n, 4),
            "repetition_rate": round(s["repetition_count"] / n, 4),
        }

    return scored_records, summary


def generate_blinded_human_sheet(
    scored_records: list[dict],
    sheet_path: Path,
    blind_key_path: Path,
    seed: int = 42,
) -> None:
    # Identify all distinct (model_id, adapter_dir, precision) configurations
    configs = sorted(
        list(
            {
                (r.get("model_id"), r.get("adapter_dir"), r["precision"])
                for r in scored_records
            }
        )
    )

    rng = random.Random(seed)
    # Assign blind codenames: Model-A, Model-B, etc.
    letters = [f"Model_{chr(65 + i)}" for i in range(len(configs))]
    rng.shuffle(letters)

    blind_key = {}
    for (m, a, p), code in zip(configs, letters):
        blind_key[code] = {
            "model_id": m,
            "adapter_dir": a,
            "precision": p,
        }

    # Save blind key
    blind_key_path.parent.mkdir(parents=True, exist_ok=True)
    with blind_key_path.open("w", encoding="utf-8") as f:
        json.dump(blind_key, f, indent=2)

    # Reverse lookup for blinding
    config_to_code = {
        (v["model_id"], v["adapter_dir"], v["precision"]): k
        for k, v in blind_key.items()
    }

    sheet_rows = []
    for r in scored_records:
        code = config_to_code.get(
            (r.get("model_id"), r.get("adapter_dir"), r["precision"]), "Model_Unknown"
        )
        sheet_rows.append(
            {
                "eval_id": r["id"],
                "blinded_model": code,
                "language": r["lang"],
                "domain": r.get("domain", "disaster"),
                "question": r["instruction"],
                "ground_truth_answer": r["ground_truth"],
                "model_generated_answer": r["prediction"],
                "correctness_score_0_to_2": "",
                "error_category": "",  # FACT_SUB, FAB_ENT, CODE_MIX_TOK, INCOHERENT, UNGROUNDED_REFUSAL, NONE
                "evaluator_name": "",
                "notes": "",
            }
        )

    # Shuffle rows so raters do not evaluate same model consecutively
    rng.shuffle(sheet_rows)

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sheet_rows)

    print(f"✓ Blinded human review sheet generated ({len(sheet_rows)} rows) -> {sheet_path}")
    print(f"✓ Confidential blind key saved -> {blind_key_path}")


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    p = argparse.ArgumentParser(description="Score predictions and create human eval sheet")
    p.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PRED_FILE,
        help=f"Predictions file (default: {DEFAULT_PRED_FILE})",
    )
    p.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help=f"Automated metrics JSON (default: {DEFAULT_METRICS_FILE})",
    )
    p.add_argument(
        "--sheet-output",
        type=Path,
        default=DEFAULT_HUMAN_SHEET,
        help=f"Human review CSV (default: {DEFAULT_HUMAN_SHEET})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for blinding (default: 42)",
    )
    args = p.parse_args(argv)

    if not args.predictions.exists():
        print(f"Error: Predictions file not found: {args.predictions}")
        print("Run scripts/evaluate_models.py first to generate predictions.")
        return 1

    print(f"Scoring predictions from {args.predictions} ...")
    scored_records, summary = score_predictions(args.predictions)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Automated metrics saved -> {args.metrics_output}")

    print("\nSummary by (Language_Precision):")
    for k, v in summary.items():
        print(f"  {k:15s} | F1: {v['avg_f1']:.3f} | Purity: {v['avg_script_purity']:.3f} | Repetition: {v['repetition_rate']:.3f}")

    generate_blinded_human_sheet(
        scored_records=scored_records,
        sheet_path=args.sheet_output,
        blind_key_path=DEFAULT_BLIND_KEY,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
