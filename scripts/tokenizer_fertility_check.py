"""
Phase 1: Tokenizer fertility comparison for Bangla vs English.

Loads candidate ~1B-scale model tokenizers and reports average fertility
(tokens / whitespace-separated words) across matched sentence pairs.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sentence_pairs.jsonl"
RESULTS_DIR = ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "fertility_comparison.csv"

CANDIDATE_MODELS = [
    "google/gemma-3-1b-it",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "meta-llama/Llama-3.2-1B-Instruct",
]


def load_sentence_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Sentence pairs not found: {path}")

    pairs: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc

            for key in ("id", "bn", "en"):
                if key not in row or not str(row[key]).strip():
                    raise ValueError(f"Missing/empty '{key}' on line {line_no} of {path}")
            pairs.append(row)

    if not pairs:
        raise ValueError(f"No sentence pairs found in {path}")
    return pairs


def word_count(text: str) -> int:
    """Whitespace word count (standard for fertility ratios)."""
    words = text.split()
    return max(len(words), 1)


def token_count(tokenizer, text: str) -> int:
    # Prefer encode without special tokens so fertility reflects content only.
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def fertility(tokenizer, text: str) -> float:
    return token_count(tokenizer, text) / word_count(text)


def _classify_load_error(exc: BaseException) -> str:
    """Turn HF load failures into a short, diagnosable message (no stack trace)."""
    msg = str(exc).strip() or exc.__class__.__name__
    lower = msg.lower()
    cls = exc.__class__.__name__

    # Prefer actionable categories over dumping the full exception blob.
    if any(
        s in lower
        for s in (
            "401",
            "403",
            "unauthorized",
            "invalid username or password",
            "invalid token",
            "cannot access gated repo",
            "gated repo",
            "access to model",
            "you are trying to access a gated",
            "must be authenticated",
            "please log in",
            "hf_token",
        )
    ):
        return (
            f"{cls}: AUTH/GATING - cannot download tokenizer for this model. "
            "Accept the model license on huggingface.co, then run "
            "`huggingface-cli login` (Read token) or set HF_TOKEN. "
            f"Detail: {msg[:300]}"
        )
    if "404" in msg or "not found" in lower or "repository not found" in lower:
        return (
            f"{cls}: NOT_FOUND - model id missing or private/renamed on Hugging Face. "
            f"Detail: {msg[:300]}"
        )
    if any(s in lower for s in ("connection", "timed out", "timeout", "network", "dns")):
        return f"{cls}: NETWORK - {msg[:300]}"
    return f"{cls}: {msg[:400]}"


def load_tokenizer(model_id: str):
    """Load tokenizer; return (tokenizer, None) or (None, error_message)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return tokenizer, None
    except Exception as exc:  # noqa: BLE001 — surface any HF/auth/network failure
        return None, _classify_load_error(exc)


def evaluate_model(tokenizer, pairs: list[dict]) -> dict[str, float]:
    bn_fertilities = [fertility(tokenizer, p["bn"]) for p in pairs]
    en_fertilities = [fertility(tokenizer, p["en"]) for p in pairs]

    avg_bn = sum(bn_fertilities) / len(bn_fertilities)
    avg_en = sum(en_fertilities) / len(en_fertilities)
    ratio = avg_bn / avg_en if avg_en > 0 else float("inf")

    return {
        "avg_bangla_fertility": avg_bn,
        "avg_english_fertility": avg_en,
        "bangla_english_ratio": ratio,
    }


def print_table(rows: list[dict]) -> None:
    headers = (
        "model",
        "avg Bangla fertility",
        "avg English fertility",
        "Bangla/English ratio",
        "status",
    )
    display = []
    for r in rows:
        if r["status"] == "ok":
            display.append(
                (
                    r["model"],
                    f"{r['avg_bangla_fertility']:.4f}",
                    f"{r['avg_english_fertility']:.4f}",
                    f"{r['bangla_english_ratio']:.4f}",
                    "ok",
                )
            )
        else:
            display.append((r["model"], "-", "-", "-", "FAILED"))

    widths = [len(h) for h in headers]
    for row in display:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt(headers))
    print(sep)
    for row in display:
        print(fmt(row))


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "avg_bangla_fertility",
        "avg_english_fertility",
        "bangla_english_ratio",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "model": r["model"],
                    "avg_bangla_fertility": (
                        f"{r['avg_bangla_fertility']:.6f}" if r["status"] == "ok" else ""
                    ),
                    "avg_english_fertility": (
                        f"{r['avg_english_fertility']:.6f}" if r["status"] == "ok" else ""
                    ),
                    "bangla_english_ratio": (
                        f"{r['bangla_english_ratio']:.6f}" if r["status"] == "ok" else ""
                    ),
                    "status": r["status"],
                    "error": r.get("error", ""),
                }
            )


def main() -> int:
    print(f"Loading sentence pairs from {DATA_PATH} ...")
    pairs = load_sentence_pairs(DATA_PATH)
    print(f"Loaded {len(pairs)} matched Bangla/English pairs.\n")

    rows: list[dict] = []
    for model_id in CANDIDATE_MODELS:
        print(f"[{model_id}] loading tokenizer ...")
        tokenizer, err = load_tokenizer(model_id)
        if err is not None:
            print(f"  ERROR: {err}")
            print("  Skipping this model; continuing with remaining candidates.\n")
            rows.append(
                {
                    "model": model_id,
                    "avg_bangla_fertility": None,
                    "avg_english_fertility": None,
                    "bangla_english_ratio": None,
                    "status": "error",
                    "error": err,
                }
            )
            continue

        metrics = evaluate_model(tokenizer, pairs)
        print(
            f"  Bangla fertility={metrics['avg_bangla_fertility']:.4f} | "
            f"English fertility={metrics['avg_english_fertility']:.4f} | "
            f"ratio={metrics['bangla_english_ratio']:.4f}\n"
        )
        rows.append(
            {
                "model": model_id,
                **metrics,
                "status": "ok",
                "error": "",
            }
        )

    print("=" * 72)
    print("Tokenizer fertility comparison (tokens / whitespace words)")
    print("=" * 72)
    print_table(rows)
    print()

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = min(ok_rows, key=lambda r: r["bangla_english_ratio"])
        print(
            f"Lowest Bangla/English fertility ratio (closer to parity): "
            f"{best['model']} ({best['bangla_english_ratio']:.4f})"
        )
    else:
        print("No tokenizers loaded successfully. Check HF auth / network.")

    save_csv(rows, OUTPUT_CSV)
    print(f"\nSaved: {OUTPUT_CSV}")
    return 0 if ok_rows else 1


if __name__ == "__main__":
    sys.exit(main())
