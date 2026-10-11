"""
Compile reviewer-approved QA pairs into fine-tuning and evaluation JSONL files.

Features:
- Enforces document-level stratified splitting across domains so evaluation facts
  never leak from fine-tuning documents into evaluation.
- Preserves full source passage text alongside instructions and responses to enable
  primary context-grounded hallucination evaluation.
- Keeps strictly identical IDs across Bangla and English pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
REVIEW_DIR = ROOT / "review"
DATA_DIR = ROOT / "data"

DOMAINS = ("disaster", "agriculture", "education")
APPROVED_VALUES = {"yes", "y"}

BN_TRAIN_PATH = DATA_DIR / "train_bangla.jsonl"
EN_TRAIN_PATH = DATA_DIR / "train_english.jsonl"
BN_EVAL_PATH = DATA_DIR / "eval_bangla.jsonl"
EN_EVAL_PATH = DATA_DIR / "eval_english.jsonl"

CSV_REQUIRED = (
    "id",
    "domain",
    "instruction_bn",
    "response_bn",
    "instruction_en",
    "response_en",
    "approved",
)


def list_draft_csvs(domain: str | None) -> list[Path]:
    if domain:
        path = REVIEW_DIR / f"draft_pairs_{domain}.csv"
        return [path] if path.exists() else []
    found = sorted(REVIEW_DIR.glob("draft_pairs_*.csv"))
    ordered: list[Path] = []
    seen: set[Path] = set()
    for name in DOMAINS:
        path = REVIEW_DIR / f"draft_pairs_{name}.csv"
        if path.exists():
            ordered.append(path)
            seen.add(path)
    for path in found:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def is_approved(value: str) -> bool:
    return value.strip().lower() in APPROVED_VALUES


def extract_doc_key(row: dict) -> str:
    """Derive parent document identifier from source_file or ID prefix."""
    sf = (row.get("source_file") or "").strip()
    if sf:
        return Path(sf).stem
    pair_id = (row.get("id") or "").strip()
    # e.g., disaster_ddm_act_0001__q1 -> disaster_ddm_act
    stem = re.sub(r"__q\d+$", "", pair_id)
    stem = re.sub(r"_\d{3,4}$", "", stem)
    return stem or "doc_unknown"


def load_chunk_passages() -> dict[str, str]:
    """Load text for chunk_ids across all chunks_*.jsonl files."""
    chunk_map: dict[str, str] = {}
    for path in PIPELINE_DIR.glob("chunks_*.jsonl"):
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    cid = obj.get("chunk_id")
                    txt = obj.get("text")
                    if cid and txt:
                        chunk_map[cid] = txt
        except Exception:
            pass
    return chunk_map


def read_approved(path: Path, chunk_map: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CSV_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            print(f"  skip {path.name}: missing columns {missing}")
            return rows
        for line_no, row in enumerate(reader, start=2):
            if not is_approved(row.get("approved") or ""):
                continue
            pair_id = (row.get("id") or "").strip()
            if not pair_id:
                print(f"  {path.name}:{line_no} approved but empty id; skipped")
                continue
            bn_i = (row.get("instruction_bn") or "").strip()
            bn_r = (row.get("response_bn") or "").strip()
            en_i = (row.get("instruction_en") or "").strip()
            en_r = (row.get("response_en") or "").strip()
            if not all((bn_i, bn_r, en_i, en_r)):
                print(f"  {path.name}:{line_no} id={pair_id} incomplete text; skipped")
                continue

            doc_key = extract_doc_key(row)
            # Find passage: from row if available, else from chunk map
            passage = (row.get("passage") or row.get("source_passage") or "").strip()
            if not passage:
                chunk_id = pair_id.rsplit("__q", 1)[0] if "__q" in pair_id else pair_id
                passage = chunk_map.get(chunk_id, (row.get("source_excerpt") or "").strip())

            rows.append(
                {
                    "id": pair_id,
                    "domain": (row.get("domain") or "").strip(),
                    "doc_id": doc_key,
                    "source_file": (row.get("source_file") or "").strip(),
                    "source_page": row.get("source_page", ""),
                    "passage": passage,
                    "instruction_bn": bn_i,
                    "response_bn": bn_r,
                    "instruction_en": en_i,
                    "response_en": en_r,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def document_level_split(
    items: list[dict], eval_ratio: float, seed: int
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """
    Split QA items by source document stratified by domain.
    Returns (train_items, eval_items, doc_assignment_map).
    """
    if eval_ratio <= 0:
        return items, [], {}

    rng = random.Random(seed)
    # Group items by domain -> doc_id -> items
    domain_docs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        domain_docs[item["domain"]][item["doc_id"]].append(item)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    doc_assignments: dict[str, str] = {}

    for domain, docs in domain_docs.items():
        doc_list = list(docs.keys())
        rng.shuffle(doc_list)

        total_domain_items = sum(len(items_in_doc) for items_in_doc in docs.values())
        target_eval_items = int(round(total_domain_items * eval_ratio))

        if len(doc_list) >= 2:
            # Multi-document: partition whole documents
            curr_eval_items = 0
            domain_eval_docs: set[str] = set()
            for doc in doc_list:
                doc_len = len(docs[doc])
                # Ensure at least one doc remains in train
                if len(domain_eval_docs) < len(doc_list) - 1 and (
                    curr_eval_items + doc_len <= target_eval_items
                    or len(domain_eval_docs) == 0
                ):
                    domain_eval_docs.add(doc)
                    curr_eval_items += doc_len

            for doc in doc_list:
                split = "eval" if doc in domain_eval_docs else "train"
                doc_assignments[doc] = split
                if split == "eval":
                    eval_rows.extend(docs[doc])
                else:
                    train_rows.extend(docs[doc])
        else:
            # Single document: partition non-overlapping chunk sections
            only_doc = doc_list[0]
            doc_items = sorted(docs[only_doc], key=lambda x: x["id"])
            cutoff = int(len(doc_items) * (1.0 - eval_ratio))
            train_rows.extend(doc_items[:cutoff])
            eval_rows.extend(doc_items[cutoff:])
            doc_assignments[only_doc] = f"chunk_split({cutoff}/{len(doc_items) - cutoff})"

    return train_rows, eval_rows, doc_assignments


def compile_all(
    domain: str | None,
    eval_ratio: float = 0.25,
    seed: int = 42,
    dry_run: bool = False,
) -> int:
    csvs = list_draft_csvs(domain)
    if not csvs:
        target = f"draft_pairs_{domain}.csv" if domain else "draft_pairs_<domain>.csv"
        print(f"No review files matching {REVIEW_DIR / target}")
        return 1

    chunk_map = load_chunk_passages()
    print(f"Loaded passage text index: {len(chunk_map)} chunks.")

    approved: list[dict] = []
    seen_ids: set[str] = set()
    for path in csvs:
        print(f"Reading {path.name}")
        for row in read_approved(path, chunk_map):
            if row["id"] in seen_ids:
                print(f"  duplicate id {row['id']}; keeping first")
                continue
            seen_ids.add(row["id"])
            approved.append(row)

    if not approved:
        print("No approved rows found.")
        return 1

    train_rows, eval_rows, doc_assignments = document_level_split(
        approved, eval_ratio=eval_ratio, seed=seed
    )

    # Sort each set by id for deterministic reproducibility
    eval_rows.sort(key=lambda x: x["id"])
    train_rows.sort(key=lambda x: x["id"])

    bn_train = [
        {
            "id": r["id"],
            "domain": r["domain"],
            "doc_id": r["doc_id"],
            "source_file": r["source_file"],
            "source_page": r["source_page"],
            "passage": r["passage"],
            "instruction": r["instruction_bn"],
            "response": r["response_bn"],
        }
        for r in train_rows
    ]
    en_train = [
        {
            "id": r["id"],
            "domain": r["domain"],
            "doc_id": r["doc_id"],
            "source_file": r["source_file"],
            "source_page": r["source_page"],
            "passage": r["passage"],
            "instruction": r["instruction_en"],
            "response": r["response_en"],
        }
        for r in train_rows
    ]
    bn_eval = [
        {
            "id": r["id"],
            "domain": r["domain"],
            "doc_id": r["doc_id"],
            "source_file": r["source_file"],
            "source_page": r["source_page"],
            "passage": r["passage"],
            "instruction": r["instruction_bn"],
            "response": r["response_bn"],
        }
        for r in eval_rows
    ]
    en_eval = [
        {
            "id": r["id"],
            "domain": r["domain"],
            "doc_id": r["doc_id"],
            "source_file": r["source_file"],
            "source_page": r["source_page"],
            "passage": r["passage"],
            "instruction": r["instruction_en"],
            "response": r["response_en"],
        }
        for r in eval_rows
    ]

    print("\n" + "=" * 64)
    print(f"Document Assignments (Document-Level Stratified Split, seed={seed}):")
    print("=" * 64)
    for doc, split in sorted(doc_assignments.items()):
        print(f"  {doc:35s} -> {split}")

    if not dry_run:
        write_jsonl(BN_TRAIN_PATH, bn_train)
        write_jsonl(EN_TRAIN_PATH, en_train)
        if eval_rows:
            write_jsonl(BN_EVAL_PATH, bn_eval)
            write_jsonl(EN_EVAL_PATH, en_eval)

    by_domain = Counter(row["domain"] or "(blank)" for row in approved)
    print()
    print("=" * 64)
    print(f"Approved QA pairs: Total = {len(approved)} (Train = {len(train_rows)}, Eval = {len(eval_rows)})")
    print("=" * 64)
    for name, count in sorted(by_domain.items()):
        print(f"  {name:16s} {count:5d}")
    print(f"  {'Bangla Train':16s} {len(bn_train):5d}  -> {BN_TRAIN_PATH}")
    print(f"  {'English Train':16s} {len(en_train):5d}  -> {EN_TRAIN_PATH}")
    if eval_rows:
        print(f"  {'Bangla Eval':16s} {len(bn_eval):5d}  -> {BN_EVAL_PATH}")
        print(f"  {'English Eval':16s} {len(en_eval):5d}  -> {EN_EVAL_PATH}")
    if dry_run:
        print("\n[DRY RUN] No files written.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write train and eval jsonl files with document-level stratified splitting."
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        default=None,
        help="Compile one domain only (default: all draft_pairs_*.csv).",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.25,
        help="Fraction of approved pairs to hold out for evaluation benchmark (default: 0.25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible document-level train/eval split (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display split and counts without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    return compile_all(
        args.domain,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
