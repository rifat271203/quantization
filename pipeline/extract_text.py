"""
Extract text from domain PDFs and converted text files, writing ~150-300 word passages.

Supports:
- Direct manifest-driven extraction from data/raw_sources (handling native Unicode PDFs
  and pre-converted legacy Bijoy text files).
- Fallback extraction from sources/<domain>/*.pdf.
- Per page extraction scoring (pdfplumber vs pypdf).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = ROOT / "sources"
RAW_SOURCES_DIR = ROOT / "data" / "raw_sources"
PIPELINE_DIR = ROOT / "pipeline"

DOMAINS = ("disaster", "agriculture", "education")
MIN_CHUNK_WORDS = 150
MAX_CHUNK_WORDS = 300

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
LETTER_RE = re.compile(r"[A-Za-z\u0980-\u09FF]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")


def word_count(text: str) -> int:
    return len(text.split())


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return (cleaned[:48] or "doc")


def score_text(text: str) -> float:
    """Higher is better: letters + Bangla bonus, junk penalty."""
    if not text or not text.strip():
        return -1.0
    letters = len(LETTER_RE.findall(text))
    bangla = len(BANGLA_RE.findall(text))
    junk = text.count("\ufffd") + text.count("")
    return float(letters + 2 * bangla - 25 * junk)


def list_pdfs(folder: Path) -> list[Path]:
    pdfs = sorted({*folder.glob("*.pdf"), *folder.glob("*.PDF")})
    return [p for p in pdfs if p.is_file()]


def extract_with_pdfplumber(path: Path) -> list[str]:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is not installed. Run: pip install -r requirements.txt"
        ) from None
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append((page.extract_text() or "").strip())
    return pages


def extract_with_pypdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is not installed. Run: pip install -r requirements.txt"
        ) from None
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append((page.extract_text() or "").strip())
        except Exception:  # noqa: BLE001 — one bad page should not kill the file
            pages.append("")
    return pages


def choose_page_text(
    plumber_text: str, pypdf_text: str, engine: str
) -> tuple[str, str]:
    if engine == "pdfplumber":
        return plumber_text, "pdfplumber"
    if engine == "pypdf":
        return pypdf_text, "pypdf"
    s_pl = score_text(plumber_text)
    s_py = score_text(pypdf_text)
    if s_py > s_pl:
        return pypdf_text, "pypdf"
    return plumber_text, "pdfplumber"


def extract_pdf_pages(path: Path, engine: str) -> list[tuple[int, str, str]]:
    """Return (1-based page number, text, engine used)."""
    plumber_pages: list[str] = []
    pypdf_pages: list[str] = []
    if engine in ("auto", "pdfplumber"):
        plumber_pages = extract_with_pdfplumber(path)
    if engine in ("auto", "pypdf"):
        pypdf_pages = extract_with_pypdf(path)

    n = max(len(plumber_pages), len(pypdf_pages))
    out: list[tuple[int, str, str]] = []
    for i in range(n):
        pl = plumber_pages[i] if i < len(plumber_pages) else ""
        py = pypdf_pages[i] if i < len(pypdf_pages) else ""
        text, used = choose_page_text(pl, py, engine)
        out.append((i + 1, text, used))
    return out


def extract_from_txt_file(path: Path) -> list[tuple[int, str, str]]:
    """Extract pages from converted text file (supporting '--- PAGE X ---')."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    pages: list[tuple[int, str, str]] = []
    parts = re.split(r"--- PAGE (\d+) ---", text)
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            page_no = int(parts[i])
            content = parts[i + 1]
            cleaned_lines = []
            for line in content.splitlines():
                l_s = line.strip()
                if not l_s or l_s.startswith("[EMPTY") or l_s.startswith("#"):
                    continue
                cleaned_lines.append(l_s)
            cleaned_text = "\n".join(cleaned_lines).strip()
            if cleaned_text:
                pages.append((page_no, cleaned_text, "converted_txt"))
    else:
        cleaned_lines = [
            ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        pages.append((1, "\n".join(cleaned_lines).strip(), "converted_txt"))
    return pages


def page_to_paragraphs(text: str) -> list[str]:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\S)-\n(\S)", r"\1\2", text)
    blocks = re.split(r"\n\s*\n+", text)
    paras: list[str] = []
    for block in blocks:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in block.split("\n")]
        lines = [ln for ln in lines if ln]
        if lines:
            paras.append(" ".join(lines))
    return paras


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_oversize(text: str) -> list[str]:
    """Split a too-long unit on sentences, then words if needed."""
    sentences = split_sentences(text)
    if len(sentences) <= 1 and word_count(text) > MAX_CHUNK_WORDS:
        words = text.split()
        return [
            " ".join(words[i : i + MAX_CHUNK_WORDS])
            for i in range(0, len(words), MAX_CHUNK_WORDS)
        ]
    packed: list[str] = []
    buf: list[str] = []
    n = 0
    for sent in sentences:
        w = word_count(sent)
        if w > MAX_CHUNK_WORDS:
            if buf:
                packed.append(" ".join(buf))
                buf, n = [], 0
            packed.extend(split_oversize(sent))
            continue
        if n + w > MAX_CHUNK_WORDS and n >= MIN_CHUNK_WORDS:
            packed.append(" ".join(buf))
            buf, n = [sent], w
        else:
            buf.append(sent)
            n += w
    if buf:
        packed.append(" ".join(buf))
    return packed


def chunk_units(units: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Pack (text, page) units into 150-300 word passages. Breaks on paragraphs."""
    chunks: list[tuple[list[str], int]] = []
    buf: list[str] = []
    buf_page = 1
    n = 0

    def flush() -> None:
        nonlocal buf, n
        if buf:
            chunks.append((buf, buf_page))
            buf = []
            n = 0

    for text, page in units:
        w = word_count(text)
        if w == 0:
            continue
        if w > MAX_CHUNK_WORDS:
            flush()
            for piece in split_oversize(text):
                chunks.append(([piece], page))
            continue
        if n + w > MAX_CHUNK_WORDS and n >= MIN_CHUNK_WORDS:
            flush()
        if not buf:
            buf_page = page
        buf.append(text)
        n += w

    if buf:
        leftover_w = n
        leftover = buf
        leftover_page = buf_page
        if chunks and leftover_w < MIN_CHUNK_WORDS:
            prev_texts, prev_page = chunks[-1]
            merged_w = word_count(" ".join(prev_texts)) + leftover_w
            if merged_w <= MAX_CHUNK_WORDS + 50:
                chunks[-1] = (prev_texts + leftover, prev_page)
            else:
                chunks.append((leftover, leftover_page))
        else:
            chunks.append((leftover, leftover_page))

    return [(" ".join(texts).strip(), page) for texts, page in chunks if " ".join(texts).strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest_documents(
    domain: str, raw_sources_dir: Path, pairs_only: bool = True
) -> list[dict]:
    manifest_path = raw_sources_dir / "manifest.csv"
    if not manifest_path.exists():
        return []
    docs: list[dict] = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] != domain:
                continue
            pair_id = row.get("pair_id") or "none"
            if pairs_only and pair_id == "none":
                continue

            text_status = row.get("text_status") or ""
            conv_status = row.get("conversion_status") or ""
            conv_path = row.get("converted_text_path") or ""

            # Check if converted text file is available
            if conv_status == "converted_success" and conv_path and conv_path != "none":
                full_conv = ROOT / conv_path if not Path(conv_path).is_absolute() else Path(conv_path)
                if full_conv.exists():
                    docs.append({
                        "doc_id": row.get("doc_id") or Path(row["filename"]).stem,
                        "pair_id": pair_id,
                        "domain": domain,
                        "language": row.get("language") or "bn",
                        "file_type": "txt",
                        "path": full_conv,
                        "filename": Path(row["filename"]).name,
                    })
                    continue

            # Otherwise, check if PDF text status is ok
            if text_status == "ok":
                rel_fn = row.get("filename") or ""
                full_pdf = raw_sources_dir / rel_fn
                if full_pdf.exists():
                    docs.append({
                        "doc_id": row.get("doc_id") or Path(row["filename"]).stem,
                        "pair_id": pair_id,
                        "domain": domain,
                        "language": row.get("language") or "en",
                        "file_type": "pdf",
                        "path": full_pdf,
                        "filename": Path(row["filename"]).name,
                    })

    return docs


def extract_domain(
    domain: str,
    engine: str,
    source_dir: Path,
    pairs_only: bool = True,
    max_docs: int | None = None,
) -> int:
    out_path = PIPELINE_DIR / f"chunks_{domain}.jsonl"
    rows: list[dict] = []
    engine_wins: dict[str, int] = {}
    empty_pages = 0
    seq = 0

    # 1. Try manifest-driven extraction
    docs_to_process = load_manifest_documents(domain, source_dir, pairs_only=pairs_only)
    if docs_to_process:
        print(f"[{domain.upper()}] Discovered {len(docs_to_process)} usable documents via manifest.")
    else:
        # 2. Fallback to scanning source_dir / domain for PDFs
        src_dir = source_dir / domain if (source_dir / domain).is_dir() else source_dir
        pdfs = list_pdfs(src_dir)
        if not pdfs:
            print(f"No usable documents or PDFs found for domain '{domain}' in {source_dir}.")
            return 1
        print(f"[{domain.upper()}] Discovered {len(pdfs)} PDFs in {src_dir}.")
        docs_to_process = [
            {
                "doc_id": safe_stem(p.name),
                "pair_id": "none",
                "domain": domain,
                "language": "bn" if "bn" in p.name.lower() else "en",
                "file_type": "pdf",
                "path": p,
                "filename": p.name,
            }
            for p in pdfs
        ]

    if max_docs:
        docs_to_process = docs_to_process[:max_docs]
        print(f"Limiting to first {max_docs} document(s).")

    for doc in docs_to_process:
        doc_path: Path = doc["path"]
        file_type: str = doc["file_type"]
        lang: str = doc["language"]
        pair_id: str = doc["pair_id"]
        filename: str = doc["filename"]

        print(f"[{domain}] [{lang}] {filename} ({file_type})")

        if file_type == "txt":
            pages = extract_from_txt_file(doc_path)
        else:
            try:
                pages = extract_pdf_pages(doc_path, engine)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR reading PDF: {exc}")
                continue

        units: list[tuple[str, int]] = []
        for page_no, text, used in pages:
            if not text.strip():
                empty_pages += 1
                continue
            engine_wins[used] = engine_wins.get(used, 0) + 1
            for para in page_to_paragraphs(text):
                units.append((para, page_no))

        if not units:
            print("  No extractable text. Skipping.")
            continue

        file_chunks = chunk_units(units)
        stem = safe_stem(filename)
        for text, page_no in file_chunks:
            seq += 1
            rows.append(
                {
                    "chunk_id": f"{domain}_{stem}_{seq:04d}",
                    "domain": domain,
                    "pair_id": pair_id,
                    "language": lang,
                    "source_file": filename,
                    "source_page": page_no,
                    "text": text,
                }
            )
        print(f"  -> {len(file_chunks)} chunk(s)")

    write_jsonl(out_path, rows)
    print()
    print(f"Wrote {len(rows)} chunks -> {out_path}")
    if engine_wins:
        summary = " ".join(f"{k}={v}" for k, v in sorted(engine_wins.items()))
        print(f"Page extractors used: {summary}")
    if empty_pages:
        print(f"Empty pages skipped: {empty_pages}")
    return 0 if rows else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and chunk text from domain documents for the study."
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=DOMAINS,
        help="Target domain (disaster, agriculture, education)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=RAW_SOURCES_DIR,
        help=f"Directory containing documents and manifest (default: {RAW_SOURCES_DIR})",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "pdfplumber", "pypdf"),
        default="auto",
        help="auto (default) scores both extractors per page.",
    )
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help="Extract all usable documents, not just confirmed bilingual pairs.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional maximum number of documents to extract.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    pairs_only = not args.all_docs
    return extract_domain(
        args.domain,
        args.engine,
        args.source_dir,
        pairs_only=pairs_only,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    sys.exit(main())
