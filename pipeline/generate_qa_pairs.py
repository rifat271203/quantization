"""
Generate bilingual factual QA pairs from chunked source passages.

Uses an OpenAI-compatible chat API once per eligible chunk. Defaults to
OpenRouter (free models). Also supports xAI Grok and Anthropic. Failures are
logged and skipped so a single bad page does not abort the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
REVIEW_DIR = ROOT / "review"
# #region agent log
_DEBUG_LOG = ROOT / "debug-24bffa.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "post-fix") -> None:
    try:
        payload = {
            "sessionId": "24bffa",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# #endregion

DOMAINS = ("disaster", "agriculture", "education")
MIN_WORDS = 50
DEFAULT_SLEEP_S = 1.0
MAX_RETRIES = 5

# provider -> (env key names, base_url, default model)
PROVIDERS = {
    "gemini": (
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash-lite",
    ),
    "openrouter": (
        ("OPENROUTER_API_KEY",),
        "https://openrouter.ai/api/v1",
        # Free Models Router — auto-picks a currently available free model.
        # Override with --model e.g. google/gemma-4-31b-it:free
        "openrouter/free",
    ),
    "xai": (
        ("XAI_API_KEY", "GROK_API_KEY"),
        "https://api.x.ai/v1",
        "grok-2-latest",
    ),
    "anthropic": (
        ("ANTHROPIC_API_KEY",),
        "https://api.anthropic.com/v1",
        "claude-sonnet-4-6",
    ),
}

CSV_FIELDS = [
    "id",
    "domain",
    "source_file",
    "source_page",
    "source_excerpt",
    "passage",
    "instruction_bn",
    "response_bn",
    "instruction_en",
    "response_en",
    "approved",
    "reviewer_notes",
]

PAIR_KEYS = ("instruction_bn", "response_bn", "instruction_en", "response_en")

SYSTEM_PROMPT = (
    "You extract bilingual factual QA pairs from a single source passage. "
    "You never add facts that are not in the passage. "
    "You return only valid JSON — no markdown, no commentary."
)

USER_PROMPT_TEMPLATE = """Read the passage below and generate 1-3 factual question-answer pairs.

PASSAGE:
\"\"\"
{passage}
\"\"\"

Rules:
1. Use ONLY facts stated in the passage. No outside knowledge. No invented names, numbers, dates, or procedures.
2. If the passage has no extractable facts, return an empty JSON array: []
3. Each pair must exist in BOTH Bangla and English as a translation-matched pair: the same fact, the same specificity. Phrasing should be natural in each language, not a word-for-word calque.
4. Prefer concrete, checkable facts over vague summaries.
5. Return ONLY a JSON array of objects with exactly these keys:
[{{"instruction_bn": "...", "response_bn": "...", "instruction_en": "...", "response_en": "..."}}]
"""

TOC_RE = re.compile(
    r"(table of contents|সূচিপত্র|content\s+page|list of (tables|figures))",
    re.IGNORECASE,
)
DOT_LEADERS_RE = re.compile(r"[.\u2026]{4,}")
PAGE_ONLY_RE = re.compile(r"^(page\s*)?\d+$", re.IGNORECASE)
LETTER_RE = re.compile(r"[A-Za-z\u0980-\u09FF]")


def word_count(text: str) -> int:
    return len(text.split())


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks not found: {path}. Run extract_text.py first.")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
            rows.append(row)
    return rows


def is_boilerplate(text: str) -> bool:
    """Heuristic: headers, TOC, page numbers, punctuation-heavy junk."""
    stripped = text.strip()
    if word_count(stripped) < MIN_WORDS:
        return True
    letters = len(LETTER_RE.findall(stripped))
    if letters / max(len(stripped), 1) < 0.30:
        return True
    if TOC_RE.search(stripped) and (
        len(DOT_LEADERS_RE.findall(stripped)) >= 3 or stripped.count("..") >= 8
    ):
        return True
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines:
        short = sum(1 for ln in lines if word_count(ln) <= 3 or PAGE_ONLY_RE.match(ln))
        if short / len(lines) >= 0.70:
            return True
    return False


def excerpt(text: str, n: int = 100) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:n]


def existing_chunk_ids(csv_path: Path, domain: str = "") -> set[str]:
    """Chunk ids already present in the draft CSV or empty chunks log (resume support)."""
    seen: set[str] = set()
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair_id = (row.get("id") or "").strip()
                if "__q" in pair_id:
                    seen.add(pair_id.rsplit("__q", 1)[0])
                elif pair_id:
                    seen.add(pair_id)
    if domain:
        empty_log = PIPELINE_DIR / f".empty_chunks_{domain}.txt"
        if empty_log.exists():
            with empty_log.open(encoding="utf-8") as f:
                for line in f:
                    cid = line.strip()
                    if cid:
                        seen.add(cid)
    return seen


def log_empty_chunk(domain: str, chunk_id: str) -> None:
    """Record chunks that yielded no factual pairs so they are skipped on resume."""
    empty_log = PIPELINE_DIR / f".empty_chunks_{domain}.txt"
    empty_log.parent.mkdir(parents=True, exist_ok=True)
    with empty_log.open("a", encoding="utf-8") as f:
        f.write(f"{chunk_id}\n")


def parse_pairs_json(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text, strict=False)
    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array")
    pairs: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Array element is not an object")
        pair = {}
        for key in PAIR_KEYS:
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing/empty '{key}'")
            pair[key] = value.strip()
        pairs.append(pair)
    return pairs


def append_failures(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    fieldnames = CSV_FIELDS
    if not new_file:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                existing = next(reader, [])
                if existing:
                    fieldnames = existing
        except Exception:
            pass
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_api_key(provider: str) -> str:
    env_names, _, _ = PROVIDERS[provider]
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    # Check ROOT / .env file if available
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            with env_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k in env_names and v:
                        os.environ[k] = v
                        return v
        except Exception:
            pass

    joined = " or ".join(env_names)
    print(f"No API key found. Set {joined} for provider={provider} or put it in .env.")
    print("Examples:")
    print('  $env:OPENROUTER_API_KEY = "sk-or-..."   # PowerShell')
    print('  $env:XAI_API_KEY = "xai-..."')
    sys.exit(1)


def get_client(provider: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("Package 'openai' is missing. Run: pip install -r requirements.txt")
        sys.exit(1)

    if provider == "anthropic":
        # Anthropic Messages API is not OpenAI-compatible; use their SDK.
        try:
            from anthropic import Anthropic
        except ImportError:
            print("Package 'anthropic' is missing. Run: pip install -r requirements.txt")
            sys.exit(1)
        return ("anthropic", Anthropic(api_key=resolve_api_key(provider)))

    _, base_url, _ = PROVIDERS[provider]
    kwargs: dict = {}
    if provider == "openrouter":
        # Optional but recommended by OpenRouter for rankings / rate limits.
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/quant-hallucination-study",
            "X-Title": "quant-hallucination-study",
        }
    client = OpenAI(api_key=resolve_api_key(provider), base_url=base_url, **kwargs)
    return ("openai", client)


def call_llm(backend: str, client, model: str, passage: str) -> str:
    user_content = USER_PROMPT_TEMPLATE.format(passage=passage)
    # #region agent log
    _agent_log(
        "A,D",
        "generate_qa_pairs.py:call_llm",
        "about to call LLM",
        {
            "backend": backend,
            "model": model,
            "passage_chars": len(passage),
            "base_url": getattr(getattr(client, "base_url", None), "host", str(getattr(client, "base_url", ""))),
        },
    )
    # #endregion
    if backend == "anthropic":
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        parts = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts).strip()

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as exc:
        # #region agent log
        err_body = getattr(exc, "body", None)
        _agent_log(
            "A,B,C",
            "generate_qa_pairs.py:call_llm",
            "LLM call failed",
            {
                "model": model,
                "exc_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "message": str(exc)[:500],
                "body": str(err_body)[:500] if err_body is not None else None,
            },
        )
        # #endregion
        raise
    content = response.choices[0].message.content
    # #region agent log
    _agent_log(
        "A",
        "generate_qa_pairs.py:call_llm",
        "LLM call succeeded",
        {"model": model, "content_chars": len(content or "")},
    )
    # #endregion
    if not content:
        raise ValueError("Empty model response")
    return content.strip()


def extract_retry_delay(err: BaseException) -> float | None:
    msg = str(err)
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m2 = re.search(r"retryDelay['\"]?\s*:\s*['\"]?([0-9]+(?:\.[0-9]+)?)s?", msg, re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass
    return None


def generate_with_retries(
    backend: str, client, model: str, passage: str, sleep_s: float
) -> tuple[list[dict], str]:
    try:
        from openai import APIError, NotFoundError, RateLimitError
    except ImportError:
        APIError = Exception  # type: ignore[misc, assignment]
        NotFoundError = Exception  # type: ignore[misc, assignment]
        RateLimitError = Exception  # type: ignore[misc, assignment]

    anthropic_rate = ()
    anthropic_api = ()
    if backend == "anthropic":
        import anthropic

        anthropic_rate = (anthropic.RateLimitError,)
        anthropic_api = (anthropic.APIError,)

    current_model = model
    last_err: BaseException | None = None
    rate_limit_attempts = 0
    max_rate_retries = 10

    for attempt in range(1, MAX_RETRIES + max_rate_retries + 1):
        try:
            raw = call_llm(backend, client, current_model, passage)
            pairs = parse_pairs_json(raw)
            time.sleep(sleep_s)
            return pairs, current_model
        except Exception as exc:
            last_err = exc
            status = getattr(exc, "status_code", None)
            is_rate = (
                isinstance(exc, (*anthropic_rate, RateLimitError))
                or status == 429
                or "429" in str(exc)
                or "RESOURCE_EXHAUSTED" in str(exc)
                or "RateLimit" in exc.__class__.__name__
            )
            if is_rate:
                rate_limit_attempts += 1
                # If hitting daily limit on gemini-2.5-flash, fallback to flash-lite
                if "GenerateRequestsPerDay" in str(exc) and current_model != "gemini-2.5-flash-lite":
                    print(f"    daily quota exhausted on {current_model}; auto-switching to gemini-2.5-flash-lite...")
                    current_model = "gemini-2.5-flash-lite"
                    time.sleep(1.0)
                    continue

                if rate_limit_attempts > max_rate_retries:
                    break

                delay = extract_retry_delay(exc)
                if delay is not None:
                    wait = delay + 2.0
                else:
                    wait = max(sleep_s, 15.0 * min(rate_limit_attempts, 4))
                print(f"    rate-limited, retry {rate_limit_attempts}/{max_rate_retries} in {wait:.1f}s")
                time.sleep(wait)
                continue

            if isinstance(exc, NotFoundError) or status == 404:
                # #region agent log
                _agent_log(
                    "A,E",
                    "generate_qa_pairs.py:generate_with_retries",
                    "404 NotFound — fail fast, no retry",
                    {
                        "model": current_model,
                        "status_code": status,
                        "message": str(exc)[:400],
                    },
                    run_id="post-fix",
                )
                # #endregion
                raise RuntimeError(str(exc)) from exc

            if attempt >= MAX_RETRIES:
                break
            wait = max(sleep_s, 1.5 * attempt)
            print(f"    {exc.__class__.__name__}: {exc}; retry {attempt}/{MAX_RETRIES}")
            time.sleep(wait)

    raise RuntimeError(str(last_err) if last_err else "unknown LLM failure")


def generate_domain(
    domain: str,
    limit: int | None,
    sleep_s: float,
    dry_run: bool,
    provider: str,
    model: str | None,
) -> int:
    chunks_path = PIPELINE_DIR / f"chunks_{domain}.jsonl"
    out_csv = REVIEW_DIR / f"draft_pairs_{domain}.csv"
    fail_path = PIPELINE_DIR / f"failed_chunks_{domain}.jsonl"

    chunks = load_chunks(chunks_path)
    already = existing_chunk_ids(out_csv, domain)

    # Free tier Gemini has 10 RPM quota; pacing at >=6.5s prevents hitting quota ceilings
    if provider == "gemini" and sleep_s < 6.5:
        sleep_s = 6.5

    eligible: list[dict] = []
    skipped_short = 0
    skipped_boiler = 0
    skipped_done = 0
    for chunk in chunks:
        cid = str(chunk.get("chunk_id") or "")
        text = str(chunk.get("text") or "")
        if cid in already:
            skipped_done += 1
            continue
        if word_count(text) < MIN_WORDS:
            skipped_short += 1
            continue
        if is_boilerplate(text):
            skipped_boiler += 1
            continue
        eligible.append(chunk)

    if limit is not None:
        eligible = eligible[: max(limit, 0)]

    print(f"[{domain}] chunks={len(chunks)} eligible={len(eligible)}")
    print(
        f"  skipped: already={skipped_done} short={skipped_short} "
        f"boilerplate={skipped_boiler}"
    )
    default_model = PROVIDERS[provider][2]
    model_id = model or default_model
    # #region agent log
    _agent_log(
        "D,E",
        "generate_qa_pairs.py:generate_domain",
        "resolved model for run",
        {
            "provider": provider,
            "cli_model": model,
            "default_model": default_model,
            "resolved_model": model_id,
            "eligible": len(eligible),
        },
    )
    # #endregion
    print(f"  provider={provider} model={model_id}")

    if dry_run:
        for chunk in eligible:
            print(f"  would send {chunk.get('chunk_id')} ({word_count(chunk.get('text') or '')} words)")
        print("Dry run: no API calls.")
        return 0

    if not eligible:
        print("Nothing to send.")
        return 0

    backend, client = get_client(provider)
    n_pairs = 0
    n_fail = 0

    for i, chunk in enumerate(eligible, start=1):
        cid = str(chunk["chunk_id"])
        print(f"  [{i}/{len(eligible)}] {cid}")
        try:
            pairs, model_id = generate_with_retries(
                backend, client, model_id, str(chunk["text"]), sleep_s
            )
        except Exception as exc:  # noqa: BLE001 — keep the run alive
            n_fail += 1
            print(f"    FAIL: {exc}")
            append_failures(
                fail_path,
                [
                    {
                        "chunk_id": cid,
                        "domain": domain,
                        "source_file": chunk.get("source_file", ""),
                        "source_page": chunk.get("source_page", ""),
                        "error": str(exc),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            )
            continue

        if not pairs:
            print("    no pairs (empty JSON array)")
            log_empty_chunk(domain, cid)
            continue

        rows = []
        for qi, pair in enumerate(pairs, start=1):
            full_txt = str(chunk.get("text") or "")
            rows.append(
                {
                    "id": f"{cid}__q{qi}",
                    "domain": domain,
                    "source_file": chunk.get("source_file", ""),
                    "source_page": chunk.get("source_page", ""),
                    "source_excerpt": excerpt(full_txt),
                    "passage": full_txt,
                    "instruction_bn": pair["instruction_bn"],
                    "response_bn": pair["response_bn"],
                    "instruction_en": pair["instruction_en"],
                    "response_en": pair["response_en"],
                    "approved": "",
                    "reviewer_notes": "",
                }
            )
        append_csv_rows(out_csv, rows)
        n_pairs += len(rows)
        print(f"    wrote {len(rows)} pair(s)")

    print()
    print(f"Wrote {n_pairs} pairs -> {out_csv}")
    if n_fail:
        print(f"Failed chunks: {n_fail} (logged {fail_path})")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bilingual QA pairs from chunks_<domain>.jsonl"
    )
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS.keys()),
        default="gemini",
        help="LLM backend (default: gemini). Also supports openrouter, xai, anthropic.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id for the chosen provider (default depends on --provider).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max eligible chunks to send (for a small test batch).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_S,
        help="Seconds to wait after each API call (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List eligible chunks without calling the API.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    try:
        return generate_domain(
            args.domain,
            args.limit,
            args.sleep,
            args.dry_run,
            args.provider,
            args.model,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
