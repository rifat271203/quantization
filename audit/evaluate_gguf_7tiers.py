# -*- coding: utf-8 -*-
"""7-Tier Quantization Degradation Evaluator using llama.cpp and Belebele Benchmark.

Evaluates Gemma 3 1B IT across:
- f16
- Q8_0
- Q6_K
- Q5_K_M
- Q4_K_M
- Q3_K_M
- Q2_K

Measures accuracy, ungrounded distractor (hallucination) rate, and cross-lingual degradation gap.
"""

import os
import sys

# Ensure UTF-8 everywhere on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_V2 = ROOT / "data_v2"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
GGUF_DIR = ROOT / "models" / "gguf"
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"

DEFAULT_BN_FILE = DATA_V2 / "belebele_eval_bn_300.jsonl"
DEFAULT_EN_FILE = DATA_V2 / "belebele_eval_en_300.jsonl"
DEFAULT_PRED_FILE = RESULTS_DIR / "gguf_7tier_predictions.jsonl"
DEFAULT_SUMMARY_FILE = RESULTS_DIR / "gguf_7tier_summary.json"

ALL_TIERS = [
    ("f16", "gemma-3-1b-it-f16.gguf", 16.0),
    ("Q8_0", "gemma-3-1b-it-Q8_0.gguf", 8.5),
    ("Q6_K", "gemma-3-1b-it-Q6_K.gguf", 6.5),
    ("Q5_K_M", "gemma-3-1b-it-Q5_K_M.gguf", 5.5),
    ("Q4_K_M", "gemma-3-1b-it-Q4_K_M.gguf", 4.5),
    ("Q3_K_M", "gemma-3-1b-it-Q3_K_M.gguf", 3.5),
    ("Q2_K", "gemma-3-1b-it-Q2_K.gguf", 2.5),
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GGUF 7-tier degradation curve")
    p.add_argument("--tiers", nargs="+", default=None, help="Specific tiers to evaluate (e.g. Q4_K_M Q2_K)")
    p.add_argument("--languages", nargs="+", choices=("bangla", "english"), default=["bangla", "english"])
    p.add_argument("--limit", type=int, default=None, help="Limit items per language (e.g. 5 for quick test)")
    p.add_argument("--port", type=int, default=8089, help="Port for llama-server")
    p.add_argument("--threads", type=int, default=4, help="CPU threads per llama-server instance")
    p.add_argument("--output-file", type=Path, default=DEFAULT_PRED_FILE)
    p.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    return p.parse_args()


def load_dataset(lang: str, limit: int | None = None) -> list[dict]:
    path = DEFAULT_BN_FILE if lang == "bangla" else DEFAULT_EN_FILE
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    items = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_user_prompt(passage: str, instruction: str, lang: str) -> str:
    if lang == "bangla":
        return (
            "অনুগ্রহ করে নিচের অনুচ্ছেদটি মনোযোগ দিয়ে পড়ুন এবং শুধুমাত্র এই অনুচ্ছেদের তথ্যের ভিত্তিতে "
            "প্রশ্নের সঠিক বিকল্পটি বেছে নিন। বিকল্পের নম্বরটি পরিষ্কারভাবে উল্লেখ করুন।\n\n"
            f"অনুচ্ছেদ:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
            f"{instruction.strip()}\n\n"
            "সঠিক উত্তর:"
        )
    else:
        return (
            "Read the following passage carefully and answer the question based ONLY on facts directly stated "
            "in the passage. Clearly indicate the correct option number.\n\n"
            f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
            f"{instruction.strip()}\n\n"
            "Correct Answer:"
        )


def extract_chosen_option(response_text: str) -> str | None:
    # Match "(1)", "1", "Option 1", "বিকল্প ১", "(১)"
    m = re.search(r"\(([1-4১-৪])\)", response_text)
    if m:
        digit = m.group(1)
        bn_to_en = {"১": "1", "২": "2", "৩": "3", "৪": "4"}
        return bn_to_en.get(digit, digit)
    m = re.search(r"\b([1-4])\b", response_text[:30])
    if m:
        return m.group(1)
    m = re.search(r"([১-৪])", response_text[:30])
    if m:
        bn_to_en = {"১": "1", "২": "2", "৩": "3", "৪": "4"}
        return bn_to_en.get(m.group(1))
    return None


def start_llama_server(model_path: Path, port: int, threads: int = 4) -> subprocess.Popen:
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(f"llama-server not found at: {LLAMA_SERVER}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "--port", str(port),
        "-t", str(threads),
        "-c", "2048",
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for_server(port: int, timeout_sec: int = 60) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def query_server(prompt: str, port: int, max_retries: int = 3, timeout_sec: int = 180) -> tuple[str, float]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            lat_ms = (time.perf_counter() - t0) * 1000.0
            text = res["choices"][0]["message"]["content"]
            return text, lat_ms
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
            
    raise TimeoutError(f"Failed to query server on port {port} after {max_retries} attempts: {last_err}")


def load_existing_keys(pred_file: Path) -> set[tuple[str, str, str]]:
    keys = set()
    if pred_file.exists():
        with pred_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        keys.add((obj["tier"], obj["language"], obj["id"]))
                    except Exception:
                        pass
    return keys


def main():
    args = parse_args()
    print("=" * 70)
    print("Gemma 3 1B IT — 7-Tier Quantization Degradation Study")
    print(f"Languages:   {args.languages}")
    print(f"Limit:       {args.limit if args.limit else 'All (300)'}")
    print(f"Output File: {args.output_file}")
    print("=" * 70)

    # Determine which tiers to evaluate
    selected_tiers = []
    for tier_name, filename, bpw in ALL_TIERS:
        if args.tiers is None or tier_name in args.tiers:
            model_file = GGUF_DIR / filename
            selected_tiers.append((tier_name, model_file, bpw))

    existing_keys = load_existing_keys(args.output_file)
    summary_data = []

    for tier_name, model_file, bpw in selected_tiers:
        if not model_file.exists():
            print(f"\n[SKIP] Tier '{tier_name}' not downloaded yet: {model_file.name}")
            continue

        print(f"\n{'='*70}")
        print(f"STARTING TIER: {tier_name} (approx {bpw} bpw)")
        print(f"Model file: {model_file.name} ({model_file.stat().st_size / (1024*1024):.1f} MB)")
        print(f"{'='*70}")

        print(f"Launching llama-server on port {args.port} with {args.threads} threads ...", flush=True)
        proc = start_llama_server(model_file, args.port, threads=args.threads)
        ready = wait_for_server(args.port, timeout_sec=60)
        if not ready:
            print(f"ERROR: Server failed to start for {tier_name} within 60s!")
            proc.kill()
            continue
        print("Server is ready and healthy!")

        try:
            for lang in args.languages:
                items = load_dataset(lang, limit=args.limit)
                print(f"\n[{tier_name} | {lang.upper()}] Evaluating {len(items)} items ...")

                total = 0
                correct = 0
                hallucinated = 0
                unparseable = 0
                total_latency = 0.0

                with args.output_file.open("a", encoding="utf-8") as out_f:
                    for idx, item in enumerate(items, start=1):
                        item_id = item["id"]
                        if (tier_name, lang, item_id) in existing_keys:
                            continue

                        prompt = format_user_prompt(item["passage"], item["instruction"], lang)
                        resp_text, lat_ms = query_server(prompt, args.port)
                        pred_opt = extract_chosen_option(resp_text)

                        gold_opt = str(item["correct_option"]).strip()
                        is_correct = (pred_opt == gold_opt)
                        is_distractor = (pred_opt is not None and not is_correct)
                        is_unparse = (pred_opt is None)

                        total += 1
                        total_latency += lat_ms
                        if is_correct:
                            correct += 1
                        elif is_distractor:
                            hallucinated += 1
                        else:
                            unparseable += 1

                        record = {
                            "tier": tier_name,
                            "bpw": bpw,
                            "language": lang,
                            "id": item_id,
                            "prompt_length": len(prompt),
                            "gold_option": gold_opt,
                            "pred_option": pred_opt,
                            "is_correct": is_correct,
                            "is_distractor_hallucination": is_distractor,
                            "is_unparseable": is_unparse,
                            "latency_ms": round(lat_ms, 2),
                            "raw_response": resp_text.strip(),
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        existing_keys.add((tier_name, lang, item_id))

                        if idx % 50 == 0 or idx == len(items):
                            acc = (correct / total * 100) if total else 0.0
                            hall_pct = (hallucinated / total * 100) if total else 0.0
                            print(f"  Processed {idx}/{len(items)} -> Acc: {acc:.1f}% | Hallucinated: {hall_pct:.1f}%")

                if total > 0:
                    tier_lang_summary = {
                        "tier": tier_name,
                        "bpw": bpw,
                        "language": lang,
                        "evaluated_items": total,
                        "accuracy_pct": round(correct / total * 100, 2),
                        "hallucination_pct": round(hallucinated / total * 100, 2),
                        "unparseable_pct": round(unparseable / total * 100, 2),
                        "avg_latency_ms": round(total_latency / total, 2),
                    }
                    summary_data.append(tier_lang_summary)
                    print(f"[{tier_name} | {lang.upper()} FINAL] Acc: {tier_lang_summary['accuracy_pct']}% | Hallucination: {tier_lang_summary['hallucination_pct']}% | Latency: {tier_lang_summary['avg_latency_ms']} ms")

        finally:
            print(f"Shutting down llama-server for {tier_name} ...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(1.0)

    # Save summary
    if summary_data:
        with args.summary_file.open("w", encoding="utf-8") as sf:
            json.dump(summary_data, sf, indent=2, ensure_ascii=False)
        print(f"\nSummary report saved to: {args.summary_file}")


if __name__ == "__main__":
    main()
