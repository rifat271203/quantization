# -*- coding: utf-8 -*-
"""7-Tier Quantization Degradation Evaluator using Log-Likelihood / Logprob Scoring.

Evaluates Gemma 3 1B IT across:
- f16
- Q8_0
- Q6_K
- Q5_K_M
- Q4_K_M
- Q3_K_M
- Q2_K

Uses native llama-server logprob extraction (max_tokens=1, top_logprobs=20)
with prefilled assistant '(' and symmetric cross-lingual English directives
to eliminate generative parsing artifacts completely.
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
import math
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_V2 = ROOT / "data_v2"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
AUDIT_DIR = ROOT / "audit"
GGUF_DIR = ROOT / "models" / "gguf"
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"

DEFAULT_BN_FILE = DATA_V2 / "belebele_eval_bn_300.jsonl"
DEFAULT_EN_FILE = DATA_V2 / "belebele_eval_en_300.jsonl"
DEFAULT_PRED_FILE = RESULTS_DIR / "logprob_7tier_predictions.jsonl"
DEFAULT_SUMMARY_FILE = RESULTS_DIR / "logprob_7tier_summary.json"

ALL_TIERS = [
    ("f16", "gemma-3-1b-it-f16.gguf", 16.0),
    ("Q8_0", "gemma-3-1b-it-Q8_0.gguf", 8.5),
    ("Q6_K", "gemma-3-1b-it-Q6_K.gguf", 6.5),
    ("Q5_K_M", "gemma-3-1b-it-Q5_K_M.gguf", 5.5),
    ("Q4_K_M", "gemma-3-1b-it-Q4_K_M.gguf", 4.5),
    ("Q3_K_M", "gemma-3-1b-it-Q3_K_M.gguf", 3.5),
    ("Q2_K", "gemma-3-1b-it-Q2_K.gguf", 2.5),
]

CANDIDATE_DIGITS = ["1", "2", "3", "4"]
BN_TO_EN_DIGIT = {"১": "1", "২": "2", "৩": "3", "৪": "4"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GGUF 7-tier logprob degradation curve")
    p.add_argument("--tiers", nargs="+", default=None, help="Specific tiers to evaluate (e.g. f16 Q4_K_M)")
    p.add_argument("--languages", nargs="+", choices=("bangla", "english"), default=["bangla", "english"])
    p.add_argument("--limit", type=int, default=None, help="Limit items per language (e.g. 5 for quick test)")
    p.add_argument("--port", type=int, default=8092, help="Port for llama-server")
    p.add_argument("--threads", type=int, default=4, help="CPU threads per llama-server instance")
    p.add_argument("--output-file", type=Path, default=DEFAULT_PRED_FILE)
    p.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
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


def format_user_prompt(passage: str, instruction: str) -> str:
    """Format prompt with symmetric English framing for both languages."""
    return (
        "Read the following passage carefully and answer the question based ONLY on facts directly stated "
        "in the passage.\n\n"
        f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
        f"{instruction.strip()}\n\n"
        "Answer with option number only:"
    )


def logsumexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf") and b == float("-inf"):
        return float("-inf")
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def extract_candidate_logprobs(top_logprobs: list[dict]) -> tuple[dict[str, float], str, float, float]:
    """Extract log-probabilities for candidates 1, 2, 3, 4 from top_logprobs.
    
    Returns:
        candidate_lps: dict mapping '1'..'4' to logprob
        best_option: argmax option ('1'..'4')
        confidence_margin: best_lp - second_best_lp
        entropy: predictive entropy over the 4 candidates
    """
    candidate_lps = {d: -99.0 for d in CANDIDATE_DIGITS}

    for entry in top_logprobs:
        raw_token = entry.get("token", "")
        clean_t = raw_token.strip()
        clean_t = BN_TO_EN_DIGIT.get(clean_t, clean_t)
        
        if clean_t in candidate_lps:
            lp = float(entry["logprob"])
            if candidate_lps[clean_t] == -99.0:
                candidate_lps[clean_t] = lp
            else:
                candidate_lps[clean_t] = logsumexp(candidate_lps[clean_t], lp)

    sorted_candidates = sorted(candidate_lps.items(), key=lambda x: x[1], reverse=True)
    best_option, best_lp = sorted_candidates[0]
    second_best_lp = sorted_candidates[1][1]
    confidence_margin = best_lp - second_best_lp

    max_lp = max(candidate_lps.values())
    sum_exp = sum(math.exp(lp - max_lp) for lp in candidate_lps.values())
    log_denom = max_lp + math.log(sum_exp)
    
    entropy = 0.0
    for lp in candidate_lps.values():
        norm_p = math.exp(lp - log_denom)
        if norm_p > 1e-12:
            entropy -= norm_p * math.log2(norm_p)

    return candidate_lps, best_option, round(confidence_margin, 4), round(entropy, 4)


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


def query_server_logprobs(prompt: str, port: int, max_retries: int = 3, timeout_sec: int = 120) -> tuple[list[dict], float]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "("}
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20
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
            top_lp = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            return top_lp, lat_ms
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
            
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


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half) * 100, min(1.0, center + half) * 100


def main():
    args = parse_args()
    print("=" * 70, flush=True)
    print("Gemma 3 1B IT — 7-Tier Logprob Degradation Evaluator", flush=True)
    print(f"Languages:   {args.languages}", flush=True)
    print(f"Limit:       {args.limit if args.limit else 'All (300)'}", flush=True)
    print(f"Output File: {args.output_file}", flush=True)
    print(f"Port:        {args.port}", flush=True)
    print("=" * 70, flush=True)

    selected_tiers = []
    for tier_name, filename, bpw in ALL_TIERS:
        if args.tiers is None or tier_name in args.tiers:
            model_file = GGUF_DIR / filename
            selected_tiers.append((tier_name, model_file, bpw))

    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()

    existing_keys = set() if args.overwrite else load_existing_keys(args.output_file)
    summary_data = []

    for tier_name, model_file, bpw in selected_tiers:
        if not model_file.exists():
            print(f"\n[SKIP] Tier '{tier_name}' not found: {model_file.name}", flush=True)
            continue

        print(f"\n{'='*70}", flush=True)
        print(f"STARTING TIER: {tier_name} (approx {bpw} bpw)", flush=True)
        print(f"Model file: {model_file.name} ({model_file.stat().st_size / (1024*1024):.1f} MB)", flush=True)
        print(f"{'='*70}", flush=True)

        print(f"Launching llama-server on port {args.port} with {args.threads} threads ...", flush=True)
        proc = start_llama_server(model_file, args.port, threads=args.threads)
        ready = wait_for_server(args.port, timeout_sec=60)
        if not ready:
            print(f"ERROR: Server failed to start for {tier_name} within 60s!", flush=True)
            proc.kill()
            continue
        print("Server is ready and healthy!", flush=True)

        try:
            for lang in args.languages:
                items = load_dataset(lang, limit=args.limit)
                print(f"\n[{tier_name} | {lang.upper()}] Evaluating {len(items)} items with Logprob Scoring ...", flush=True)

                total = 0
                correct = 0
                hallucinated = 0
                total_latency = 0.0
                total_entropy = 0.0

                with args.output_file.open("a", encoding="utf-8") as out_f:
                    for idx, item in enumerate(items, start=1):
                        item_id = item["id"]
                        if (tier_name, lang, item_id) in existing_keys:
                            continue

                        prompt = format_user_prompt(item["passage"], item["instruction"])
                        top_lp, lat_ms = query_server_logprobs(prompt, args.port)
                        cand_lps, pred_opt, margin, entropy = extract_candidate_logprobs(top_lp)

                        gold_opt = str(item["correct_option"]).strip()
                        is_correct = (pred_opt == gold_opt)
                        is_distractor = not is_correct

                        total += 1
                        total_latency += lat_ms
                        total_entropy += entropy
                        if is_correct:
                            correct += 1
                        else:
                            hallucinated += 1

                        record = {
                            "tier": tier_name,
                            "bpw": bpw,
                            "language": lang,
                            "id": item_id,
                            "gold_option": gold_opt,
                            "pred_option": pred_opt,
                            "is_correct": is_correct,
                            "is_distractor_hallucination": is_distractor,
                            "candidate_logprobs": cand_lps,
                            "confidence_margin": margin,
                            "predictive_entropy": entropy,
                            "latency_ms": round(lat_ms, 2),
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        existing_keys.add((tier_name, lang, item_id))

                        if idx % 50 == 0 or idx == len(items):
                            acc = (correct / total * 100) if total else 0.0
                            hall_pct = (hallucinated / total * 100) if total else 0.0
                            print(f"  Processed {idx}/{len(items)} -> Acc: {acc:.1f}% | Distractor: {hall_pct:.1f}% | Avg Latency: {total_latency/total:.1f}ms", flush=True)

                if total > 0:
                    acc_pct = correct / total * 100
                    ci_l, ci_u = wilson_ci(correct, total)
                    tier_lang_summary = {
                        "tier": tier_name,
                        "bpw": bpw,
                        "language": lang,
                        "evaluated_items": total,
                        "correct_items": correct,
                        "accuracy_pct": round(acc_pct, 2),
                        "wilson_ci_95": [round(ci_l, 2), round(ci_u, 2)],
                        "hallucination_pct": round(hallucinated / total * 100, 2),
                        "avg_predictive_entropy": round(total_entropy / total, 3),
                        "avg_latency_ms": round(total_latency / total, 2),
                    }
                    summary_data.append(tier_lang_summary)
                    print(f"[{tier_name} | {lang.upper()} FINAL] Acc: {tier_lang_summary['accuracy_pct']}% [{ci_l:.1f}, {ci_u:.1f}] | Distractor: {tier_lang_summary['hallucination_pct']}% | Latency: {tier_lang_summary['avg_latency_ms']} ms", flush=True)

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
        print(f"\nSummary report saved to: {args.summary_file}", flush=True)


if __name__ == "__main__":
    main()
