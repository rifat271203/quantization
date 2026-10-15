# -*- coding: utf-8 -*-
"""Full 900-Item Belebele Logprob Evaluation Harness for Llama 3.2 1B Instruct across 7 Quantization Tiers.

Evaluates Meta Llama-3.2-1B-Instruct across:
- f16
- Q8_0
- Q6_K
- Q5_K_M
- Q4_K_M
- Q3_K_M
- Q2_K

Uses native llama-server logprob extraction (max_tokens=1, top_logprobs=20)
with prefilled assistant '(' and symmetric cross-lingual English directives.
Includes automatic resumption, error recovery, and robust scoring.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Force UTF-8 on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PARALLEL_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "llama32_logprob_predictions.jsonl"
SUMMARY_FILE = RESULTS_DIR / "llama32_logprob_summary.json"
GGUF_DIR = ROOT / "models" / "gguf"
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"

CANDIDATE_DIGITS = ["1", "2", "3", "4"]
BN_TO_EN_DIGIT = {"১": "1", "২": "2", "৩": "3", "৪": "4"}

TIERS = [
    ("f16",    "Llama-3.2-1B-Instruct-F16.gguf",    16.0),
    ("Q8_0",   "Llama-3.2-1B-Instruct-Q8_0.gguf",   8.5),
    ("Q6_K",   "Llama-3.2-1B-Instruct-Q6_K.gguf",   6.5),
    ("Q5_K_M", "Llama-3.2-1B-Instruct-Q5_K_M.gguf", 5.5),
    ("Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 4.5),
    ("Q3_K_M", "Llama-3.2-1B-Instruct-Q3_K_M.gguf", 3.5),
    ("Q2_K",   "Llama-3.2-1B-Instruct-Q2_K.gguf",   2.5),
]

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Llama 3.2 1B 7-Tier Belebele Logprobs")
    p.add_argument("--tiers", nargs="+", default=None, help="Specific tiers to evaluate (e.g. f16 Q4_K_M Q2_K)")
    p.add_argument("--languages", nargs="+", choices=("bangla", "english"), default=["bangla", "english"])
    p.add_argument("--limit", type=int, default=None, help="Limit items per language (e.g. 10 for smoke test)")
    p.add_argument("--port", type=int, default=8096, help="Port for llama-server")
    p.add_argument("--threads", type=int, default=6, help="CPU threads per llama-server instance")
    p.add_argument("--out", type=Path, default=OUT_FILE, help="Output JSONL predictions file")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    return p.parse_args()

def format_user_prompt(passage: str, instruction: str) -> str:
    return (
        "Read the following passage carefully and answer the question based ONLY on facts directly stated "
        "in the passage.\n\n"
        f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
        f"{instruction.strip()}\n\n"
        "Answer with option number only:"
    )

def logsumexp(a: float, b: float) -> float:
    if a == float("-inf") and b == float("-inf"):
        return float("-inf")
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def extract_candidate_logprobs(top_logprobs: list[dict]) -> tuple[dict[str, float], str, float, float]:
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

def start_llama_server(model_path: Path, port: int, threads: int = 6) -> subprocess.Popen:
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
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
    
    # In UTF-8, Bengali characters are multi-byte (3 bytes). When max_tokens=1, llama-server
    # generates only 1 byte token if the model outputs a Bengali character. Since 1 byte is an
    # incomplete UTF-8 sequence, llama-server's OpenAI API serializer omits the 'logprobs' object.
    # Falling back to max_tokens=2 or 4 allows llama-server to complete the character piece
    # and return the full logprobs for the first token step without error.
    for max_tok in [1, 2, 4]:
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": max_tok,
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
                    res = json.loads(resp.read().decode("utf-8", errors="replace"))
                lat_ms = (time.perf_counter() - t0) * 1000.0
                choice = res["choices"][0]
                lp_obj = choice.get("logprobs")
                if lp_obj and lp_obj.get("content") and len(lp_obj["content"]) > 0:
                    top_lp = lp_obj["content"][0].get("top_logprobs", [])
                    return top_lp, lat_ms
                # If logprobs was omitted by llama-server (incomplete UTF-8), try next max_tok
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0)
                
    # If logprobs was omitted across all token limits (or network/timeout error), return empty
    if last_err is not None and not isinstance(last_err, KeyError):
        raise RuntimeError(f"Server query failed after {max_retries} attempts: {last_err}")
    return [], 0.0

def load_dataset() -> list[dict]:
    items = []
    with open(PARALLEL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def main():
    args = parse_args()
    
    active_tiers = TIERS
    if args.tiers:
        tier_names = [t.lower() for t in args.tiers]
        active_tiers = [t for t in TIERS if t[0].lower() in tier_names]

    all_items = load_dataset()
    print("=" * 80)
    print("LLAMA 3.2 1B INSTRUCT — LOGPROB DEGRADATION EVALUATOR")
    print(f"Total Parallel Items Available: {len(all_items)}")
    print(f"Active Tiers: {[t[0] for t in active_tiers]}")
    print(f"Active Languages: {args.languages}")
    print(f"Output File: {args.out}")
    print("=" * 80)

    # Load existing completed keys: (tier, language, pair_id)
    existing_keys = set()
    if args.out.exists() and not args.overwrite:
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_keys.add((rec["tier"], rec["language"], rec["pair_id"]))
                    except Exception:
                        pass
        print(f"Found {len(existing_keys)} existing completed predictions in {args.out.name}")

    out_file_mode = "w" if args.overwrite else "a"
    out_fh = open(args.out, out_file_mode, encoding="utf-8", buffering=1)

    try:
        for tier_idx, (tier_name, filename, bpw) in enumerate(active_tiers, 1):
            model_path = GGUF_DIR / filename
            if not model_path.exists():
                print(f"\n[WARNING] Model file {filename} not found in {GGUF_DIR}. Skipping tier.")
                continue

            print(f"\n[{tier_idx}/{len(active_tiers)}] Starting llama-server for {tier_name} ({filename}, {bpw} bpw)...")
            proc = start_llama_server(model_path, port=args.port, threads=args.threads)

            try:
                healthy = wait_for_server(args.port, timeout_sec=60)
                if not healthy:
                    print(f"ERROR: Server for {tier_name} failed to become healthy on port {args.port}.")
                    continue
                print(f"Server healthy on port {args.port}! Running evaluation...")

                for lang in args.languages:
                    items_to_eval = all_items[:args.limit] if args.limit else all_items
                    correct_count = 0
                    evaluated_count = 0

                    print(f"\n--- Tier: {tier_name} | Language: {lang.upper()} ({len(items_to_eval)} items) ---")
                    t_start = time.time()

                    for idx, item in enumerate(items_to_eval, 1):
                        pair_id = item["pair_id"]
                        cache_key = (tier_name, lang, pair_id)
                        if cache_key in existing_keys:
                            continue

                        lang_key = "bn" if lang == "bangla" else "en"
                        sub = item[lang_key]
                        passage = sub["passage"]
                        question = sub["question"]
                        opts = sub["options"]
                        gold_str = str(sub["correct_answer_num"])

                        instr_parts = [question]
                        for opt_k in ["1", "2", "3", "4"]:
                            instr_parts.append(f"{opt_k}. {opts.get(opt_k, '')}")
                        instruction = "\n".join(instr_parts)
                        user_prompt = format_user_prompt(passage, instruction)

                        try:
                            top_lps, lat_ms = query_server_logprobs(user_prompt, port=args.port)
                            cand_lps, pred_opt, margin, ent = extract_candidate_logprobs(top_lps)
                            is_corr = (str(pred_opt) == gold_str)
                            if is_corr:
                                correct_count += 1
                            evaluated_count += 1

                            rec = {
                                "model": "meta-llama/Llama-3.2-1B-Instruct",
                                "tier": tier_name,
                                "bpw": bpw,
                                "language": lang,
                                "pair_id": pair_id,
                                "link": item.get("link", ""),
                                "passage": passage,
                                "predicted_option": str(pred_opt),
                                "gold_option": gold_str,
                                "is_correct": is_corr,
                                "confidence_margin": margin,
                                "predictive_entropy": ent,
                                "candidate_logprobs": cand_lps,
                                "latency_ms": round(lat_ms, 2),
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            }
                            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            existing_keys.add(cache_key)

                            if idx % 50 == 0 or idx == len(items_to_eval):
                                acc_so_far = (correct_count / evaluated_count * 100) if evaluated_count else 0
                                print(f"  [{idx}/{len(items_to_eval)}] Evaluated: {evaluated_count} | Correct: {correct_count} | Acc: {acc_so_far:.1f}%")

                        except Exception as e:
                            print(f"  Item {pair_id} failed: {e}")

                    elapsed = time.time() - t_start
                    acc_final = (correct_count / evaluated_count * 100) if evaluated_count else 0
                    print(f"Finished {tier_name} - {lang}: {evaluated_count} items evaluated in {elapsed:.1f}s | Acc: {acc_final:.2f}%")

            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(1.0)

    finally:
        out_fh.close()

    print("\n" + "=" * 80)
    print(f"Llama 3.2 1B evaluation batch finished! Predictions saved to: {args.out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
