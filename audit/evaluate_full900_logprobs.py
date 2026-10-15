# -*- coding: utf-8 -*-
"""Score the remaining 600 Belebele items across all 7 tiers and merge with existing 300 items."""

import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PARALLEL_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"
PRED_300_FILE = ROOT / "results" / "logprob_7tier_predictions.jsonl"
OUT_FILE = ROOT / "results" / "logprob_full900_predictions.jsonl"
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"

CANDIDATE_DIGITS = ["1", "2", "3", "4"]
BN_TO_EN_DIGIT = {"১": "1", "২": "2", "৩": "3", "৪": "4"}

TIERS = [
    ("f16", "gemma-3-1b-it-f16.gguf", 16.0),
    ("Q8_0", "gemma-3-1b-it-Q8_0.gguf", 8.5),
    ("Q6_K", "gemma-3-1b-it-Q6_K.gguf", 6.5),
    ("Q5_K_M", "gemma-3-1b-it-Q5_K_M.gguf", 5.5),
    ("Q4_K_M", "gemma-3-1b-it-Q4_K_M.gguf", 4.5),
    ("Q3_K_M", "gemma-3-1b-it-Q3_K_M.gguf", 3.5),
    ("Q2_K", "gemma-3-1b-it-Q2_K.gguf", 2.5),
]

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

def wait_for_server(port: int, timeout_sec: int = 60) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
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

def main():
    print("=" * 80)
    print("EVALUATING FULL 900-ITEM BELEBELE SET ACROSS 7 TIERS")
    print("=" * 80)

    # 1. Load all 900 parallel items
    all_items = []
    with open(PARALLEL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))
    print(f"Loaded {len(all_items)} total parallel items.")

    # Unique passages
    bn_passages = {x["bn"]["passage"].strip() for x in all_items}
    en_passages = {x["en"]["passage"].strip() for x in all_items}
    print(f"Unique Bangla passages among 900: {len(bn_passages)}")
    print(f"Unique English passages among 900: {len(en_passages)}")

    item_by_id = {x["pair_id"]: x for x in all_items}

    # 2. Check existing predictions in OUT_FILE or initialize from PRED_300_FILE
    existing_keys = set()
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_keys.add((rec["tier"], rec["language"], rec["id"]))
                    except Exception:
                        pass
        print(f"Found {len(existing_keys)} existing predictions in {OUT_FILE}.")
    else:
        # Initialize by copying the 300 existing predictions with passage attached
        print(f"Initializing {OUT_FILE} from {PRED_300_FILE}...")
        init_records = []
        with open(PRED_300_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    pair = item_by_id[rec["id"]]
                    lang_key = "bn" if rec["language"] == "bangla" else "en"
                    rec["passage"] = pair[lang_key]["passage"].strip()
                    init_records.append(rec)
                    existing_keys.add((rec["tier"], rec["language"], rec["id"]))
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            for rec in init_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(init_records)} initial records to {OUT_FILE}.")

    # Items to evaluate: the 600 items not in the primary 300
    items_to_eval = [x for x in all_items if not x.get("is_primary_300")]
    print(f"Items to score: {len(items_to_eval)} items per language per tier.")

    port = 8150
    for tier_name, filename, bpw in TIERS:
        model_path = ROOT / "models" / "gguf" / filename
        if not model_path.exists():
            print(f"[SKIP] Model {filename} not found!")
            continue

        # Check how many items need scoring for this tier
        needed_bn = [x for x in items_to_eval if (tier_name, "bangla", x["pair_id"]) not in existing_keys]
        needed_en = [x for x in items_to_eval if (tier_name, "english", x["pair_id"]) not in existing_keys]

        if not needed_bn and not needed_en:
            print(f"\n[TIER {tier_name}] All 600 items already evaluated for both languages. Skipping.")
            continue

        print(f"\n{'='*70}\nSTARTING TIER: {tier_name} ({bpw} bpw) -> Need BN: {len(needed_bn)}, EN: {len(needed_en)}\n{'='*70}", flush=True)

        # Launch server
        cmd = [
            str(LLAMA_SERVER),
            "-m", str(model_path),
            "--port", str(port),
            "-t", "8",
            "-c", "2048",
            "--log-disable",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        if not wait_for_server(port, timeout_sec=60):
            proc.terminate()
            raise RuntimeError(f"Server on port {port} failed to start for {tier_name}")

        try:
            # Open OUT_FILE in append mode
            with open(OUT_FILE, "a", encoding="utf-8") as out_f:
                # 1. Bangla
                if needed_bn:
                    print(f"\n--- Scoring {tier_name} on BANGLA ({len(needed_bn)} items) ---", flush=True)
                    corr_bn = 0
                    for idx, p in enumerate(needed_bn, 1):
                        opts = "\n".join([f"({k}) {v}" for k, v in p["bn"]["options"].items()])
                        instruction = f"{p['bn']['question']}\n\nবিকল্পসমূহ:\n{opts}"
                        prompt = format_user_prompt(p["bn"]["passage"], instruction)
                        gold = str(p["correct_answer_num"]).strip()

                        top_lp, lat_ms = query_server_logprobs(prompt, port)
                        cand_lps, best_opt, margin, entropy = extract_candidate_logprobs(top_lp)

                        is_corr = (best_opt == gold)
                        is_dist = (not is_corr and best_opt in CANDIDATE_DIGITS)
                        if is_corr:
                            corr_bn += 1

                        rec = {
                            "tier": tier_name,
                            "bpw": bpw,
                            "language": "bangla",
                            "id": p["pair_id"],
                            "passage": p["bn"]["passage"].strip(),
                            "gold_option": gold,
                            "pred_option": best_opt,
                            "is_correct": is_corr,
                            "is_distractor_hallucination": is_dist,
                            "candidate_logprobs": cand_lps,
                            "confidence_margin": margin,
                            "predictive_entropy": entropy,
                            "latency_ms": round(lat_ms, 2)
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                        existing_keys.add((tier_name, "bangla", p["pair_id"]))

                        if idx % 50 == 0 or idx == len(needed_bn):
                            print(f"  [{tier_name} | BN] {idx}/{len(needed_bn)} -> Running Acc: {corr_bn/idx*100:.1f}%", flush=True)

                # 2. English
                if needed_en:
                    print(f"\n--- Scoring {tier_name} on ENGLISH ({len(needed_en)} items) ---", flush=True)
                    corr_en = 0
                    for idx, p in enumerate(needed_en, 1):
                        opts = "\n".join([f"({k}) {v}" for k, v in p["en"]["options"].items()])
                        instruction = f"{p['en']['question']}\n\nOptions:\n{opts}"
                        prompt = format_user_prompt(p["en"]["passage"], instruction)
                        gold = str(p["correct_answer_num"]).strip()

                        top_lp, lat_ms = query_server_logprobs(prompt, port)
                        cand_lps, best_opt, margin, entropy = extract_candidate_logprobs(top_lp)

                        is_corr = (best_opt == gold)
                        is_dist = (not is_corr and best_opt in CANDIDATE_DIGITS)
                        if is_corr:
                            corr_en += 1

                        rec = {
                            "tier": tier_name,
                            "bpw": bpw,
                            "language": "english",
                            "id": p["pair_id"],
                            "passage": p["en"]["passage"].strip(),
                            "gold_option": gold,
                            "pred_option": best_opt,
                            "is_correct": is_corr,
                            "is_distractor_hallucination": is_dist,
                            "candidate_logprobs": cand_lps,
                            "confidence_margin": margin,
                            "predictive_entropy": entropy,
                            "latency_ms": round(lat_ms, 2)
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                        existing_keys.add((tier_name, "english", p["pair_id"]))

                        if idx % 50 == 0 or idx == len(needed_en):
                            print(f"  [{tier_name} | EN] {idx}/{len(needed_en)} -> Running Acc: {corr_en/idx*100:.1f}%", flush=True)

        finally:
            proc.terminate()
            proc.wait()
            time.sleep(1.0)

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE! Total records in {OUT_FILE}: {len(existing_keys)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
