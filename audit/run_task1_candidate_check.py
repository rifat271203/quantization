# -*- coding: utf-8 -*-
"""Task 1: Candidate Token Probability Mass & Top-1 Analysis for f16 and Q4_K_M."""

import collections
import csv
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_BN = ROOT / "data_v2" / "belebele_eval_bn_300.jsonl"
DATA_EN = ROOT / "data_v2" / "belebele_eval_en_300.jsonl"
OUT_CSV = ROOT / "audit" / "candidate_mass.csv"
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"

ASCII_DIGITS = {"1", "2", "3", "4"}
BN_DIGITS = {"১", "২", "৩", "৪"}  # U+09E7 to U+09EA

def format_user_prompt(passage: str, instruction: str) -> str:
    return (
        "Read the following passage carefully and answer the question based ONLY on facts directly stated "
        "in the passage.\n\n"
        f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
        f"{instruction.strip()}\n\n"
        "Answer with option number only:"
    )

def query_item(prompt: str, port: int) -> list[dict]:
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
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    return res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

def evaluate_tier(model_path: Path, port: int, items_bn: list[dict], items_en: list[dict], threads: int = 8) -> dict:
    proc = subprocess.Popen([str(LLAMA_SERVER), "-m", str(model_path), "--port", str(port), "-t", str(threads), "-c", "2048", "--log-disable"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    results = {"bangla": [], "english": []}
    try:
        for lang, items in [("bangla", items_bn), ("english", items_en)]:
            print(f"Evaluating {lang} (50 items) on port {port}...", flush=True)
            for idx, item in enumerate(items, 1):
                prompt = format_user_prompt(item["passage"], item["instruction"])
                top_lp = query_item(prompt, port)
                
                # Top-1 token
                top1_token = top_lp[0]["token"]
                
                # Probabilities
                ascii_mass = 0.0
                bn_mass = 0.0
                for entry in top_lp:
                    t = entry["token"].strip()
                    p = math.exp(entry["logprob"])
                    if t in ASCII_DIGITS:
                        ascii_mass += p
                    if t in BN_DIGITS:
                        bn_mass += p
                
                is_top1_not_ascii_14 = (top1_token.strip() not in ASCII_DIGITS)
                
                results[lang].append({
                    "id": item["id"],
                    "top1_token": top1_token,
                    "ascii_mass": ascii_mass,
                    "bn_mass": bn_mass,
                    "is_top1_not_ascii_14": is_top1_not_ascii_14
                })
    finally:
        proc.terminate()
        proc.wait()

    return results

def main():
    import math
    # Load 50 BN and 50 EN items
    with open(DATA_BN, encoding="utf-8") as f:
        bn_50 = [json.loads(line) for idx, line in enumerate(f) if idx < 50]
    with open(DATA_EN, encoding="utf-8") as f:
        en_50 = [json.loads(line) for idx, line in enumerate(f) if idx < 50]

    print("=== TASK 1: Candidate Token Probability Mass & Top-1 Distribution ===")
    print(f"Loaded 50 BN and 50 EN items.")

    f16_model = ROOT / "models" / "gguf" / "gemma-3-1b-it-f16.gguf"
    q4_model = ROOT / "models" / "gguf" / "gemma-3-1b-it-Q4_K_M.gguf"

    print("\n--- Running f16 ---", flush=True)
    f16_res = evaluate_tier(f16_model, port=8105, items_bn=bn_50, items_en=en_50)

    print("\n--- Running Q4_K_M ---", flush=True)
    q4_res = evaluate_tier(q4_model, port=8106, items_bn=bn_50, items_en=en_50)

    all_data = [("f16", f16_res), ("Q4_K_M", q4_res)]

    # Analysis & Printing
    summary_rows = []
    top1_freqs = {}

    for tier_name, res in all_data:
        for lang in ["bangla", "english"]:
            items = res[lang]
            n = len(items)
            mean_ascii = float(np.mean([x["ascii_mass"] for x in items]))
            mean_bn = float(np.mean([x["bn_mass"] for x in items]))
            frac_top1_not_14 = float(np.mean([x["is_top1_not_ascii_14"] for x in items]))
            
            # Counts of top-1 tokens
            c = collections.Counter([x["top1_token"] for x in items])
            top1_freqs[(tier_name, lang)] = c.most_common(10)

            summary_rows.append({
                "tier": tier_name,
                "language": lang,
                "n_items": n,
                "mean_ascii_1_4_prob_mass": round(mean_ascii, 6),
                "mean_bn_digits_prob_mass": round(mean_bn, 6),
                "fraction_top1_not_in_1_4": round(frac_top1_not_14, 4),
            })

    print("\n" + "=" * 90)
    print("TASK 1 SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Tier':<8} | {'Lang':<7} | {'Mean Mass 1-4 (ASCII)':<23} | {'Mean Mass ১-৪ (BN)':<20} | {'Frac Top1 Not in 1-4':<22}")
    print("-" * 90)
    for r in summary_rows:
        print(f"{r['tier']:<8} | {r['language']:<7} | {r['mean_ascii_1_4_prob_mass']:<23.6f} | {r['mean_bn_digits_prob_mass']:<20.6f} | {r['fraction_top1_not_in_1_4']:<22.4f}")

    print("\n" + "=" * 90)
    print("10 MOST FREQUENT TOP-1 TOKENS PER LANGUAGE AND TIER")
    print("=" * 90)
    for (t, l), top10 in top1_freqs.items():
        print(f"\n[{t} | {l.upper()}] Top-10 Tokens:")
        for rank, (tok, count) in enumerate(top10, 1):
            print(f"  {rank:>2}. {tok!r:<12} : {count:>2} items ({count/50*100:.1f}%)")

    # Save to CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "language", "n_items", "mean_ascii_1_4_prob_mass", "mean_bn_digits_prob_mass", "fraction_top1_not_in_1_4"])
        for r in summary_rows:
            writer.writerow([r["tier"], r["language"], r["n_items"], r["mean_ascii_1_4_prob_mass"], r["mean_bn_digits_prob_mass"], r["fraction_top1_not_in_1_4"]])
        
        writer.writerow([])
        writer.writerow(["tier", "language", "rank", "token", "count", "frequency_pct"])
        for (t, l), top10 in top1_freqs.items():
            for rank, (tok, count) in enumerate(top10, 1):
                writer.writerow([t, l, rank, tok, count, round(count / 50 * 100, 2)])

    print(f"\nSaved Task 1 results to: {OUT_CSV}")

if __name__ == "__main__":
    main()
