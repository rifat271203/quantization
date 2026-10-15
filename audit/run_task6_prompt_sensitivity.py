# -*- coding: utf-8 -*-
"""Task 6: Prompt Sensitivity Evaluation for f16 and Q4_K_M on 300 Belebele items."""

import csv
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_BN = ROOT / "data_v2" / "belebele_eval_bn_300.jsonl"
DATA_EN = ROOT / "data_v2" / "belebele_eval_en_300.jsonl"
OUT_CSV = ROOT / "audit" / "prompt_variants.csv"
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"

CANDIDATES = ["1", "2", "3", "4"]
BN_TO_EN = {"১": "1", "২": "2", "৩": "3", "৪": "4"}

def logsumexp(a: float, b: float) -> float:
    if a == float("-inf") and b == float("-inf"):
        return float("-inf")
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def extract_pred(top_lp: list[dict]) -> str:
    cand_lps = {c: -99.0 for c in CANDIDATES}
    for entry in top_lp:
        tok = entry["token"].strip()
        tok = BN_TO_EN.get(tok, tok)
        if tok in cand_lps:
            lp = float(entry["logprob"])
            if cand_lps[tok] == -99.0:
                cand_lps[tok] = lp
            else:
                cand_lps[tok] = logsumexp(cand_lps[tok], lp)
    return max(cand_lps.items(), key=lambda x: x[1])[0]

def query_server(messages: list[dict], port: int) -> str:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    top_lp = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return extract_pred(top_lp)

def evaluate_variant(model_path: Path, port: int, variant_name: str, items_bn: list[dict], items_en: list[dict], threads: int = 6) -> dict:
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

    scores = {}
    try:
        # Bangla
        corr_bn = 0
        for idx, item in enumerate(items_bn, 1):
            gold = str(item["correct_option"]).strip()
            if variant_name == "Variant_A":
                # Same English instruction, NO assistant prefill
                prompt = (
                    "Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\n"
                    f"Passage:\n\"\"\"\n{item['passage'].strip()}\n\"\"\"\n\n"
                    f"{item['instruction'].strip()}\n\n"
                    "Answer with option number only:"
                )
                msgs = [{"role": "user", "content": prompt}]
            elif variant_name == "Variant_B":
                # Bengali instruction for Bangla items, with prefill '('
                prompt = (
                    "অনুগ্রহ করে নিচের অনুচ্ছেদটি মনোযোগ দিয়ে পড়ুন এবং শুধুমাত্র এই অনুচ্ছেদের তথ্যের ভিত্তিতে প্রশ্নের সঠিক বিকল্পটি বেছে নিন।\n\n"
                    f"অনুচ্ছেদ:\n\"\"\"\n{item['passage'].strip()}\n\"\"\"\n\n"
                    f"{item['instruction'].strip()}\n\n"
                    "সঠিক উত্তর:"
                )
                msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "("}]
            
            pred = query_server(msgs, port)
            if pred == gold:
                corr_bn += 1
            if idx % 100 == 0 or idx == len(items_bn):
                print(f"  [{variant_name} | BN] {idx}/{len(items_bn)} -> Acc: {corr_bn/idx*100:.1f}%", flush=True)

        scores["bangla"] = round(corr_bn / len(items_bn) * 100, 2)

        # English
        corr_en = 0
        for idx, item in enumerate(items_en, 1):
            gold = str(item["correct_option"]).strip()
            if variant_name == "Variant_A":
                # Same English instruction, NO assistant prefill
                prompt = (
                    "Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\n"
                    f"Passage:\n\"\"\"\n{item['passage'].strip()}\n\"\"\"\n\n"
                    f"{item['instruction'].strip()}\n\n"
                    "Answer with option number only:"
                )
                msgs = [{"role": "user", "content": prompt}]
            elif variant_name == "Variant_B":
                # English instruction for English items, with prefill '('
                prompt = (
                    "Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\n"
                    f"Passage:\n\"\"\"\n{item['passage'].strip()}\n\"\"\"\n\n"
                    f"{item['instruction'].strip()}\n\n"
                    "Answer with option number only:"
                )
                msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "("}]
            
            pred = query_server(msgs, port)
            if pred == gold:
                corr_en += 1
            if idx % 100 == 0 or idx == len(items_en):
                print(f"  [{variant_name} | EN] {idx}/{len(items_en)} -> Acc: {corr_en/idx*100:.1f}%", flush=True)

        scores["english"] = round(corr_en / len(items_en) * 100, 2)

    finally:
        proc.terminate()
        proc.wait()

    return scores

def main():
    print("=" * 80)
    print("TASK 6: PROMPT SENSITIVITY EVALUATION (f16 and Q4_K_M on 300 items)")
    print("=" * 80)

    with open(DATA_BN, encoding="utf-8") as f:
        items_bn = [json.loads(line) for line in f if line.strip()]
    with open(DATA_EN, encoding="utf-8") as f:
        items_en = [json.loads(line) for line in f if line.strip()]

    f16_model = ROOT / "models" / "gguf" / "gemma-3-1b-it-f16.gguf"
    q4_model = ROOT / "models" / "gguf" / "gemma-3-1b-it-Q4_K_M.gguf"

    # Baseline results from results/logprob_7tier_predictions.jsonl
    # f16: BN 32.33%, EN 47.33%
    # Q4_K_M: BN 32.67%, EN 44.67%
    baseline = {
        "f16": {"bangla": 32.33, "english": 47.33},
        "Q4_K_M": {"bangla": 32.67, "english": 44.67},
    }

    results = []
    for tier_name, model_path, port in [("f16", f16_model, 8111), ("Q4_K_M", q4_model, 8112)]:
        print(f"\n{'='*70}\nEVALUATING TIER: {tier_name}\n{'='*70}", flush=True)
        # Variant A
        print(f"\n--- Running Variant A (No Assistant Prefill '(') on {tier_name} ---", flush=True)
        scores_a = evaluate_variant(model_path, port, "Variant_A", items_bn, items_en, threads=6)
        print(f"Variant A Result: BN = {scores_a['bangla']}%, EN = {scores_a['english']}%", flush=True)

        # Variant B
        print(f"\n--- Running Variant B (Bengali Instruction for BN) on {tier_name} ---", flush=True)
        scores_b = evaluate_variant(model_path, port, "Variant_B", items_bn, items_en, threads=6)
        print(f"Variant B Result: BN = {scores_b['bangla']}%, EN = {scores_b['english']}%", flush=True)

        results.append({
            "tier": tier_name,
            "baseline_bn": baseline[tier_name]["bangla"],
            "baseline_en": baseline[tier_name]["english"],
            "var_a_bn": scores_a["bangla"],
            "var_a_en": scores_a["english"],
            "var_b_bn": scores_b["bangla"],
            "var_b_en": scores_b["english"],
        })

    print("\n" + "=" * 95)
    print("PROMPT SENSITIVITY COMPARISON TABLE")
    print("=" * 95)
    print(f"{'Tier':<8} | {'Current (EN inst + prefill)':<30} | {'Variant A (EN inst, NO prefill)':<32} | {'Variant B (Native BN/EN inst)':<30}")
    print(f"{'':<8} | {'BN Acc (%)':<14} {'EN Acc (%)':<14} | {'BN Acc (%)':<15} {'EN Acc (%)':<15} | {'BN Acc (%)':<14} {'EN Acc (%)':<14}")
    print("-" * 95)
    for r in results:
        t = r["tier"]
        print(f"{t:<8} | {r['baseline_bn']:<14.2f} {r['baseline_en']:<14.2f} | {r['var_a_bn']:<15.2f} {r['var_a_en']:<15.2f} | {r['var_b_bn']:<14.2f} {r['var_b_en']:<14.2f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "language", "current_prompt_accuracy", "variant_a_accuracy", "variant_b_accuracy"])
        for r in results:
            t = r["tier"]
            writer.writerow([t, "bangla", r["baseline_bn"], r["var_a_bn"], r["var_b_bn"]])
            writer.writerow([t, "english", r["baseline_en"], r["var_a_en"], r["var_b_en"]])

    print(f"\nSaved prompt sensitivity results to: {OUT_CSV}")

if __name__ == "__main__":
    main()
