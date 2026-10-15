# -*- coding: utf-8 -*-
"""Task 4: Run llama-perplexity across all 7 tiers for Belebele unique passages."""

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_BN = ROOT / "data_v2" / "belebele_eval_bn_300.jsonl"
DATA_EN = ROOT / "data_v2" / "belebele_eval_en_300.jsonl"
OUT_CSV = ROOT / "audit" / "perplexity.csv"
LLAMA_PPL = ROOT / "llama.cpp" / "llama-perplexity.exe"

PASSAGE_BN = ROOT / "audit" / "belebele_passages_bn.txt"
PASSAGE_EN = ROOT / "audit" / "belebele_passages_en.txt"

TIERS = [
    ("f16", "gemma-3-1b-it-f16.gguf", 16.0),
    ("Q8_0", "gemma-3-1b-it-Q8_0.gguf", 8.5),
    ("Q6_K", "gemma-3-1b-it-Q6_K.gguf", 6.5),
    ("Q5_K_M", "gemma-3-1b-it-Q5_K_M.gguf", 5.5),
    ("Q4_K_M", "gemma-3-1b-it-Q4_K_M.gguf", 4.5),
    ("Q3_K_M", "gemma-3-1b-it-Q3_K_M.gguf", 3.5),
    ("Q2_K", "gemma-3-1b-it-Q2_K.gguf", 2.5),
]

def build_passage_files() -> tuple[int, int]:
    with open(DATA_BN, encoding="utf-8") as f:
        bn_passages = list(dict.fromkeys(json.loads(line)["passage"].strip() for line in f))
    with open(DATA_EN, encoding="utf-8") as f:
        en_passages = list(dict.fromkeys(json.loads(line)["passage"].strip() for line in f))

    with open(PASSAGE_BN, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bn_passages))
    with open(PASSAGE_EN, "w", encoding="utf-8") as f:
        f.write("\n\n".join(en_passages))

    return len(bn_passages), len(en_passages)

def run_ppl(model_path: Path, text_file: Path, threads: int = 8) -> tuple[float, float, float]:
    cmd = [
        str(LLAMA_PPL),
        "-m", str(model_path),
        "-f", str(text_file),
        "-c", "512",
        "-t", str(threads),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - t0

    output = res.stdout + "\n" + res.stderr
    # Look for "Final estimate: PPL = 33.7711 +/- 1.08167"
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", output)
    if m:
        ppl = float(m.group(1))
        err = float(m.group(2))
        return ppl, err, elapsed

    # Fallback pattern
    m2 = re.search(r"PPL\s*=\s*([0-9.]+)", output)
    if m2:
        return float(m2.group(1)), 0.0, elapsed

    raise RuntimeError(f"Could not parse PPL from output:\n{output[-1000:]}")

def main():
    print("=" * 80)
    print("TASK 4: BELEBELE PERPLEXITY EVALUATION ACROSS 7 TIERS")
    print("=" * 80)

    n_bn, n_en = build_passage_files()
    print(f"Unique Bangla passages: {n_bn} (saved to {PASSAGE_BN})")
    print(f"Unique English passages: {n_en} (saved to {PASSAGE_EN})")

    results = []
    f16_ppl = {}

    for tier_name, filename, bpw in TIERS:
        model_path = ROOT / "models" / "gguf" / filename
        if not model_path.exists():
            print(f"[SKIP] {tier_name} ({filename}) not found!")
            continue

        for lang, text_file in [("bangla", PASSAGE_BN), ("english", PASSAGE_EN)]:
            print(f"\nRunning {tier_name} ({bpw} bpw) on {lang.upper()} ...", flush=True)
            ppl, err, elapsed = run_ppl(model_path, text_file, threads=8)
            print(f"  PPL = {ppl:.4f} +/- {err:.4f} (took {elapsed:.1f}s)", flush=True)

            if tier_name == "f16":
                f16_ppl[lang] = ppl
                rel_inc = 0.0
            else:
                rel_inc = ((ppl - f16_ppl[lang]) / f16_ppl[lang]) * 100.0

            results.append({
                "tier": tier_name,
                "bpw": bpw,
                "language": lang,
                "ppl": round(ppl, 4),
                "std_error": round(err, 4),
                "rel_increase_vs_f16_pct": round(rel_inc, 2),
                "elapsed_s": round(elapsed, 1),
            })

    print("\n" + "=" * 90)
    print("PERPLEXITY SUMMARY TABLE (Context = 512)")
    print("=" * 90)
    print(f"{'Tier':<8} | {'BPW':<6} | {'Lang':<7} | {'PPL':<10} | {'Std Error':<10} | {'Relative Inc vs f16 (%)':<25}")
    print("-" * 90)
    for r in results:
        rel_str = f"{r['rel_increase_vs_f16_pct']:+.2f}%" if r["tier"] != "f16" else "0.00% (ref)"
        print(f"{r['tier']:<8} | {r['bpw']:<6.1f} | {r['language']:<7} | {r['ppl']:<10.4f} | {r['std_error']:<10.4f} | {rel_str:<25}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "bpw", "language", "ppl", "std_error", "rel_increase_vs_f16_pct", "elapsed_s"])
        for r in results:
            writer.writerow([r["tier"], r["bpw"], r["language"], r["ppl"], r["std_error"], r["rel_increase_vs_f16_pct"], r["elapsed_s"]])

    print(f"\nSaved perplexity results to: {OUT_CSV}")

if __name__ == "__main__":
    main()
