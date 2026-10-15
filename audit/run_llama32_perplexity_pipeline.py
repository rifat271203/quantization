# -*- coding: utf-8 -*-
"""Perplexity and Information Density Benchmark for Llama-3.2-1B-Instruct across Quantization Tiers."""

import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
AUDIT_DIR = ROOT / "audit"
OUT_CSV = AUDIT_DIR / "llama32_perplexity_normalised.csv"
LLAMA_PPL = ROOT / "llama.cpp" / "llama-perplexity.exe"
LLAMA_TOK = ROOT / "llama.cpp" / "llama-tokenize.exe"
MODEL_DIR = ROOT / "models" / "gguf"

PASSAGE_BN_P1 = AUDIT_DIR / "belebele_passages_bn_part1.txt"
PASSAGE_EN_P1 = AUDIT_DIR / "belebele_passages_en_part1.txt"

TIERS = [
    ("f16",    "Llama-3.2-1B-Instruct-F16.gguf",    16.0),
    ("Q8_0",   "Llama-3.2-1B-Instruct-Q8_0.gguf",   8.5),
    ("Q6_K",   "Llama-3.2-1B-Instruct-Q6_K.gguf",   6.5),
    ("Q5_K_M", "Llama-3.2-1B-Instruct-Q5_K_M.gguf", 5.5),
    ("Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 4.5),
    ("Q3_K_M", "Llama-3.2-1B-Instruct-Q3_K_M.gguf", 3.5),
    ("Q2_K",   "Llama-3.2-1B-Instruct-Q2_K.gguf",   2.5),
]

def get_token_count(model_path: Path, text_file: Path) -> int:
    cmd = [
        str(LLAMA_TOK),
        "-m", str(model_path),
        "-f", str(text_file),
        "--show-count"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    m = re.search(r"Total number of tokens:\s*(\d+)", res.stdout + "\n" + res.stderr)
    if m:
        return int(m.group(1))
    raise RuntimeError(f"Could not parse token count for {text_file.name}")

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
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", output)
    if m:
        return float(m.group(1)), float(m.group(2)), elapsed
    m2 = re.search(r"PPL\s*=\s*([0-9.]+)", output)
    if m2:
        return float(m2.group(1)), 0.0, elapsed
    raise RuntimeError(f"Could not parse PPL for {text_file.name}:\n{output[-1000:]}")

def main():
    print("=" * 80)
    print("LLAMA 3.2 1B — PERPLEXITY NORMALISATION & DEGRADATION BENCHMARK")
    print("=" * 80)

    # 1. Measure text characteristics
    f16_model = MODEL_DIR / "Llama-3.2-1B-Instruct-F16.gguf"
    text_stats = {}
    for lang, path in [("bangla", PASSAGE_BN_P1), ("english", PASSAGE_EN_P1)]:
        with open(path, encoding="utf-8") as f:
            t = f.read()
        n_chars = len(t)
        n_bytes = len(t.encode("utf-8"))
        n_toks = get_token_count(f16_model, path)
        text_stats[lang] = {"chars": n_chars, "bytes": n_bytes, "tokens": n_toks}
        print(f"[{lang.upper()}] Chars: {n_chars:,} | Bytes: {n_bytes:,} | Llama Tokens: {n_toks:,} | Tok/Char: {n_toks/n_chars:.3f}")

    # Load existing results if any
    existing_results = {}
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                existing_results[(row["tier"], row["language"])] = {
                    "ppl": float(row["ppl"]),
                    "std_error": float(row.get("std_error", 0.0)),
                    "elapsed_s": float(row.get("elapsed_s", 0.0)),
                }

    # Run perplexity across tiers
    raw_results = {}
    for tier, filename, bpw in TIERS:
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            print(f"Model file {filename} not found, skipping {tier}.")
            continue

        for lang, path in [("english", PASSAGE_EN_P1), ("bangla", PASSAGE_BN_P1)]:
            key = (tier, lang)
            if key in existing_results:
                raw_results[key] = existing_results[key]
                print(f"[{tier} - {lang}] Already cached: PPL = {raw_results[key]['ppl']:.4f}")
                continue

            print(f"[{tier} ({bpw}b) - {lang.upper()}] Running llama-perplexity on {path.name}...")
            ppl, stderr, elap = run_ppl(model_path, path, threads=8)
            raw_results[key] = {"ppl": ppl, "std_error": stderr, "elapsed_s": elap}
            print(f"[{tier} - {lang}] PPL = {ppl:.4f} +/- {stderr:.4f} (took {elap:.1f}s)")

    # Compute normalized BPC, BPB, and Relative PPL increase
    f16_bn_ppl = raw_results.get(("f16", "bangla"), {}).get("ppl", 1.0)
    f16_en_ppl = raw_results.get(("f16", "english"), {}).get("ppl", 1.0)

    rows = []
    for tier, filename, bpw in TIERS:
        for lang in ["bangla", "english"]:
            key = (tier, lang)
            if key not in raw_results:
                continue
            item = raw_results[key]
            ppl = item["ppl"]
            bpt = math.log2(ppl)
            tot_bits = text_stats[lang]["tokens"] * bpt
            bpc = tot_bits / text_stats[lang]["chars"]
            bpb = tot_bits / text_stats[lang]["bytes"]

            ref_ppl = f16_bn_ppl if lang == "bangla" else f16_en_ppl
            rel_inc = ((ppl - ref_ppl) / ref_ppl) * 100.0

            rows.append({
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "part": "part1",
                "ppl": round(ppl, 4),
                "std_error": round(item.get("std_error", 0.0), 4),
                "bits_per_token": round(bpt, 4),
                "bits_per_char": round(bpc, 4),
                "bits_per_byte": round(bpb, 4),
                "rel_ppl_inc_vs_f16_pct": round(rel_inc, 2),
                "elapsed_s": round(item.get("elapsed_s", 0.0), 1),
            })

    # Save to CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved normalized perplexity table to: {OUT_CSV.name}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Tier':<8} | {'Lang':<7} | {'PPL':<9} | {'Bits/Tok':<9} | {'BPC':<8} | {'BPB':<8} | {'Rel PPL Inc vs f16':<18}")
    print("-" * 80)
    for r in rows:
        print(f"{r['tier']:<8} | {r['language']:<7} | {r['ppl']:<9.4f} | {r['bits_per_token']:<9.4f} | {r['bits_per_char']:<8.4f} | {r['bits_per_byte']:<8.4f} | {r['rel_ppl_inc_vs_f16_pct']:>+6.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
