# -*- coding: utf-8 -*-
"""Task 5: Perplexity Normalisation and 3-Part Spread Estimation.

Computes:
1. Characters (Unicode code points), UTF-8 bytes, and Gemma tokens for the Belebele passage texts.
2. Bits per character (BPC) and bits per byte (BPB) for each tier, and % increase over f16.
3. Splits each passage file into three equal parts (new files).
4. Runs llama-perplexity (context 512) on each part for f16, Q4_K_M, Q3_K_M, and Q2_K.
5. Computes relative increase vs f16 per part to estimate spread.
6. Saves audit/perplexity_normalised.csv.
"""

import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
AUDIT_DIR = ROOT / "audit"
OUT_CSV = AUDIT_DIR / "perplexity_normalised.csv"
LLAMA_PPL = ROOT / "llama.cpp" / "llama-perplexity.exe"
LLAMA_TOK = ROOT / "llama.cpp" / "llama-tokenize.exe"
MODEL_DIR = ROOT / "models" / "gguf"

PASSAGE_BN = AUDIT_DIR / "belebele_passages_bn.txt"
PASSAGE_EN = AUDIT_DIR / "belebele_passages_en.txt"
EXISTING_PPL_CSV = AUDIT_DIR / "perplexity.csv"

ALL_TIERS = [
    ("f16", "gemma-3-1b-it-f16.gguf", 16.0),
    ("Q8_0", "gemma-3-1b-it-Q8_0.gguf", 8.5),
    ("Q6_K", "gemma-3-1b-it-Q6_K.gguf", 6.5),
    ("Q5_K_M", "gemma-3-1b-it-Q5_K_M.gguf", 5.5),
    ("Q4_K_M", "gemma-3-1b-it-Q4_K_M.gguf", 4.5),
    ("Q3_K_M", "gemma-3-1b-it-Q3_K_M.gguf", 3.5),
    ("Q2_K", "gemma-3-1b-it-Q2_K.gguf", 2.5),
]

SPREAD_TIERS = ["f16", "Q4_K_M", "Q3_K_M", "Q2_K"]


def get_token_count(text_file: Path) -> int:
    cmd = [
        str(LLAMA_TOK),
        "-m", str(MODEL_DIR / "gemma-3-1b-it-f16.gguf"),
        "-f", str(text_file),
        "--show-count"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    m = re.search(r"Total number of tokens:\s*(\d+)", res.stdout + "\n" + res.stderr)
    if m:
        return int(m.group(1))
    raise RuntimeError(f"Could not parse token count for {text_file}")


def split_passage_files():
    parts_map = {}
    for lang, path in [("bn", PASSAGE_BN), ("en", PASSAGE_EN)]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        passages = text.split("\n\n")
        n = len(passages)
        # 253 passages: 84, 84, 85
        p1 = passages[:84]
        p2 = passages[84:168]
        p3 = passages[168:]
        parts = [p1, p2, p3]
        parts_map[lang] = []
        for idx, plist in enumerate(parts, 1):
            out_p = AUDIT_DIR / f"belebele_passages_{lang}_part{idx}.txt"
            with open(out_p, "w", encoding="utf-8") as out_f:
                out_f.write("\n\n".join(plist))
            parts_map[lang].append((idx, out_p, len(plist)))
            content_len = len('\n\n'.join(plist))
            print(f"Created {out_p.name}: {len(plist)} passages, {content_len} chars")
    return parts_map


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
    print("TASK 5: PERPLEXITY NORMALISATION AND SPREAD ESTIMATION")
    print("=" * 80)

    # 1. Text measurements
    stats = {}
    for lang, path in [("bangla", PASSAGE_BN), ("english", PASSAGE_EN)]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))
        n_tokens = get_token_count(path)
        stats[lang] = {
            "chars": n_chars,
            "bytes": n_bytes,
            "tokens": n_tokens,
            "bytes_per_char": n_bytes / n_chars,
            "tokens_per_char": n_tokens / n_chars,
            "chars_per_token": n_chars / n_tokens,
            "bytes_per_token": n_bytes / n_tokens,
        }
        print(f"\n[{lang.upper()}] Text Statistics:")
        print(f"  Characters (code points) : {n_chars:,}")
        print(f"  UTF-8 Bytes              : {n_bytes:,}")
        print(f"  Gemma Tokens             : {n_tokens:,}")
        print(f"  Bytes / Char             : {n_bytes / n_chars:.4f}")
        print(f"  Tokens / Char            : {n_tokens / n_chars:.4f}")
        print(f"  Chars / Token            : {n_chars / n_tokens:.4f}")
        print(f"  Bytes / Token            : {n_bytes / n_tokens:.4f}")

    # 2. Existing PPL and BPC/BPB normalisation across all 7 tiers
    existing_ppl = {}
    with open(EXISTING_PPL_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ppl[(row["tier"], row["language"])] = float(row["ppl"])

    print("\n" + "=" * 80)
    print("ALL 7 TIERS: BPC AND BPB NORMALISATION")
    print("=" * 80)
    normalised_rows = []
    f16_bpc = {}
    f16_bpb = {}

    for tier, filename, bpw in ALL_TIERS:
        for lang in ["bangla", "english"]:
            ppl = existing_ppl[(tier, lang)]
            # Bits per token = log2(PPL)
            bpt = math.log2(ppl)
            # Total bits = tokens * bpt
            total_bits = stats[lang]["tokens"] * bpt
            bpc = total_bits / stats[lang]["chars"]
            bpb = total_bits / stats[lang]["bytes"]

            if tier == "f16":
                f16_bpc[lang] = bpc
                f16_bpb[lang] = bpb
                inc_bpc = 0.0
                inc_bpb = 0.0
            else:
                inc_bpc = ((bpc - f16_bpc[lang]) / f16_bpc[lang]) * 100.0
                inc_bpb = ((bpb - f16_bpb[lang]) / f16_bpb[lang]) * 100.0

            normalised_rows.append({
                "type": "full_corpus",
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "part": "full",
                "ppl": round(ppl, 4),
                "bits_per_token": round(bpt, 4),
                "bits_per_char": round(bpc, 4),
                "bits_per_byte": round(bpb, 4),
                "inc_bpc_vs_f16_pct": round(inc_bpc, 2),
                "inc_bpb_vs_f16_pct": round(inc_bpb, 2),
                "rel_ppl_inc_vs_f16_pct": round(((ppl - existing_ppl[("f16", lang)]) / existing_ppl[("f16", lang)]) * 100.0, 2),
            })

    print(f"{'Tier':<8} | {'Lang':<7} | {'PPL':<9} | {'BPC':<8} | {'BPB':<8} | {'BPC Inc vs f16':<16} | {'PPL Inc vs f16':<16}")
    print("-" * 80)
    for r in normalised_rows:
        print(f"{r['tier']:<8} | {r['language']:<7} | {r['ppl']:<9.4f} | {r['bits_per_char']:<8.4f} | {r['bits_per_byte']:<8.4f} | {r['inc_bpc_vs_f16_pct']:>+6.2f}%          | {r['rel_ppl_inc_vs_f16_pct']:>+6.2f}%")

    # 3. Split passage files
    print("\n" + "=" * 80)
    print("SPLITTING PASSAGE FILES INTO 3 EQUAL PARTS")
    print("=" * 80)
    parts_map = split_passage_files()

    # 4. Run llama-perplexity on each part for f16, Q4_K_M, Q3_K_M, Q2_K
    print("\n" + "=" * 80)
    print("RUNNING 3-PART PERPLEXITY ON SPREAD TIERS (f16, Q4_K_M, Q3_K_M, Q2_K)")
    print("=" * 80)
    # Load existing spread_ppl if OUT_CSV exists
    spread_ppl = {}
    spread_rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["type"] == "part_spread" and r["ppl"]:
                    # extract part int
                    p_num = int(r["part"].replace("part", ""))
                    spread_ppl[(r["tier"], r["language"], p_num)] = float(r["ppl"])
                    spread_rows.append(r)
        print(f"Loaded {len(spread_ppl)} existing part perplexity calculations.")

    for tier in SPREAD_TIERS:
        fname = [f for t, f, _ in ALL_TIERS if t == tier][0]
        mpath = MODEL_DIR / fname
        bpw = [b for t, _, b in ALL_TIERS if t == tier][0]

        for lang, lang_code in [("bangla", "bn"), ("english", "en")]:
            for part_idx, part_file, n_p in parts_map[lang_code]:
                if (tier, lang, part_idx) in spread_ppl:
                    print(f"Skipping {tier} {lang} part {part_idx} (already computed: PPL = {spread_ppl[(tier, lang, part_idx)]})")
                    continue
                print(f"Running {tier} on {lang} part {part_idx} ({n_p} passages) ...", flush=True)
                ppl, err, elapsed = run_ppl(mpath, part_file, threads=8)
                spread_ppl[(tier, lang, part_idx)] = ppl
                print(f"  -> PPL = {ppl:.4f} +/- {err:.4f} ({elapsed:.1f}s)", flush=True)

                if tier == "f16":
                    rel_inc = 0.0
                else:
                    ref_ppl = spread_ppl[("f16", lang, part_idx)]
                    rel_inc = ((ppl - ref_ppl) / ref_ppl) * 100.0

                bpt = math.log2(ppl)
                spread_rows.append({
                    "type": "part_spread",
                    "tier": tier,
                    "bpw": bpw,
                    "language": lang,
                    "part": f"part{part_idx}",
                    "ppl": round(ppl, 4),
                    "bits_per_token": round(bpt, 4),
                    "bits_per_char": "",
                    "bits_per_byte": "",
                    "inc_bpc_vs_f16_pct": "",
                    "inc_bpb_vs_f16_pct": "",
                    "rel_ppl_inc_vs_f16_pct": round(rel_inc, 2),
                })

    # Save everything to OUT_CSV
    fieldnames = [
        "type", "tier", "bpw", "language", "part", "ppl", "bits_per_token",
        "bits_per_char", "bits_per_byte", "inc_bpc_vs_f16_pct", "inc_bpb_vs_f16_pct", "rel_ppl_inc_vs_f16_pct"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in normalised_rows:
            writer.writerow(r)
        for r in spread_rows:
            writer.writerow(r)

    print(f"\nSuccessfully written normalised perplexity and spread results to: {OUT_CSV}")


if __name__ == "__main__":
    main()
