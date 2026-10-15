# -*- coding: utf-8 -*-
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
AUDIT_DIR = ROOT / "audit"
LLAMA_TOK = ROOT / "llama.cpp" / "llama-tokenize.exe"
MODEL_F16 = ROOT / "models" / "gguf" / "gemma-3-1b-it-f16.gguf"

PASSAGE_BN = AUDIT_DIR / "belebele_passages_bn.txt"
PASSAGE_EN = AUDIT_DIR / "belebele_passages_en.txt"
EXISTING_PPL_CSV = AUDIT_DIR / "perplexity.csv"

ALL_TIERS = [
    ("f16", 16.0),
    ("Q8_0", 8.5),
    ("Q6_K", 6.5),
    ("Q5_K_M", 5.5),
    ("Q4_K_M", 4.5),
    ("Q3_K_M", 3.5),
    ("Q2_K", 2.5),
]

def get_token_count(text_file: Path) -> int:
    cmd = [
        str(LLAMA_TOK),
        "-m", str(MODEL_F16),
        "-f", str(text_file),
        "--show-count"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    m = re.search(r"Total number of tokens:\s*(\d+)", res.stdout + "\n" + res.stderr)
    if m:
        return int(m.group(1))
    raise RuntimeError(f"Could not parse token count for {text_file}")

def main():
    print("=" * 80)
    print("TASK 5: TEXT STATISTICS AND BPC / BPB NORMALISATION")
    print("=" * 80)

    # 1. Text statistics
    text_stats = {}
    for lang, path in [("bangla", PASSAGE_BN), ("english", PASSAGE_EN)]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))
        n_tokens = get_token_count(path)
        text_stats[lang] = {
            "chars": n_chars,
            "bytes": n_bytes,
            "tokens": n_tokens,
        }
        print(f"\n[{lang.upper()}] Text Used:")
        print(f"  Characters (code points) : {n_chars}")
        print(f"  UTF-8 Bytes              : {n_bytes}")
        print(f"  Gemma Tokens             : {n_tokens}")
        print(f"  Bytes / Character        : {n_bytes / n_chars:.4f}")
        print(f"  Characters / Token       : {n_chars / n_tokens:.4f}")
        print(f"  Bytes / Token            : {n_bytes / n_tokens:.4f}")

    # 2. Existing PPL
    existing_ppl = {}
    with open(EXISTING_PPL_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ppl[(row["tier"], row["language"])] = float(row["ppl"])

    print("\n" + "=" * 90)
    print("ALL 7 TIERS: BPC AND BPB NORMALISATION ACROSS TEXT")
    print("=" * 90)
    print(f"{'Tier':<8} | {'BPW':<5} | {'Lang':<7} | {'PPL':<9} | {'BPC':<8} | {'BPB':<8} | {'BPC Inc vs f16':<16} | {'PPL Inc vs f16':<16}")
    print("-" * 90)

    f16_bpc = {}
    f16_bpb = {}
    rows = []

    for tier, bpw in ALL_TIERS:
        for lang in ["bangla", "english"]:
            ppl = existing_ppl[(tier, lang)]
            bpt = math.log2(ppl)
            total_bits = text_stats[lang]["tokens"] * bpt
            bpc = total_bits / text_stats[lang]["chars"]
            bpb = total_bits / text_stats[lang]["bytes"]

            if tier == "f16":
                f16_bpc[lang] = bpc
                f16_bpb[lang] = bpb
                inc_bpc = 0.0
                inc_bpb = 0.0
            else:
                inc_bpc = ((bpc - f16_bpc[lang]) / f16_bpc[lang]) * 100.0
                inc_bpb = ((bpb - f16_bpb[lang]) / f16_bpb[lang]) * 100.0

            rel_ppl = ((ppl - existing_ppl[("f16", lang)]) / existing_ppl[("f16", lang)]) * 100.0

            rows.append({
                "tier": tier,
                "bpw": bpw,
                "language": lang,
                "ppl": round(ppl, 4),
                "bpt": round(bpt, 4),
                "bpc": round(bpc, 4),
                "bpb": round(bpb, 4),
                "inc_bpc": round(inc_bpc, 2),
                "inc_bpb": round(inc_bpb, 2),
                "rel_ppl": round(rel_ppl, 2),
            })
            bpc_str = f"{inc_bpc:>+6.2f}%" if tier != "f16" else "0.00% (ref)"
            ppl_str = f"{rel_ppl:>+6.2f}%" if tier != "f16" else "0.00% (ref)"
            print(f"{tier:<8} | {bpw:<5.1f} | {lang:<7} | {ppl:<9.4f} | {bpc:<8.4f} | {bpb:<8.4f} | {bpc_str:<16} | {ppl_str:<16}")

    # 3. Split passage files into 3 equal parts
    print("\n" + "=" * 90)
    print("SPLITTING PASSAGE FILES INTO 3 EQUAL PARTS")
    print("=" * 90)
    for lang_code, lang_file in [("bn", PASSAGE_BN), ("en", PASSAGE_EN)]:
        with open(lang_file, encoding="utf-8") as f:
            passages = f.read().split("\n\n")
        n = len(passages)
        p1 = passages[:84]
        p2 = passages[84:168]
        p3 = passages[168:]
        for idx, plist in enumerate([p1, p2, p3], 1):
            out_p = AUDIT_DIR / f"belebele_passages_{lang_code}_part{idx}.txt"
            content = "\n\n".join(plist)
            with open(out_p, "w", encoding="utf-8") as out_f:
                out_f.write(content)
            tok_count = get_token_count(out_p)
            print(f"  {out_p.name}: {len(plist)} passages, {len(content)} chars, {len(content.encode('utf-8'))} bytes, {tok_count} Gemma tokens")

if __name__ == "__main__":
    main()
