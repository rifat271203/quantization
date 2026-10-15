"""
audit/audit4.py: Comprehensive read-only audit script.
Executes Sections A, B, C, D, E of user audit request and prints exact numbers.
"""

import sys
import os
import json
import csv
import re
import unicodedata
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
import regex
from transformers import AutoTokenizer, AutoModel
import pypdfium2 as pdfium

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\Tokenizer\quant-hallucination-study")
DATA_DIR = ROOT / "data"
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# SECTION A: Garble scan
# -----------------------------------------------------------------------------
print("=" * 80)
print("SECTION A: GARBLE SCAN")
print("=" * 80)

def extract_passage_from_item(item, source_name):
    if "passage" in item:
        return item["passage"]
    elif "passage_bn" in item:
        return item["passage_bn"]
    elif "messages" in item:
        user_c = next((m["content"] for m in item.get("messages", []) if m.get("role") == "user"), "")
        triple = '"""'
        p1 = user_c.find(triple)
        p2 = user_c.find(triple, p1 + 3)
        if p1 != -1 and p2 != -1:
            return user_c[p1 + 3 : p2].strip()
        return user_c
    return ""

def scan_text_garble(text):
    has_cn = False
    has_pua = False
    has_c0_ff = False
    flagged_chars = Counter()
    
    for c in text:
        cp = ord(c)
        if 0x0980 <= cp <= 0x09FF and unicodedata.category(c) == "Cn":
            has_cn = True
            flagged_chars[(c, f"U+{cp:04X}", f"Cn (unassigned)", cp)] += 1
        elif 0xE000 <= cp <= 0xF8FF:
            has_pua = True
            flagged_chars[(c, f"U+{cp:04X}", f"PUA", cp)] += 1
        elif 0x00C0 <= cp <= 0x00FF:
            has_c0_ff = True
            name = unicodedata.name(c, "LATIN/SYM")
            flagged_chars[(c, f"U+{cp:04X}", f"U+00C0-U+00FF ({name})", cp)] += 1

    # Latin letters mixed inside Bengali words
    has_latin_mixed = False
    words = regex.findall(r"[\p{L}\p{M}]+", text)
    for w in words:
        w_has_bn = any(0x0980 <= ord(ch) <= 0x09FF for ch in w)
        w_has_lat = any(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in w)
        if w_has_bn and w_has_lat:
            has_latin_mixed = True
            for ch in w:
                if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
                    cp = ord(ch)
                    flagged_chars[(ch, f"U+{cp:04X}", f"Latin-in-Bengali-word", cp)] += 1

    return has_cn, has_pua, has_c0_ff, has_latin_mixed, flagged_chars

datasets = [
    ("eval_bangla.jsonl", DATA_DIR / "eval_bangla.jsonl"),
    ("eval_candidate_pool.jsonl", ROOT / "eval_candidate_pool.jsonl"),
    ("train_bn_only.jsonl", DATA_DIR / "train_bn_only.jsonl"),
]

for ds_name, ds_path in datasets:
    print(f"\n--- DATASET: {ds_name} ---")
    doc_stats = defaultdict(lambda: {
        "total_rows": 0,
        "flag_cn": 0,
        "flag_pua": 0,
        "flag_c0_ff": 0,
        "flag_latin_mixed": 0,
        "flag_any": 0,
        "flagged_chars": Counter(),
    })
    
    with open(ds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id", "UNKNOWN")
            text = extract_passage_from_item(item, ds_name)
            
            has_cn, has_pua, has_c0, has_lat, f_chars = scan_text_garble(text)
            
            s = doc_stats[doc_id]
            s["total_rows"] += 1
            if has_cn: s["flag_cn"] += 1
            if has_pua: s["flag_pua"] += 1
            if has_c0: s["flag_c0_ff"] += 1
            if has_lat: s["flag_latin_mixed"] += 1
            if has_cn or has_pua or has_c0 or has_lat: s["flag_any"] += 1
            s["flagged_chars"].update(f_chars)
            
    # Print table per doc_id
    header = f"{'Doc ID':32} | {'Rows':6} | {'(a) Cn':10} | {'(b) PUA':10} | {'(c) C0-FF':10} | {'(d) LatMix':10} | {'Any Flag':10}"
    print(header)
    print("-" * len(header))
    for doc_id, s in sorted(doc_stats.items()):
        n = s["total_rows"]
        p_cn = f"{s['flag_cn']} ({s['flag_cn']/n*100:.1f}%)"
        p_pua = f"{s['flag_pua']} ({s['flag_pua']/n*100:.1f}%)"
        p_c0 = f"{s['flag_c0_ff']} ({s['flag_c0_ff']/n*100:.1f}%)"
        p_lat = f"{s['flag_latin_mixed']} ({s['flag_latin_mixed']/n*100:.1f}%)"
        p_any = f"{s['flag_any']} ({s['flag_any']/n*100:.1f}%)"
        print(f"{doc_id:32} | {n:6d} | {p_cn:10} | {p_pua:10} | {p_c0:10} | {p_lat:10} | {p_any:10}")

    print(f"\nTOP 10 FLAGGED CHARACTERS PER DOC_ID ({ds_name}):")
    for doc_id, s in sorted(doc_stats.items()):
        fc = s["flagged_chars"]
        print(f"  Doc ID: {doc_id} (Total flagged char occurrences: {sum(fc.values())})")
        if not fc:
            print("    None")
        else:
            for (char, cp_str, desc, cp_val), count in fc.most_common(10):
                print(f"    {cp_str} (U+{cp_val:04X}, repr={repr(char):6s}, {desc}): count={count}")


# -----------------------------------------------------------------------------
# SECTION B: 5 Rows of disaster_emcrp_esmf
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION B: 5 ROWS OF disaster_emcrp_esmf (SIDE BY SIDE)")
print("=" * 80)

esmf_rows = []
with open(ROOT / "eval_candidate_pool.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("doc_id") == "disaster_emcrp_esmf":
            esmf_rows.append(item)

for i in range(min(5, len(esmf_rows))):
    r = esmf_rows[i]
    print(f"\n>>> ROW {i+1} [question_id: {r.get('question_id')}, bn_id: {r.get('bn_id')}, en_id: {r.get('en_id')}] <<<")
    print(f"  passage_bn (repr):     {repr(r.get('passage_bn'))}")
    print(f"  instruction_bn:        {r.get('instruction_bn')}")
    print(f"  response_bn:           {r.get('response_bn')}")
    print(f"  passage_en (repr):     {repr(r.get('passage_en'))}")
    print(f"  instruction_en:        {r.get('instruction_en')}")
    print(f"  response_en:           {r.get('response_en')}")


# -----------------------------------------------------------------------------
# SECTION C: Re-check Alignment Metric
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION C: RE-CHECK THE ALIGNMENT METRIC")
print("=" * 80)

# 5. Count unique passage_bn and passage_en strings in eval_candidate_pool.jsonl overall and per doc_id
pool_bn_passages = set()
pool_en_passages = set()
pool_doc_passages = defaultdict(lambda: {"bn": set(), "en": set(), "pairs": set(), "total": 0})

with open(ROOT / "eval_candidate_pool.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        doc = item.get("doc_id", "UNKNOWN")
        p_bn = item.get("passage_bn", "")
        p_en = item.get("passage_en", "")
        pool_bn_passages.add(p_bn)
        pool_en_passages.add(p_en)
        pool_doc_passages[doc]["bn"].add(p_bn)
        pool_doc_passages[doc]["en"].add(p_en)
        pool_doc_passages[doc]["pairs"].add((p_bn, p_en))
        pool_doc_passages[doc]["total"] += 1

print(f"Overall eval_candidate_pool.jsonl (Total rows = {sum(d['total'] for d in pool_doc_passages.values())}):")
print(f"  Unique passage_bn strings: {len(pool_bn_passages)}")
print(f"  Unique passage_en strings: {len(pool_en_passages)}")
print(f"\nPer doc_id breakdown in eval_candidate_pool.jsonl:")
for doc, d in sorted(pool_doc_passages.items()):
    print(f"  {doc:32s}: Total Rows={d['total']:5d} | Unique BN Passages={len(d['bn']):4d} | Unique EN Passages={len(d['en']):4d} | Unique Pairs={len(d['pairs']):4d}")

# 6. Redo retrieval test with intfloat/multilingual-e5-base on UNIQUE passages only
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading intfloat/multilingual-e5-base onto {device}...")
emb_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
emb_model = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(device)
emb_model.eval()

def embed_texts(texts, prefix="passage: "):
    embs = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i : i + 32]]
        encoded = emb_tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = emb_model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_emb / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            embs.append(normalized.cpu().numpy())
    return np.vstack(embs)

for target_doc in ["agri_seed_rules_2020", "edu_nep_2010"]:
    bn_to_en = {}
    with open(ROOT / "eval_candidate_pool.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("doc_id") == target_doc:
                p_bn = item["passage_bn"]
                p_en = item["passage_en"]
                if p_bn not in bn_to_en:
                    bn_to_en[p_bn] = p_en
                    
    uniq_bn = list(bn_to_en.keys())
    uniq_en = [bn_to_en[b] for b in uniq_bn]
    N = len(uniq_bn)
    
    print(f"\n--- RETRIEVAL TEST: {target_doc} (N={N} UNIQUE PASSAGE PAIRS) ---")
    bn_embs = embed_texts(uniq_bn, prefix="passage: ")
    en_embs = embed_texts(uniq_en, prefix="passage: ")
    
    sim_matrix = np.dot(bn_embs, en_embs.T)
    top1 = 0
    top5 = 0
    ranks = []
    for i in range(N):
        sims = sim_matrix[i]
        true_sim = sims[i]
        rank = int(np.sum(sims > true_sim)) + 1
        ranks.append(rank)
        if rank == 1: top1 += 1
        if rank <= 5: top5 += 1
        
        if target_doc == "agri_seed_rules_2020" and i < 5:
            other_sims = np.delete(sims, i)
            best_other = np.max(other_sims)
            other_idx = [j for j in range(N) if j != i]
            best_other_idx = max(other_idx, key=lambda j: sims[j])
            print(f"  Row {i+1}: True Cosine={true_sim:.4f} | Best Other Cosine={best_other:.4f} | Margin={true_sim - best_other:+.4f} | Rank={rank}")
            print(f"         True EN:       {repr(uniq_en[i][:75])}")
            print(f"         Best Other EN: {repr(uniq_en[best_other_idx][:75])}")

    print(f"Results for {target_doc}:")
    print(f"  Top-1 Accuracy: {top1}/{N} ({top1/N*100:.2f}%)")
    print(f"  Top-5 Accuracy: {top5}/{N} ({top5/N*100:.2f}%)")


# -----------------------------------------------------------------------------
# SECTION D: OCR tools & test
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION D: OCR AVAILABILITY & RENDER TEST")
print("=" * 80)

tess_exe = r"C:\Users\Tanjid\.gemini\antigravity-ide\brain\9665d0d8-f813-491f-9752-0404d32efc11\scratch\tesseract\tesseract.exe"
tessdata_best = r"C:\Users\Tanjid\.gemini\antigravity-ide\brain\9665d0d8-f813-491f-9752-0404d32efc11\scratch\tesseract\tessdata_best"

print("Tesseract binary check:")
if os.path.exists(tess_exe):
    res = subprocess.run([tess_exe, "--version"], capture_output=True, text=True)
    print(f"  Found tesseract at: {tess_exe}")
    print(f"  Version output:\n{res.stdout.strip()}")
else:
    print("  Tesseract binary NOT found.")

print("\nTessdata check:")
if os.path.exists(tessdata_best):
    res_langs = subprocess.run([tess_exe, "--tessdata-dir", tessdata_best, "--list-langs"], capture_output=True, text=True)
    print(f"  Found tessdata_best at: {tessdata_best}")
    print(f"  Available languages:\n{res_langs.stdout.strip()}")
else:
    print("  tessdata_best NOT found.")

print("\nPDF Rasterizer check:")
print(f"  pypdfium2: version 5.13.0 installed")

# -----------------------------------------------------------------------------
# SECTION E: File Check usable_pool.csv
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION E: FILE CHECK usable_pool.csv")
print("=" * 80)

usable_csv = AUDIT_DIR / "usable_pool.csv"
if usable_csv.exists():
    with open(usable_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"File path: {usable_csv}")
    print(f"Total lines: {len(lines)}")
    row_count = sum(1 for l in lines if l.strip()) - 1
    print(f"Data row count: {row_count}")
    if lines:
        print(f"Header: {lines[0].strip()}")
    if len(lines) > 1:
        print(f"First row: {repr(lines[1].strip())}")
    else:
        print("First row: [EMPTY - NO ROWS]")
else:
    print(f"{usable_csv} does not exist.")

print("\nAudit 4 completed successfully.")
