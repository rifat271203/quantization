"""
audit/check_passages_in_source.py:
Performs Task 5: checks unique passage_bn and passage_en from eval_candidate_pool.jsonl
against extracted source texts. Reports verbatim, fuzzy >= 0.9, and not found.
Writes audit/passages_not_in_source.csv.
"""

import sys
import os
import json
import csv
import difflib
from pathlib import Path
from collections import defaultdict
import pypdfium2 as pdfium

ROOT = Path(r"d:\Tokenizer\quant-hallucination-study")
DATA_DIR = ROOT / "data"
AUDIT_DIR = ROOT / "audit"
RAW_DIR = DATA_DIR / "raw_sources"

# 1. Load candidate pool passages
pool_rows = []
with open(ROOT / "eval_candidate_pool.jsonl", "r", encoding="utf-8") as f:
    for l in f:
        if l.strip():
            pool_rows.append(json.loads(l))

# Deduplicate passages: (doc_id, lang, passage_str) -> qid
unique_passages = {}
for r in pool_rows:
    doc = r["doc_id"]
    qid = r["question_id"]
    p_bn = r["passage_bn"]
    p_en = r["passage_en"]
    if (doc, "bn", p_bn) not in unique_passages:
        unique_passages[(doc, "bn", p_bn)] = qid
    if (doc, "en", p_en) not in unique_passages:
        unique_passages[(doc, "en", p_en)] = qid

print(f"Total unique passage entries across all docs & languages: {len(unique_passages)}")

# 2. Load source PDF texts
sources = {}

# agri_seed_rules_2020: BN converted text / PDF, EN PDF
with open(RAW_DIR / "converted_text" / "agri_seed_rules_2020_converted.txt", "r", encoding="utf-8") as f:
    seed_bn_conv = f.read()
doc_seed_bn = pdfium.PdfDocument(str(RAW_DIR / "agriculture" / "bn" / "seed_rules_2020_bn.pdf"))
seed_bn_pdf = "\n".join([doc_seed_bn[i].get_textpage().get_text_range() for i in range(len(doc_seed_bn))])
doc_seed_en = pdfium.PdfDocument(str(RAW_DIR / "agriculture" / "en" / "seed_rules_2020_en.pdf"))
seed_en_pdf = "\n".join([doc_seed_en[i].get_textpage().get_text_range() for i in range(len(doc_seed_en))])
sources["agri_seed_rules_2020"] = {
    "bn": seed_bn_conv + "\n" + seed_bn_pdf,
    "en": seed_en_pdf,
}

# disaster_emcrp_esmf: BN PDF, EN PDF
doc_esmf_bn = pdfium.PdfDocument(str(RAW_DIR / "disaster" / "bn" / "esmf_emcrp_bn.pdf"))
doc_esmf_en = pdfium.PdfDocument(str(RAW_DIR / "disaster" / "en" / "esmf_emcrp_en.pdf"))
sources["disaster_emcrp_esmf"] = {
    "bn": "\n".join([doc_esmf_bn[i].get_textpage().get_text_range() for i in range(len(doc_esmf_bn))]),
    "en": "\n".join([doc_esmf_en[i].get_textpage().get_text_range() for i in range(len(doc_esmf_en))]),
}

# disaster_emcrp_rpf: BN PDF, EN PDF
doc_rpf_bn = pdfium.PdfDocument(str(RAW_DIR / "disaster" / "bn" / "rpf_emcrp_bn.pdf"))
doc_rpf_en = pdfium.PdfDocument(str(RAW_DIR / "disaster" / "en" / "rpf_emcrp_en.pdf"))
sources["disaster_emcrp_rpf"] = {
    "bn": "\n".join([doc_rpf_bn[i].get_textpage().get_text_range() for i in range(len(doc_rpf_bn))]),
    "en": "\n".join([doc_rpf_en[i].get_textpage().get_text_range() for i in range(len(doc_rpf_en))]),
}

# edu_nep_2010: BN converted/OCR, EN PDF
doc_nep_en = pdfium.PdfDocument(str(RAW_DIR / "education" / "en" / "national_education_policy_2010_en.pdf"))
nep_en_pdf = "\n".join([doc_nep_en[i].get_textpage().get_text_range() for i in range(len(doc_nep_en))])
nep_bn_text = ""
nep_ocr_dir = RAW_DIR / "ocr_samples" / "national_education_policy_2010_bn"
if nep_ocr_dir.exists():
    for fn in os.listdir(nep_ocr_dir):
        if fn.endswith(".txt"):
            with open(nep_ocr_dir / fn, "r", encoding="utf-8") as f:
                nep_bn_text += f.read() + "\n"
sources["edu_nep_2010"] = {
    "bn": nep_bn_text,
    "en": nep_en_pdf,
}

def clean_ws(s):
    return " ".join(s.split())

def match_passage(sub, full_text):
    if sub in full_text:
        return "verbatim", 1.0
    sub_c = clean_ws(sub)
    full_c = clean_ws(full_text)
    if sub_c in full_c:
        return "verbatim", 1.0
        
    # Test fuzzy ratio
    words = [w for w in sub_c.split() if len(w) >= 4]
    if not words:
        r = difflib.SequenceMatcher(None, sub_c, full_c[: len(sub_c) * 2]).ratio()
        return ("fuzzy", r) if r >= 0.9 else ("not_found", r)
        
    anchors = [
        " ".join(words[:2]),
        " ".join(words[len(words) // 2 : len(words) // 2 + 2]),
        " ".join(words[-2:]),
    ]
    best_r = 0.0
    L = len(sub_c)
    for anch in anchors:
        idx = 0
        while True:
            idx = full_c.find(anch, idx)
            if idx == -1:
                break
            win_start = max(0, idx - int(L * 0.1))
            win_end = min(len(full_c), idx + int(L * 1.1))
            win = full_c[win_start:win_end]
            r = difflib.SequenceMatcher(None, sub_c, win).ratio()
            if r > best_r:
                best_r = r
            if best_r >= 0.9:
                return "fuzzy", best_r
            idx += len(anch)
            
    if best_r >= 0.9:
        return "fuzzy", best_r
    return "not_found", best_r

# Execute check
results = defaultdict(lambda: {"verbatim": 0, "fuzzy": 0, "not_found": 0, "total": 0})
not_found_list = []

print("Scanning passages against source PDF texts...")
for (doc, lang, p_str), qid in unique_passages.items():
    src_text = sources[doc][lang]
    status, ratio = match_passage(p_str, src_text)
    results[(doc, lang)]["total"] += 1
    results[(doc, lang)][status] += 1
    if status == "not_found":
        not_found_list.append({
            "question_id": qid,
            "doc_id": doc,
            "lang": lang,
            "fuzzy_ratio": round(ratio, 4),
            "first_200_chars": p_str[:200],
        })

print("\n" + "=" * 80)
print("PASSAGE SOURCE CHECK RESULTS:")
print("=" * 80)
for (doc, lang) in sorted(results.keys()):
    st = results[(doc, lang)]
    tot = st["total"]
    verb = st["verbatim"]
    fuzz = st["fuzzy"]
    nf = st["not_found"]
    print(f"  {doc:24s} [{lang.upper()}]: Total={tot:4d} | Verbatim={verb:4d} | Fuzzy>=0.9={fuzz:4d} | Not Found={nf:4d}")

# Write audit/passages_not_in_source.csv
not_found_csv = AUDIT_DIR / "passages_not_in_source.csv"
with open(not_found_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question_id", "doc_id", "lang", "fuzzy_ratio", "first_200_chars"])
    writer.writeheader()
    for r in not_found_list:
        writer.writerow(r)

print(f"\nWrote {len(not_found_list)} not-found passages to {not_found_csv}")
