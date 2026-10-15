import csv
import json
import re
import sys
from pathlib import Path
import pypdf

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path("d:/Tokenizer/quant-hallucination-study")

# Load pages_bn.jsonl
pages_bn = {}
with open(ROOT / "data_v2/pages_bn.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            doc_id = row['doc_id']
            if doc_id not in pages_bn:
                pages_bn[doc_id] = {}
            pages_bn[doc_id][int(row['page'])] = row['text']

# Load EN pdf texts
en_pdfs = {
    'agri_seed_rules_2020': ROOT / 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf',
    'disaster_emcrp_esmf': ROOT / 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf',
    'disaster_emcrp_rpf': ROOT / 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf',
}

pages_en = {}
for doc_id, pdf_path in en_pdfs.items():
    reader = pypdf.PdfReader(str(pdf_path))
    pages_en[doc_id] = {}
    for idx, page in enumerate(reader.pages):
        pages_en[doc_id][idx + 1] = page.extract_text() or ""

# Load anchors.csv
with open(ROOT / "data_v2/anchors.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    anchors_rows = list(reader)

print(f"Total rows in anchors.csv: {len(anchors_rows)}")
# Check for first 5 matched anchors of agri_seed_rules_2020
for r in anchors_rows[:5]:
    doc_id = r['doc_id']
    anc = r['anchor']
    bn_p = int(float(r['bn_page'])) if r['bn_page'] else None
    en_p = int(float(r['en_page'])) if r['en_page'] else None
    print(f"\nDoc: {doc_id}, Anchor: {anc}, BN page: {bn_p}, EN page: {en_p}")
    if bn_p and bn_p in pages_bn.get(doc_id, {}):
        txt = pages_bn[doc_id][bn_p]
        print("  BN page text snippet:", ascii(txt[:150]))
    if en_p and en_p in pages_en.get(doc_id, {}):
        txt = pages_en[doc_id][en_p]
        print("  EN page text snippet:", ascii(txt[:150]))
