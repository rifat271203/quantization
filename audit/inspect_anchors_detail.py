import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path("d:/Tokenizer/quant-hallucination-study")

with open(ROOT / "data_v2/anchors.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

by_doc = defaultdict(list)
for r in rows:
    by_doc[r['doc_id']].append(r)

for doc_id, doc_rows in by_doc.items():
    print(f"=== {doc_id} (total {len(doc_rows)}) ===")
    for r in doc_rows[:15]:
        bn_p = r['bn_page']
        en_p = r['en_page']
        bn_s = ascii(r['bn_first_80_chars'][:35])
        en_s = ascii(r['en_first_80_chars'][:35])
        print(f"  Anchor {r['anchor']:<8} | BN p{bn_p:<4} | EN p{en_p:<4} | BN: {bn_s} | EN: {en_s}")
