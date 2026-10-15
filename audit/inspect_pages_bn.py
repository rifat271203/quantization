import json
from pathlib import Path
from collections import Counter

ROOT = Path("d:/Tokenizer/quant-hallucination-study")

pages_path = ROOT / "data_v2/pages_bn.jsonl"
print(f"Checking {pages_path}:")
docs = Counter()
pages_by_doc = Counter()
first_rows = {}
with open(pages_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            doc_id = row.get("doc_id")
            docs[doc_id] += 1
            if doc_id not in first_rows:
                first_rows[doc_id] = row

print(f"Total pages in pages_bn.jsonl: {sum(docs.values())}")
for d, cnt in docs.items():
    print(f"  {d}: {cnt} pages")

for d, row in first_rows.items():
    print(f"\nSample page for {d}:")
    print(f"  Keys: {list(row.keys())}")
    print(f"  page: {row.get('page')}")
    print(f"  text snippet: {ascii(row.get('text', '')[:120])}")
