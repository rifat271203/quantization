import csv
from collections import Counter
from pathlib import Path

csv_path = Path("data_v2/anchors.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in anchors.csv: {len(rows)}")
c = Counter(r["doc_id"] for r in rows)
for doc, cnt in c.items():
    print(f"  {doc}: {cnt}")
