import os
import re
import json
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
import pypdf

ROOT = Path("d:/Tokenizer/quant-hallucination-study")

print("="*60)
print("TASK 1: LITERAL STRINGS AUDIT")
print("="*60)

# 1. Parse build_study_eval_sets.py for literal question IDs
script_path = ROOT / "data/raw_sources/build_study_eval_sets.py"
with open(script_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

literal_definitions = []
# Patterns: add_clean_pair('id', ...), act_items ('id', ...), low-conf items 'id': '...', nctb_items ('id', ...)
for i, l in enumerate(lines, 1):
    strip_l = l.strip()
    # Check add_clean_pair
    m = re.search(r"add_clean_pair\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'", strip_l)
    if m:
        literal_definitions.append({
            "line": i,
            "qid": m.group(1),
            "domain": m.group(2),
            "doc_id": m.group(3),
            "type": "add_clean_pair"
        })
    # Check act_items tuple: ('disaster_act_01', 21, ...)
    m = re.search(r"\('([a-zA-Z0-9_]+)',\s*(\d+),", strip_l)
    if m and ("act_" in m.group(1) or "sci_" in m.group(1)):
        doc = "disaster_act_2012" if "act_" in m.group(1) else "edu_nctb_science_class9"
        literal_definitions.append({
            "line": i,
            "qid": m.group(1),
            "doc_id": doc,
            "type": "tuple_item"
        })
    # Check 'id': '...'
    m = re.search(r"'id':\s*'([^']+)'", strip_l)
    if m:
        literal_definitions.append({
            "line": i,
            "qid": m.group(1),
            "doc_id": "disaster_act_2012" if "disaster_act" in m.group(1) else "unknown",
            "type": "dict_id"
        })

print(f"Total literal definitions identified in build_study_eval_sets.py: {len(literal_definitions)}")
for item in literal_definitions:
    print(f"  Line {item['line']:3d}: QID='{item['qid']}' (doc_id={item.get('doc_id')}, type={item['type']})")

# Also check other scripts: generate_dense_qa.py
dense_script = ROOT / "data/raw_sources/generate_dense_qa.py"
with open(dense_script, "r", encoding="utf-8") as f:
    dense_lines = f.readlines()

dense_literals = []
for i, l in enumerate(dense_lines, 1):
    # Check ADDITIONAL_SENTENCE_PAIRS or manual QA pairs
    if "ADDITIONAL_SENTENCE_PAIRS" in l or "pair_doc_" in l:
        dense_literals.append((i, l.strip()))
print(f"\nLiteral references in generate_dense_qa.py: {len(dense_literals)}")
for i, l in dense_literals[:10]:
    print(f"  Line {i:3d}: {l[:100]}")

# 2. Check occurrences in eval_candidate_pool.jsonl and train files
# Let's inspect eval_candidate_pool.jsonl (both in root and in data/)
pool_paths = [ROOT / "eval_candidate_pool.jsonl", ROOT / "data/eval_candidate_pool.jsonl"]
# train files:
train_bn_path = ROOT / "data/train_bn_only.jsonl"
train_en_path = ROOT / "data/train_en_only.jsonl"

all_literal_qids = set()
for item in literal_definitions:
    qid = item["qid"]
    all_literal_qids.add(qid)
    # in jsonl, it may be f"{qid}__bn", f"{qid}__en", f"{qid}_bn", etc.
    all_literal_qids.add(f"{qid}__bn")
    all_literal_qids.add(f"{qid}__en")
    all_literal_qids.add(f"{qid}_bn")
    all_literal_qids.add(f"{qid}_en")

def audit_file_for_literals(fpath, label):
    if not fpath.exists():
        print(f"\n{label}: File does not exist ({fpath})")
        return
    total_rows = 0
    matched_rows = 0
    matches_per_doc = Counter()
    matched_id_list = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            row_id = row.get("id", "")
            doc_id = row.get("doc_id", "unknown")
            
            # Check if row_id matches literal ID or base ID
            base_id = row_id.split("__")[0] if "__" in row_id else row_id
            if row_id in all_literal_qids or base_id in all_literal_qids or any(base_id.startswith(item["qid"]) for item in literal_definitions):
                matched_rows += 1
                matches_per_doc[doc_id] += 1
                matched_id_list.append((row_id, doc_id))
                
    print(f"\nAudit of {label} ({fpath.name}):")
    print(f"  Total rows: {total_rows}")
    print(f"  Matched literal rows: {matched_rows}")
    print(f"  Matched per doc_id:")
    for doc, cnt in sorted(matches_per_doc.items()):
        print(f"    {doc}: {cnt}")
    if matched_rows > 0:
        print(f"  Sample matched IDs: {[x[0] for x in matched_id_list[:5]]}")

audit_file_for_literals(ROOT / "eval_candidate_pool.jsonl", "Root eval_candidate_pool.jsonl")
audit_file_for_literals(ROOT / "data/eval_candidate_pool.jsonl", "Data eval_candidate_pool.jsonl")
audit_file_for_literals(train_bn_path, "Train BN Only")
audit_file_for_literals(train_en_path, "Train EN Only")

# Also check data/eval_bangla.jsonl and data/eval_english.jsonl
audit_file_for_literals(ROOT / "data/eval_bangla.jsonl", "Eval Bangla (data/eval_bangla.jsonl)")
audit_file_for_literals(ROOT / "data/eval_english.jsonl", "Eval English (data/eval_english.jsonl)")

# (c) Authorship / origin of literal strings
print("\n" + "="*60)
print("TASK 1(c): AUTHORSHIP / ORIGIN OF LITERAL STRINGS")
print("="*60)
# Look at docstrings and comments in build_study_eval_sets.py
docstring_lines = lines[:40]
print("Top lines of build_study_eval_sets.py:")
print("".join(docstring_lines[:25]))

# Git log author
git_author = subprocess.run("git log -n 1 --format=\"Commit: %H%nAuthor: %an <%ae>%nDate: %ad%nMessage: %s\"", 
                            cwd=str(ROOT), capture_output=True, text=True, shell=True)
print("\nGit commit log:")
print(git_author.stdout.strip())

