"""
Read-only Audit Script for quant-hallucination-study data and splits.json.
Writes all outputs cleanly to stdout and audit/audit_report.json.
DO NOT modify, delete, or overwrite any existing file outside audit/.
"""

import sys
import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\Tokenizer\quant-hallucination-study")
DATA_DIR = ROOT / "data"
SPLITS_PATH = DATA_DIR / "splits.json"

FILES_TO_INSPECT = [
    "train_bilingual.jsonl",
    "train_bn_only.jsonl",
    "train_en_only.jsonl",
    "eval_bangla.jsonl",
    "eval_english.jsonl",
    "eval_candidate_pool.jsonl",
]

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

print("=" * 80)
print("SECTION 0: FIELD NAMES AND 2 SAMPLE ROWS PER FILE")
print("=" * 80)

samples_data = {}

for fname in FILES_TO_INSPECT:
    fpath = DATA_DIR / fname
    print(f"\n--- File: {fname} ---")
    if not fpath.exists():
        print("STATUS: MISSING FILE")
        samples_data[fname] = {"status": "MISSING"}
        continue
    
    rows = []
    with fpath.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < 2:
                rows.append(json.loads(line.strip()))
            else:
                break
    
    if rows:
        field_names = list(rows[0].keys())
        print(f"Fields ({len(field_names)}): {field_names}")
        for i, r in enumerate(rows, 1):
            print(f"Sample Row {i}: {json.dumps(r, ensure_ascii=False)}")
        samples_data[fname] = {"fields": field_names, "samples": rows}
    else:
        print("STATUS: EMPTY FILE")
        samples_data[fname] = {"status": "EMPTY"}

print("\n" + "=" * 80)
print("SECTION 1: TOKEN LENGTHS & TRUNCATION ANALYSIS")
print("=" * 80)

print("Loading google/gemma-3-1b-it tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Load training files
train_bn_rows = load_jsonl(DATA_DIR / "train_bn_only.jsonl")
train_en_rows = load_jsonl(DATA_DIR / "train_en_only.jsonl")
train_bilingual_rows = load_jsonl(DATA_DIR / "train_bilingual.jsonl")

print(f"Loaded train_bn_only: {len(train_bn_rows)} rows")
print(f"Loaded train_en_only: {len(train_en_rows)} rows")
print(f"Loaded train_bilingual: {len(train_bilingual_rows)} rows")

def get_row_text(r):
    # If messages structure present
    if "messages" in r:
        msgs = r["messages"]
        user_c = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_c = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return user_c, asst_c, user_c + " " + asst_c
    else:
        # passage + instruction + response
        ctx = r.get("passage", "")
        q = r.get("instruction", "")
        a = r.get("response", "")
        return ctx + " " + q, a, f"{ctx} {q} {a}"

# Compute token lengths formatted (as trained via chat_template) AND raw (context + question + answer text)
bn_chat_lens = []
bn_raw_lens = []
for r in train_bn_rows:
    if "messages" in r:
        formatted = tokenizer.apply_chat_template(r["messages"], tokenize=False)
        bn_chat_lens.append(len(tokenizer(formatted)["input_ids"]))
    u, a, full = get_row_text(r)
    bn_raw_lens.append(len(tokenizer(full)["input_ids"]))

en_chat_lens = []
en_raw_lens = []
for r in train_en_rows:
    if "messages" in r:
        formatted = tokenizer.apply_chat_template(r["messages"], tokenize=False)
        en_chat_lens.append(len(tokenizer(formatted)["input_ids"]))
    u, a, full = get_row_text(r)
    en_raw_lens.append(len(tokenizer(full)["input_ids"]))

def stats(arr, name):
    arr = np.array(arr)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
        "trunc_512": int(np.sum(arr > 512)),
        "trunc_1024": int(np.sum(arr > 1024)),
        "trunc_2048": int(np.sum(arr > 2048)),
    }

bn_chat_stats = stats(bn_chat_lens, "Bangla (Chat Template)")
bn_raw_stats = stats(bn_raw_lens, "Bangla (Raw Context+Question+Answer)")
en_chat_stats = stats(en_chat_lens, "English (Chat Template)")
en_raw_stats = stats(en_raw_lens, "English (Raw Context+Question+Answer)")

print("\n--- BANGLA TRAINING EXAMPLES (Chat-template formatted) ---")
for k, v in bn_chat_stats.items():
    print(f"  {k}: {v}")

print("\n--- BANGLA TRAINING EXAMPLES (Raw text: Context + Question + Answer) ---")
for k, v in bn_raw_stats.items():
    print(f"  {k}: {v}")

print("\n--- ENGLISH TRAINING EXAMPLES (Chat-template formatted) ---")
for k, v in en_chat_stats.items():
    print(f"  {k}: {v}")

print("\n--- ENGLISH TRAINING EXAMPLES (Raw text: Context + Question + Answer) ---")
for k, v in en_raw_stats.items():
    print(f"  {k}: {v}")

print("\n--- Truncation Comparison (Chat-Template Sequence Length) ---")
print(f"Bangla Total: {bn_chat_stats['count']}")
print(f"  Truncated at 512:  {bn_chat_stats['trunc_512']} ({bn_chat_stats['trunc_512']/bn_chat_stats['count']*100:.2f}%)")
print(f"  Truncated at 1024: {bn_chat_stats['trunc_1024']} ({bn_chat_stats['trunc_1024']/bn_chat_stats['count']*100:.2f}%)")
print(f"  Truncated at 2048: {bn_chat_stats['trunc_2048']} ({bn_chat_stats['trunc_2048']/bn_chat_stats['count']*100:.2f}%)")

print(f"English Total: {en_chat_stats['count']}")
print(f"  Truncated at 512:  {en_chat_stats['trunc_512']} ({en_chat_stats['trunc_512']/en_chat_stats['count']*100:.2f}%)")
print(f"  Truncated at 1024: {en_chat_stats['trunc_1024']} ({en_chat_stats['trunc_1024']/en_chat_stats['count']*100:.2f}%)")
print(f"  Truncated at 2048: {en_chat_stats['trunc_2048']} ({en_chat_stats['trunc_2048']/en_chat_stats['count']*100:.2f}%)")

# Check train_qlora.py and notebooks for max_seq_length
print("\n--- Checking max_seq_length configured in scripts and notebooks ---")
scripts_to_check = [
    ROOT / "train_qlora.py",
    ROOT / "gemma3-qlora-bilingual.ipynb",
    ROOT / "gemma3-qlora-bn-only.ipynb",
    ROOT / "gemma3-qlora-en-only.ipynb",
    ROOT / "kaggle_finetune_gemma3_qlora.ipynb",
    ROOT / "scripts" / "finetune_lora.py",
]
configured_max_lens = {}
for p in scripts_to_check:
    if p.exists():
        content = p.read_text(encoding="utf-8")
        matches = re.findall(r"max_seq_length.*?(\d+)", content, re.IGNORECASE)
        sft_matches = re.findall(r"max_length.*?(\d+)", content, re.IGNORECASE)
        configured_max_lens[p.name] = {"max_seq_length_matches": matches, "max_length_matches": sft_matches}
        print(f"  {p.name}: max_seq_length matches={matches}, max_length matches={sft_matches}")
    else:
        print(f"  {p.name}: NOT FOUND")

print("\n" + "=" * 80)
print("SECTION 2: UNANSWERABLE QUESTIONS / REFUSAL ITEMS")
print("=" * 80)

# Check all fields across train and eval files
all_files = {
    "train_bilingual.jsonl": load_jsonl(DATA_DIR / "train_bilingual.jsonl"),
    "train_bn_only.jsonl": train_bn_rows,
    "train_en_only.jsonl": train_en_rows,
    "eval_bangla.jsonl": load_jsonl(DATA_DIR / "eval_bangla.jsonl"),
    "eval_english.jsonl": load_jsonl(DATA_DIR / "eval_english.jsonl"),
    "eval_candidate_pool.jsonl": load_jsonl(DATA_DIR / "eval_candidate_pool.jsonl"),
}

refusal_keywords_en = [
    "not mentioned", "not stated", "cannot be answered", "not provided", 
    "passage does not", "not in the passage", "not supported", "unanswerable"
]
refusal_keywords_bn = [
    "উল্লেখ নেই", "উল্লেখ করা হয়নি", "দেওয়া নেই", "পাওয়া যায়নি", 
    "অনুচ্ছেদে নেই", "উত্তর দেওয়া সম্ভব নয়", "তথ্য নেই"
]

refusal_findings = {}
for name, rows in all_files.items():
    # Check if there is an explicit field
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    
    explicit_refusal_field = [k for k in all_keys if any(term in k.lower() for term in ["unanswerable", "refusal", "is_supported", "answerable", "supported"])]
    
    # Text-based search in response/answer
    refusal_count = 0
    for r in rows:
        ans_text = ""
        if "messages" in r:
            ans_text = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        elif "response" in r:
            ans_text = r.get("response", "")
        elif "response_bn" in r or "response_en" in r:
            ans_text = str(r.get("response_bn", "")) + " " + str(r.get("response_en", ""))
        
        lower_ans = ans_text.lower()
        if any(kw in lower_ans for kw in refusal_keywords_en) or any(kw in ans_text for kw in refusal_keywords_bn):
            refusal_count += 1
            
    refusal_findings[name] = {
        "total_rows": len(rows),
        "explicit_field_found": explicit_refusal_field,
        "keyword_refusal_count": refusal_count,
        "keyword_refusal_fraction": refusal_count / len(rows) if rows else 0.0
    }
    print(f"File: {name} (total {len(rows)})")
    print(f"  Explicit refusal/unanswerable field: {explicit_refusal_field if explicit_refusal_field else 'NONE'}")
    print(f"  Responses matching refusal keywords: {refusal_count} ({refusal_findings[name]['keyword_refusal_fraction']:.4%})")

print("\n" + "=" * 80)
print("SECTION 3: LEAKAGE CHECK (TRAIN VS EVAL & SPLITS.JSON)")
print("=" * 80)

# Load splits.json
with SPLITS_PATH.open("r", encoding="utf-8") as f:
    splits_config = json.load(f)

print("Splits document assignments in splits.json:")
for k, v in splits_config.items():
    if k != "_metadata":
        if isinstance(v, dict):
            print(f"  {k}: Partitioned (chapter_level): {list(v.get('chapter_allocations', {}).keys())}")
        else:
            print(f"  {k}: {v}")

# Gather doc_ids from train and eval
train_docs = defaultdict(set)
eval_docs = defaultdict(set)

for r in train_bilingual_rows:
    doc_id = r.get("doc_id", "MISSING")
    train_docs[doc_id].add(r.get("id"))

eval_bn_rows = all_files["eval_bangla.jsonl"]
eval_en_rows = all_files["eval_english.jsonl"]

for r in eval_bn_rows:
    doc_id = r.get("doc_id", "MISSING")
    eval_docs[doc_id].add(r.get("id"))

for r in eval_en_rows:
    doc_id = r.get("doc_id", "MISSING")
    eval_docs[doc_id].add(r.get("id"))

print(f"\nUnique doc_ids in train_bilingual: {sorted(list(train_docs.keys()))}")
print(f"Unique doc_ids in eval (bangla + english): {sorted(list(eval_docs.keys()))}")

overlapping_docs = set(train_docs.keys()).intersection(set(eval_docs.keys()))
print(f"\nDocument IDs appearing in BOTH train and eval: {sorted(list(overlapping_docs))}")

for odoc in overlapping_docs:
    print(f"\nInvestigating overlap for doc_id: '{odoc}'")
    print(f"  Train item count: {len(train_docs[odoc])}")
    print(f"  Eval item count:  {len(eval_docs[odoc])}")
    # Inspect splits.json policy for this doc
    policy = splits_config.get(odoc)
    print(f"  splits.json policy for '{odoc}': {policy}")

    # Inspect chapter/page distribution for edu_nep_2010 or others
    train_samples_pages = [r.get("source_page") for r in train_bilingual_rows if r.get("doc_id") == odoc and "source_page" in r]
    # Check IDs prefix
    train_ids = [r.get("id") for r in train_bilingual_rows if r.get("doc_id") == odoc][:5]
    eval_ids = [r.get("id") for r in eval_bn_rows if r.get("doc_id") == odoc][:5]
    print(f"  Sample train IDs: {train_ids}")
    print(f"  Sample eval IDs:  {eval_ids}")

print("\n" + "=" * 80)
print("SECTION 4: PARALLELISM (QUESTION ID SHARING ACROSS LANGUAGES)")
print("=" * 80)

def extract_base_qid(full_id):
    if not full_id:
        return "MISSING"
    # standard format is <base_id>__bn or <base_id>__en
    if full_id.endswith("__bn") or full_id.endswith("__en"):
        return full_id[:-4]
    return full_id

# In Train (using train_bn_only and train_en_only)
train_bn_base_ids = set(extract_base_qid(r.get("id")) for r in train_bn_rows)
train_en_base_ids = set(extract_base_qid(r.get("id")) for r in train_en_rows)

train_shared = train_bn_base_ids.intersection(train_en_base_ids)
train_only_bn = train_bn_base_ids - train_en_base_ids
train_only_en = train_en_base_ids - train_bn_base_ids

print("TRAINING SET PARALLELISM:")
print(f"  Total Bangla item base IDs:  {len(train_bn_base_ids)}")
print(f"  Total English item base IDs: {len(train_en_base_ids)}")
print(f"  Base IDs in BOTH languages:  {len(train_shared)}")
print(f"  Base IDs ONLY in Bangla:     {len(train_only_bn)}")
print(f"  Base IDs ONLY in English:    {len(train_only_en)}")

# In Eval
eval_bn_base_ids = set(extract_base_qid(r.get("id")) for r in eval_bn_rows)
eval_en_base_ids = set(extract_base_qid(r.get("id")) for r in eval_en_rows)

eval_shared = eval_bn_base_ids.intersection(eval_en_base_ids)
eval_only_bn = eval_bn_base_ids - eval_en_base_ids
eval_only_en = eval_en_base_ids - eval_bn_base_ids

print("\nEVAL SET PARALLELISM (eval_bangla.jsonl vs eval_english.jsonl):")
print(f"  Total Bangla item base IDs:  {len(eval_bn_base_ids)}")
print(f"  Total English item base IDs: {len(eval_en_base_ids)}")
print(f"  Base IDs in BOTH languages:  {len(eval_shared)}")
print(f"  Base IDs ONLY in Bangla:     {len(eval_only_bn)}")
print(f"  Base IDs ONLY in English:    {len(eval_only_en)}")

# In Candidate pool
eval_cand_rows = all_files["eval_candidate_pool.jsonl"]
cand_qids = [r.get("question_id") for r in eval_cand_rows if "question_id" in r]
cand_bn_ids = [r.get("bn_id") for r in eval_cand_rows if "bn_id" in r]
cand_en_ids = [r.get("en_id") for r in eval_cand_rows if "en_id" in r]
print(f"\nEVAL CANDIDATE POOL (eval_candidate_pool.jsonl):")
print(f"  Total rows: {len(eval_cand_rows)}")
print(f"  Rows with both bn_id & en_id: {sum(1 for r in eval_cand_rows if r.get('bn_id') and r.get('en_id'))}")

print("\n" + "=" * 80)
print("SECTION 5: PROVENANCE (ORIGINAL / MT / LLM / HUMAN VERIFIED)")
print("=" * 80)

provenance_fields_found = {}
for name, rows in all_files.items():
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    prov_keys = [k for k in all_keys if any(term in k.lower() for term in ["prov", "origin", "human", "native", "annotat", "llm", "mt", "translat", "source_conf", "review"])]
    provenance_fields_found[name] = {
        "all_keys": sorted(list(all_keys)),
        "candidate_provenance_keys": prov_keys
    }
    print(f"File: {name}")
    print(f"  All keys: {sorted(list(all_keys))}")
    print(f"  Provenance/Quality-related keys: {prov_keys if prov_keys else 'NONE'}")
    if "source_conf" in all_keys or "source_conf_bn" in all_keys:
        # show summary of source_conf and needs_review
        confs = [r.get("source_conf") for r in rows if "source_conf" in r and r.get("source_conf") is not None]
        reviews = [r.get("needs_review") for r in rows if "needs_review" in r]
        print(f"  'source_conf' count: {len(confs)}, min={min(confs) if confs else None}, max={max(confs) if confs else None}")
        print(f"  'needs_review' count: {Counter(reviews)}")

# Check external files or docs for provenance documentation
raw_docs = list((ROOT / "docs").glob("*")) if (ROOT / "docs").exists() else []
print(f"\nDocumentation files in docs/: {[p.name for p in raw_docs]}")
raw_sources_docs = [
    DATA_DIR / "raw_sources" / "SUMMARY.md",
    DATA_DIR / "raw_sources" / "manifest.csv",
    DATA_DIR / "eval_rubric.md",
]
print("Checking documentation files for provenance records:")
for doc in raw_sources_docs:
    print(f"  File {doc.name}: Exists={doc.exists()}")

print("\n" + "=" * 80)
print("SECTION 6: DOMAIN BALANCE")
print("=" * 80)

def domain_breakdown(rows, lang_label=None):
    counts = Counter()
    for r in rows:
        d = r.get("domain", "UNKNOWN")
        if lang_label:
            l = lang_label
        else:
            l = r.get("lang", "UNKNOWN")
        counts[(d, l)] += 1
    return counts

train_bi_domains = domain_breakdown(train_bilingual_rows)
print("\nTrain Bilingual counts by (domain, lang):")
for (d, l), c in sorted(train_bi_domains.items()):
    print(f"  Domain: {d:<15} Lang: {l:<10} Count: {c}")

eval_bn_domains = Counter(r.get("domain", "UNKNOWN") for r in eval_bn_rows)
print("\nEval Bangla counts by domain:")
for d, c in sorted(eval_bn_domains.items()):
    print(f"  Domain: {d:<15} Count: {c}")

eval_en_domains = Counter(r.get("domain", "UNKNOWN") for r in eval_en_rows)
print("\nEval English counts by domain:")
for d, c in sorted(eval_en_domains.items()):
    print(f"  Domain: {d:<15} Count: {c}")

eval_cand_domains = Counter(r.get("domain", "UNKNOWN") for r in eval_cand_rows)
print("\nEval Candidate Pool counts by domain:")
for d, c in sorted(eval_cand_domains.items()):
    print(f"  Domain: {d:<15} Count: {c}")

print("\n" + "=" * 80)
print("SECTION 7: DUPLICATES")
print("=" * 80)

def extract_q_ctx(r):
    # returns (question, context)
    if "messages" in r:
        user_c = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        # extract passage and question from user prompt
        p_match = re.search(r'"""(.*?)"""', user_c, re.DOTALL)
        ctx = p_match.group(1).strip() if p_match else user_c
        q_match = re.search(r'(?:প্রশ্ন:|Question:)\s*(.*?)(?:\n\s*উত্তর:|\n\s*Answer:|$)', user_c, re.DOTALL)
        q = q_match.group(1).strip() if q_match else user_c
        return q, ctx
    else:
        return r.get("instruction", "").strip(), r.get("passage", "").strip()

# Check within train
train_questions = defaultdict(list)
train_contexts = defaultdict(list)

for r in train_bilingual_rows:
    qid = r.get("id")
    q, ctx = extract_q_ctx(r)
    if q:
        train_questions[q].append(qid)
    if ctx:
        train_contexts[ctx].append(qid)

dup_train_q = {q: ids for q, ids in train_questions.items() if len(ids) > 1}
dup_train_ctx = {ctx: ids for ctx, ids in train_contexts.items() if len(ids) > 1}

print(f"Exact Duplicate Questions within train_bilingual: {len(dup_train_q)}")
if dup_train_q:
    print(f"  Sample duplicate question: {list(dup_train_q.keys())[0][:100]}... (count: {len(list(dup_train_q.values())[0])})")

print(f"Exact Duplicate Contexts within train_bilingual: {len(dup_train_ctx)}")
if dup_train_ctx:
    print(f"  Sample duplicate context occurrences: {len(list(dup_train_ctx.values())[0])}")

# Check between train and eval
eval_questions = defaultdict(list)
eval_contexts = defaultdict(list)

for r in eval_bn_rows + eval_en_rows:
    qid = r.get("id")
    q, ctx = extract_q_ctx(r)
    if q:
        eval_questions[q].append(qid)
    if ctx:
        eval_contexts[ctx].append(qid)

overlap_q = set(train_questions.keys()).intersection(set(eval_questions.keys()))
overlap_ctx = set(train_contexts.keys()).intersection(set(eval_contexts.keys()))

print(f"\nExact Duplicate Questions BETWEEN Train and Eval: {len(overlap_q)}")
if overlap_q:
    print(f"  Sample overlapping questions: {list(overlap_q)[:3]}")

print(f"Exact Duplicate Contexts BETWEEN Train and Eval: {len(overlap_ctx)}")
if overlap_ctx:
    print(f"  Sample overlapping context: {list(overlap_ctx)[0][:100]}...")

print("\nAUDIT SCRIPT EXECUTION FINISHED.")
