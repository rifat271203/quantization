"""
audit/audit2.py: Read-only deep audit of eval_candidate_pool.jsonl and train files.
Uses eval_candidate_pool.jsonl as the ONLY eval source.
Uses google/gemma-3-1b-it tokenizer.
Uses intfloat/multilingual-e5-small for BN-EN multilingual embeddings.
"""

import sys
import os
import json
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\Tokenizer\quant-hallucination-study")
DATA_DIR = ROOT / "data"
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_POOL_PATH = DATA_DIR / "eval_candidate_pool.jsonl"
TRAIN_BN_PATH = DATA_DIR / "train_bn_only.jsonl"
TRAIN_EN_PATH = DATA_DIR / "train_en_only.jsonl"

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

print("=" * 80)
print("TASK 1: TRUNCATION FLAGS IN TRAIN FILES")
print("=" * 80)

def audit_truncation_flags(path: Path, lang_name: str):
    rows = load_jsonl(path)
    total_rows = len(rows)
    pass_trunc_total = 0
    resp_aff_total = 0
    per_doc = defaultdict(lambda: {"total": 0, "pass_trunc": 0, "resp_aff": 0})
    for r in rows:
        doc = r.get("doc_id", "UNKNOWN")
        pt = bool(r.get("passage_truncated", False))
        ra = bool(r.get("response_affected", False))
        per_doc[doc]["total"] += 1
        if pt:
            pass_trunc_total += 1
            per_doc[doc]["pass_trunc"] += 1
        if ra:
            resp_aff_total += 1
            per_doc[doc]["resp_aff"] += 1
    print(f"\nLanguage: {lang_name} (Total items: {total_rows})")
    print(f"  passage_truncated == True: {pass_trunc_total} ({pass_trunc_total/total_rows*100:.2f}%)")
    print(f"  response_affected == True: {resp_aff_total} ({resp_aff_total/total_rows*100:.2f}%)")
    print("  Per doc_id breakdown:")
    for doc, dstat in sorted(per_doc.items()):
        print(f"    - {doc}: total={dstat['total']}, passage_truncated={dstat['pass_trunc']} ({dstat['pass_trunc']/dstat['total']*100:.2f}%), response_affected={dstat['resp_aff']} ({dstat['resp_aff']/dstat['total']*100:.2f}%)")

audit_truncation_flags(TRAIN_BN_PATH, "Bangla (train_bn_only.jsonl)")
audit_truncation_flags(TRAIN_EN_PATH, "English (train_en_only.jsonl)")

print("\n" + "=" * 80)
print("TASK 2: PARALLEL TOKEN RATIO IN EVAL_CANDIDATE_POOL.JSONL")
print("=" * 80)

print("Loading google/gemma-3-1b-it tokenizer...")
gemma_tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
pool_rows = load_jsonl(EVAL_POOL_PATH)
print(f"Loaded {len(pool_rows)} rows from eval_candidate_pool.jsonl")

passage_ratios_overall = []
passage_ratios_by_domain = defaultdict(list)

prompt_ratios_overall = []
prompt_ratios_by_domain = defaultdict(list)

bn_tpw_list = []
en_tpw_list = []

for r in pool_rows:
    dom = r.get("domain", "UNKNOWN")
    p_bn = r.get("passage_bn", "")
    p_en = r.get("passage_en", "")
    q_bn = r.get("instruction_bn", "")
    q_en = r.get("instruction_en", "")
    a_bn = r.get("response_bn", "")
    a_en = r.get("response_en", "")

    # Tokenize passages
    p_bn_toks = len(gemma_tok(p_bn)["input_ids"])
    p_en_toks = len(gemma_tok(p_en)["input_ids"])
    if p_en_toks > 0:
        p_ratio = p_bn_toks / p_en_toks
        passage_ratios_overall.append(p_ratio)
        passage_ratios_by_domain[dom].append(p_ratio)

    # Full prompt: passage + instruction + response
    prompt_bn_text = f"{p_bn} {q_bn} {a_bn}"
    prompt_en_text = f"{p_en} {q_en} {a_en}"
    pr_bn_toks = len(gemma_tok(prompt_bn_text)["input_ids"])
    pr_en_toks = len(gemma_tok(prompt_en_text)["input_ids"])
    if pr_en_toks > 0:
        pr_ratio = pr_bn_toks / pr_en_toks
        prompt_ratios_overall.append(pr_ratio)
        prompt_ratios_by_domain[dom].append(pr_ratio)

    # Plain text tokens per word (whitespace-split words)
    bn_words = len(p_bn.split())
    en_words = len(p_en.split())
    if bn_words > 0:
        bn_tpw_list.append(p_bn_toks / bn_words)
    if en_words > 0:
        en_tpw_list.append(p_en_toks / en_words)

def print_ratio_stats(arr, title):
    arr = np.array(arr)
    mean_v = np.mean(arr)
    med_v = np.median(arr)
    p10_v = np.percentile(arr, 10)
    p90_v = np.percentile(arr, 90)
    print(f"  {title:35s}: Mean={mean_v:.4f}, Median={med_v:.4f}, p10={p10_v:.4f}, p90={p90_v:.4f}")

print("\n(A) Passage bn_tokens / en_tokens ratio:")
print_ratio_stats(passage_ratios_overall, "OVERALL")
for dom, arr in sorted(passage_ratios_by_domain.items()):
    print_ratio_stats(arr, f"Domain: {dom}")

print("\n(B) Full Prompt (Passage + Q + A) bn_tokens / en_tokens ratio:")
print_ratio_stats(prompt_ratios_overall, "OVERALL")
for dom, arr in sorted(prompt_ratios_by_domain.items()):
    print_ratio_stats(arr, f"Domain: {dom}")

print("\n(C) Plain-Text Tokens per Word:")
print(f"  Bangla Tokens per Word:  Mean={np.mean(bn_tpw_list):.4f}, Median={np.median(bn_tpw_list):.4f}, p10={np.percentile(bn_tpw_list, 10):.4f}, p90={np.percentile(bn_tpw_list, 90):.4f}")
print(f"  English Tokens per Word: Mean={np.mean(en_tpw_list):.4f}, Median={np.median(en_tpw_list):.4f}, p10={np.percentile(en_tpw_list, 10):.4f}, p90={np.percentile(en_tpw_list, 90):.4f}")

print("\n" + "=" * 80)
print("TASK 3: EVAL POOL COMPOSITION & SOURCE CONFIDENCE")
print("=" * 80)

domain_counts = Counter(r.get("domain", "UNKNOWN") for r in pool_rows)
doc_counts = Counter(r.get("doc_id", "UNKNOWN") for r in pool_rows)
needs_review_count = sum(1 for r in pool_rows if r.get("needs_review") is True)

print(f"Total rows in eval_candidate_pool.jsonl: {len(pool_rows)}")
print(f"Rows with needs_review == True: {needs_review_count} ({needs_review_count/len(pool_rows)*100:.2f}%)")

print("\nCounts per domain:")
for dom, c in sorted(domain_counts.items()):
    print(f"  {dom:<20}: {c} ({c/len(pool_rows)*100:.2f}%)")

print("\nCounts per doc_id:")
for doc, c in sorted(doc_counts.items()):
    print(f"  {doc:<35}: {c} ({c/len(pool_rows)*100:.2f}%)")

# Per doc_id source_conf_bn and source_conf_en
doc_conf_bn = defaultdict(list)
doc_conf_en = defaultdict(list)

for r in pool_rows:
    doc = r.get("doc_id", "UNKNOWN")
    c_bn = r.get("source_conf_bn")
    c_en = r.get("source_conf_en")
    if c_bn is not None:
        doc_conf_bn[doc].append(float(c_bn))
    if c_en is not None:
        doc_conf_en[doc].append(float(c_en))

print("\nSource Confidence Distribution per doc_id:")
for doc in sorted(doc_counts.keys()):
    print(f"\nDocument: {doc}")
    bn_vals = np.array(doc_conf_bn[doc]) if doc_conf_bn[doc] else None
    en_vals = np.array(doc_conf_en[doc]) if doc_conf_en[doc] else None
    
    if bn_vals is not None and len(bn_vals) > 0:
        print(f"  source_conf_bn (n={len(bn_vals)}): Min={np.min(bn_vals):.2f}, p5={np.percentile(bn_vals, 5):.2f}, p25={np.percentile(bn_vals, 25):.2f}, Median={np.median(bn_vals):.2f}, <95={np.sum(bn_vals < 95)}, <90={np.sum(bn_vals < 90)}")
    else:
        print("  source_conf_bn: NO DATA / EMPTY")

    if en_vals is not None and len(en_vals) > 0:
        print(f"  source_conf_en (n={len(en_vals)}): Min={np.min(en_vals):.2f}, p5={np.percentile(en_vals, 5):.2f}, p25={np.percentile(en_vals, 25):.2f}, Median={np.median(en_vals):.2f}, <95={np.sum(en_vals < 95)}, <90={np.sum(en_vals < 90)}")
    else:
        print("  source_conf_en: NO DATA / EMPTY")

print("\n" + "=" * 80)
print("TASK 4 & 6: BN-EN ALIGNMENT & QUESTION PARALLELISM EMBEDDINGS")
print("=" * 80)

device = "cpu"
print(f"Loading embedding model intfloat/multilingual-e5-small onto {device}...")
emb_model_id = "intfloat/multilingual-e5-small"
emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_id)
emb_model = AutoModel.from_pretrained(emb_model_id).to(device)
emb_model.eval()

def embed_texts(texts: list[str], batch_size=32, prefix="passage: "):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i+batch_size]]
        encoded = emb_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = emb_model(**encoded)
            # Mean pooling with attention mask
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            # Normalize
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embs.append(normalized.cpu().numpy())
    return np.vstack(all_embs)

print("Embedding passage_bn and passage_en...")
p_bn_texts = [r.get("passage_bn", "") for r in pool_rows]
p_en_texts = [r.get("passage_en", "") for r in pool_rows]

p_bn_embs = embed_texts(p_bn_texts, prefix="passage: ")
p_en_embs = embed_texts(p_en_texts, prefix="passage: ")

# Cosine similarity is dot product of normalized vectors
passage_sims = np.sum(p_bn_embs * p_en_embs, axis=1)

print("\n--- Passage Alignment Similarity Distribution (intfloat/multilingual-e5-small) ---")
print(f"  Min:    {np.min(passage_sims):.4f}")
print(f"  p5:     {np.percentile(passage_sims, 5):.4f}")
print(f"  p25:    {np.percentile(passage_sims, 25):.4f}")
print(f"  Median: {np.median(passage_sims):.4f}")
print(f"  Mean:   {np.mean(passage_sims):.4f}")

# Write 50 lowest-similarity rows to audit/low_alignment.csv
lowest_50_indices = np.argsort(passage_sims)[:50]
low_align_csv_path = AUDIT_DIR / "low_alignment.csv"
with low_align_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "similarity", "passage_bn", "passage_en"])
    for idx in lowest_50_indices:
        r = pool_rows[idx]
        writer.writerow([r.get("question_id"), f"{passage_sims[idx]:.4f}", r.get("passage_bn"), r.get("passage_en")])

print(f"\nWritten 50 lowest-similarity rows to {low_align_csv_path}")

# Task 6: Question parallelism
print("\nEmbedding instruction_bn and instruction_en for Task 6...")
q_bn_texts = [r.get("instruction_bn", "") for r in pool_rows]
q_en_texts = [r.get("instruction_en", "") for r in pool_rows]

q_bn_embs = embed_texts(q_bn_texts, prefix="query: ")
q_en_embs = embed_texts(q_en_texts, prefix="query: ")

q_sims = np.sum(q_bn_embs * q_en_embs, axis=1)

print("\n--- Question (Instruction) Parallelism Similarity Distribution ---")
print(f"  Min:    {np.min(q_sims):.4f}")
print(f"  p5:     {np.percentile(q_sims, 5):.4f}")
print(f"  p25:    {np.percentile(q_sims, 25):.4f}")
print(f"  Median: {np.median(q_sims):.4f}")
print(f"  Mean:   {np.mean(q_sims):.4f}")

below_06_count = int(np.sum(q_sims < 0.6))
print(f"  Count below 0.6 similarity: {below_06_count} ({below_06_count/len(q_sims)*100:.2f}%)")

print("\n" + "=" * 80)
print("TASK 5: QUESTION QUALITY FLAGS")
print("=" * 80)

# Flags:
# (a) instruction contains page number reference like "Page N", "পৃষ্ঠা", "নং পৃষ্ঠা" in either language
# (b) instruction text appears in more than one row (exact duplicate in either BN or EN)
# (c) instruction is under 5 words (< 5 words in either BN or EN)
# (d) response_bn or response_en is under 5 words

page_pattern_en = re.compile(r"\bpage\s*\d+\b", re.IGNORECASE)
page_pattern_bn = re.compile(r"(?:পৃষ্ঠা|\d+\s*নং\s*পৃষ্ঠা|পৃষ্ঠা\s*\d+)", re.IGNORECASE)

bn_instructions = [r.get("instruction_bn", "").strip() for r in pool_rows]
en_instructions = [r.get("instruction_en", "").strip() for r in pool_rows]

bn_q_counts = Counter(bn_instructions)
en_q_counts = Counter(en_instructions)

flag_a_count = 0
flag_b_count = 0
flag_c_count = 0
flag_d_count = 0

flags_csv_path = AUDIT_DIR / "question_flags.csv"
with flags_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "question_id",
        "has_page_ref",
        "is_duplicate_instruction",
        "instruction_under_5_words",
        "response_under_5_words",
        "instruction_bn",
        "instruction_en"
    ])
    for r in pool_rows:
        qid = r.get("question_id")
        q_bn = r.get("instruction_bn", "").strip()
        q_en = r.get("instruction_en", "").strip()
        ans_bn = r.get("response_bn", "").strip()
        ans_en = r.get("response_en", "").strip()

        # (a) Page ref
        has_page = bool(page_pattern_en.search(q_en) or page_pattern_bn.search(q_bn))
        if has_page:
            flag_a_count += 1

        # (b) Duplicate instruction
        is_dup = (bn_q_counts[q_bn] > 1) or (en_q_counts[q_en] > 1)
        if is_dup:
            flag_b_count += 1

        # (c) Under 5 words
        under_5_words_q = (len(q_bn.split()) < 5) or (len(q_en.split()) < 5)
        if under_5_words_q:
            flag_c_count += 1

        # (d) Response under 5 words
        under_5_words_a = (len(ans_bn.split()) < 5) or (len(ans_en.split()) < 5)
        if under_5_words_a:
            flag_d_count += 1

        writer.writerow([
            qid,
            has_page,
            is_dup,
            under_5_words_q,
            under_5_words_a,
            q_bn,
            q_en
        ])

print(f"Written question flags to {flags_csv_path}")
print(f"  Flag (a) Contains page reference ('Page N' / 'পৃষ্ঠা'): {flag_a_count} ({flag_a_count/len(pool_rows)*100:.2f}%)")
print(f"  Flag (b) Duplicate instruction text (>1 row):           {flag_b_count} ({flag_b_count/len(pool_rows)*100:.2f}%)")
print(f"  Flag (c) Instruction is under 5 words (<5 words):       {flag_c_count} ({flag_c_count/len(pool_rows)*100:.2f}%)")
print(f"  Flag (d) Response is under 5 words (<5 words):          {flag_d_count} ({flag_d_count/len(pool_rows)*100:.2f}%)")

print("\n" + "=" * 80)
print("TASK 7: REFERENCES TO EVAL_BANGLA.JSONL AND EVAL_ENGLISH.JSONL")
print("=" * 80)

EXTS = (".py", ".ipynb", ".md", ".ps1", ".sh")
MATCH_TERMS = ["eval_bangla.jsonl", "eval_english.jsonl"]

ref_matches = []
for root, dirs, files in os.walk(ROOT):
    # skip .venv, .git
    if ".venv" in root or ".git" in root or "__pycache__" in root:
        continue
    for f in files:
        if f.endswith(EXTS):
            fpath = Path(root) / f
            try:
                with fpath.open("r", encoding="utf-8", errors="ignore") as file_obj:
                    for line_no, line in enumerate(file_obj, 1):
                        for term in MATCH_TERMS:
                            if term in line:
                                rel_path = fpath.relative_to(ROOT)
                                ref_matches.append((str(rel_path), line_no, term, line.strip()))
            except Exception as e:
                pass

print(f"Found {len(ref_matches)} references to eval_bangla.jsonl / eval_english.jsonl:")
for rel_path, line_no, term, line_content in ref_matches:
    print(f"  {rel_path}:{line_no} [{term}] -> {line_content[:100]}")

print("\nAUDIT2 SCRIPT EXECUTION FINISHED.")
