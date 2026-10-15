"""
audit/audit3.py: Deep read-only investigation on tokenizer fertility, truncation,
alignment rankings, usable pool filtering, and chat prompt lengths.

DO NOT modify, rename, or delete any existing file outside audit/.
Uses google/gemma-3-1b-it for tokenization.
Uses intfloat/multilingual-e5-base for embeddings (LaBSE failed with CAS client download error).
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

print("Loading google/gemma-3-1b-it tokenizer...")
gemma_tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

pool_rows = load_jsonl(EVAL_POOL_PATH)
train_bn_rows = load_jsonl(TRAIN_BN_PATH)
train_en_rows = load_jsonl(TRAIN_EN_PATH)

print(f"Loaded {len(pool_rows)} pool rows, {len(train_bn_rows)} train BN rows, {len(train_en_rows)} train EN rows.")

print("\n" + "=" * 80)
print("SECTION A.1: CONTENT LENGTH & TOKEN RATIOS PER DOC_ID")
print("=" * 80)

# 1. For eval_candidate_pool.jsonl and train files per doc_id:
# mean chars, mean whitespace words, mean Gemma tokens per passage (BN and EN), ratio of BN words to EN words
pool_by_doc = defaultdict(list)
for r in pool_rows:
    pool_by_doc[r.get("doc_id", "UNKNOWN")].append(r)

print("\n--- EVAL_CANDIDATE_POOL.JSONL STATS PER DOC_ID ---")
for doc_id, rows in sorted(pool_by_doc.items()):
    bn_chars = [len(r.get("passage_bn", "")) for r in rows]
    en_chars = [len(r.get("passage_en", "")) for r in rows]
    bn_words = [len(r.get("passage_bn", "").split()) for r in rows]
    en_words = [len(r.get("passage_en", "").split()) for r in rows]
    bn_toks = [len(gemma_tok(r.get("passage_bn", ""))["input_ids"]) for r in rows]
    en_toks = [len(gemma_tok(r.get("passage_en", ""))["input_ids"]) for r in rows]
    
    # ratio of BN words to EN words
    w_ratios = [bw / ew if ew > 0 else 0 for bw, ew in zip(bn_words, en_words)]
    
    print(f"\nDoc ID: {doc_id} (count={len(rows)})")
    print(f"  BN: Mean Chars={np.mean(bn_chars):.2f}, Mean Words={np.mean(bn_words):.2f}, Mean Tokens={np.mean(bn_toks):.2f}")
    print(f"  EN: Mean Chars={np.mean(en_chars):.2f}, Mean Words={np.mean(en_words):.2f}, Mean Tokens={np.mean(en_toks):.2f}")
    print(f"  Word Ratio (BN_words / EN_words): Mean={np.mean(w_ratios):.4f}, Median={np.median(w_ratios):.4f}")
    print(f"  Token Ratio (BN_toks / EN_toks):  Mean={np.mean(bn_toks)/np.mean(en_toks):.4f}")

# Train files per doc_id
def extract_train_passage(r):
    msgs = r.get("messages", [])
    user_c = next((m["content"] for m in msgs if m["role"] == "user"), "")
    m = re.search(r'"""(.*?)"""', user_c, re.DOTALL)
    return m.group(1).strip() if m else user_c

train_bn_by_doc = defaultdict(list)
for r in train_bn_rows:
    train_bn_by_doc[r.get("doc_id", "UNKNOWN")].append(extract_train_passage(r))

train_en_by_doc = defaultdict(list)
for r in train_en_rows:
    train_en_by_doc[r.get("doc_id", "UNKNOWN")].append(extract_train_passage(r))

print("\n--- TRAIN FILES STATS PER DOC_ID ---")
all_train_docs = sorted(list(set(train_bn_by_doc.keys()).union(set(train_en_by_doc.keys()))))
for doc_id in all_train_docs:
    bn_passages = train_bn_by_doc.get(doc_id, [])
    en_passages = train_en_by_doc.get(doc_id, [])
    print(f"\nDoc ID: {doc_id} (BN count={len(bn_passages)}, EN count={len(en_passages)})")
    if bn_passages:
        bn_c = [len(p) for p in bn_passages]
        bn_w = [len(p.split()) for p in bn_passages]
        bn_t = [len(gemma_tok(p)["input_ids"]) for p in bn_passages]
        print(f"  BN: Mean Chars={np.mean(bn_c):.2f}, Mean Words={np.mean(bn_w):.2f}, Mean Tokens={np.mean(bn_t):.2f}")
    if en_passages:
        en_c = [len(p) for p in en_passages]
        en_w = [len(p.split()) for p in en_passages]
        en_t = [len(gemma_tok(p)["input_ids"]) for p in en_passages]
        print(f"  EN: Mean Chars={np.mean(en_c):.2f}, Mean Words={np.mean(en_w):.2f}, Mean Tokens={np.mean(en_t):.2f}")
    if bn_passages and en_passages:
        print(f"  Ratio BN_words / EN_words: {np.mean(bn_w)/np.mean(en_w):.4f}")
        print(f"  Ratio BN_tokens / EN_tokens: {np.mean(bn_t)/np.mean(en_t):.4f}")

print("\n" + "=" * 80)
print("SECTION A.2: CHARACTER COMPOSITION & UNUSUAL CHARACTERS")
print("=" * 80)

# Pool rows + train BN rows combined per doc_id for Bangla passages
all_bn_passages_by_doc = defaultdict(list)
for r in pool_rows:
    all_bn_passages_by_doc[r.get("doc_id", "UNKNOWN")].append(r.get("passage_bn", ""))
for doc_id, passages in train_bn_by_doc.items():
    all_bn_passages_by_doc[doc_id].extend(passages)

for doc_id, passages in sorted(all_bn_passages_by_doc.items()):
    total_chars = 0
    bengali_chars = 0
    ascii_chars = 0
    zwj_count = 0   # U+200D
    zwnj_count = 0  # U+200C
    unusual_counts = Counter()

    for p in passages:
        for ch in p:
            total_chars += 1
            cp = ord(ch)
            if 0x0980 <= cp <= 0x09FF:
                bengali_chars += 1
            elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x0030 <= cp <= 0x0039):
                ascii_chars += 1
            elif cp == 0x200D:
                zwj_count += 1
            elif cp == 0x200C:
                zwnj_count += 1
            elif not (ch.isspace() or ch in ".,;:!?'\"-–—()[]{}/*%+=_<>#@\\"):
                unusual_counts[(ch, f"U+{cp:04X}")] += 1

    print(f"\nDoc ID: {doc_id} (Total Chars analyzed: {total_chars:,})")
    print(f"  Bengali Unicode (U+0980 - U+09FF): {bengali_chars} ({bengali_chars/total_chars*100:.2f}%)")
    print(f"  ASCII letters & digits:           {ascii_chars} ({ascii_chars/total_chars*100:.2f}%)")
    print(f"  ZWJ (U+200D):                      {zwj_count}")
    print(f"  ZWNJ (U+200C):                     {zwnj_count}")
    print("  10 Most Frequent Unusual Characters:")
    for (ch, hex_cp), cnt in unusual_counts.most_common(10):
        print(f"    - {repr(ch)} ({hex_cp}): count={cnt}")

print("\n" + "=" * 80)
print("SECTION A.3: TOKENIZER SINGLE-CHAR & BYTE-FALLBACK TOKENS")
print("=" * 80)

for doc_id, passages in sorted(all_bn_passages_by_doc.items()):
    total_tokens = 0
    single_char_tokens = 0
    byte_fallback_tokens = 0

    for p in passages:
        token_ids = gemma_tok(p, add_special_tokens=False)["input_ids"]
        for tid in token_ids:
            total_tokens += 1
            decoded = gemma_tok.decode([tid])
            if len(decoded) == 1:
                single_char_tokens += 1
            # byte fallback check: <0x..> or token string representations
            tok_str = gemma_tok.convert_ids_to_tokens(tid)
            if re.match(r"^<0x[0-9A-Fa-f]{2}>$", str(tok_str)):
                byte_fallback_tokens += 1

    print(f"\nDoc ID: {doc_id} (Total Tokens: {total_tokens:,})")
    print(f"  Single-character decoded tokens: {single_char_tokens} ({single_char_tokens/total_tokens*100:.2f}%)")
    print(f"  Byte-fallback tokens (<0x..>):   {byte_fallback_tokens} ({byte_fallback_tokens/total_tokens*100:.2f}%)")

print("\n" + "=" * 80)
print("SECTION A.4: EXTREME EXAMPLES COMPARISON")
print("=" * 80)

# Disater pool passages
disaster_pool = [r for r in pool_rows if r.get("domain") == "disaster"]
disaster_tpw = []
for r in disaster_pool:
    p = r.get("passage_bn", "")
    w = len(p.split())
    if w > 0:
        t = len(gemma_tok(p, add_special_tokens=False)["input_ids"])
        disaster_tpw.append((t / w, r["question_id"], p))

disaster_tpw.sort(key=lambda x: x[0], reverse=True)

print("--- 3 BANGLA PASSAGES FROM DISASTER DOMAIN WITH HIGHEST TOKENS-PER-WORD ---")
for idx, (tpw, qid, text) in enumerate(disaster_tpw[:3], 1):
    print(f"\n[DISASTER HIGHEST #{idx}] question_id: {qid}, Tokens-per-word: {tpw:.2f}")
    print(f"RAW TEXT (repr): {repr(text[:250])}")
    tok_strs = [gemma_tok.convert_ids_to_tokens(tid) for tid in gemma_tok(text, add_special_tokens=False)["input_ids"][:30]]
    print(f"FIRST 30 TOKENS: {tok_strs}")

# Agriculture pool passages
agri_pool = [r for r in pool_rows if r.get("domain") == "agriculture"]
agri_tpw = []
for r in agri_pool:
    p = r.get("passage_bn", "")
    w = len(p.split())
    if w > 0:
        t = len(gemma_tok(p, add_special_tokens=False)["input_ids"])
        agri_tpw.append((t / w, r["question_id"], p))

agri_tpw.sort(key=lambda x: x[0])

print("\n--- 3 BANGLA PASSAGES FROM AGRICULTURE DOMAIN WITH LOWEST TOKENS-PER-WORD ---")
for idx, (tpw, qid, text) in enumerate(agri_tpw[:3], 1):
    print(f"\n[AGRI LOWEST #{idx}] question_id: {qid}, Tokens-per-word: {tpw:.2f}")
    print(f"RAW TEXT (repr): {repr(text[:250])}")
    tok_strs = [gemma_tok.convert_ids_to_tokens(tid) for tid in gemma_tok(text, add_special_tokens=False)["input_ids"][:30]]
    print(f"FIRST 30 TOKENS: {tok_strs}")

print("\n" + "=" * 80)
print("SECTION B.6: WRITE AUDIT/RESPONSE_AFFECTED.CSV")
print("=" * 80)

# Check all rows in train_bn_only and train_en_only where response_affected == True
resp_affected_rows = []
for r in train_bn_rows:
    if r.get("response_affected") is True:
        resp_affected_rows.append(r)
for r in train_en_rows:
    if r.get("response_affected") is True:
        resp_affected_rows.append(r)

print(f"Found {len(resp_affected_rows)} rows with response_affected == True")
resp_csv_path = AUDIT_DIR / "response_affected.csv"
with resp_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "doc_id", "passage_stored", "response"])
    for r in resp_affected_rows:
        qid = r.get("id")
        doc_id = r.get("doc_id")
        msgs = r.get("messages", [])
        user_c = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_c = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        # extract passage
        m = re.search(r'"""(.*?)"""', user_c, re.DOTALL)
        stored_passage = m.group(1).strip() if m else user_c
        writer.writerow([qid, doc_id, stored_passage, asst_c])

print(f"Successfully wrote {len(resp_affected_rows)} rows to {resp_csv_path}")

print("\n" + "=" * 80)
print("SECTION C.7 & C.8: ALIGNMENT PROPERLY MEASURED (RANKING ACCURACY)")
print("=" * 80)

device = "cpu"
print(f"Loading embedding model intfloat/multilingual-e5-base onto {device}...")
emb_model_id = "intfloat/multilingual-e5-base"
emb_tok = AutoTokenizer.from_pretrained(emb_model_id)
emb_model = AutoModel.from_pretrained(emb_model_id).to(device)
emb_model.eval()

def embed_texts(texts: list[str], batch_size=32, prefix="passage: "):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i+batch_size]]
        encoded = emb_tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = emb_model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embs.append(normalized.numpy())
    return np.vstack(all_embs)

print("Embedding 1,813 BN and EN passages...")
p_bn_embs = embed_texts([r["passage_bn"] for r in pool_rows], prefix="passage: ")
p_en_embs = embed_texts([r["passage_en"] for r in pool_rows], prefix="passage: ")

print("Embedding 1,813 BN and EN instructions...")
q_bn_embs = embed_texts([r["instruction_bn"] for r in pool_rows], prefix="query: ")
q_en_embs = embed_texts([r["instruction_en"] for r in pool_rows], prefix="query: ")

print("Embedding 1,813 BN and EN responses...")
a_bn_embs = embed_texts([r["response_bn"] for r in pool_rows], prefix="passage: ")
a_en_embs = embed_texts([r["response_en"] for r in pool_rows], prefix="passage: ")

def compute_retrieval_ranks(bn_embs, en_embs):
    # sim_matrix: [N, N] where [i, j] is similarity between BN[i] and EN[j]
    sim_matrix = np.dot(bn_embs, en_embs.T)
    top1 = 0
    top5 = 0
    ranks = []
    for i in range(len(bn_embs)):
        # True EN is index i
        sims = sim_matrix[i]
        # rank is 1 + number of ENs with higher similarity than true index
        true_sim = sims[i]
        rank = int(np.sum(sims > true_sim)) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
    return np.array(ranks), top1, top5

print("Calculating ranking accuracy for passages...")
p_ranks, p_top1, p_top5 = compute_retrieval_ranks(p_bn_embs, p_en_embs)
print("Calculating ranking accuracy for instructions...")
q_ranks, q_top1, q_top5 = compute_retrieval_ranks(q_bn_embs, q_en_embs)
print("Calculating ranking accuracy for responses...")
a_ranks, a_top1, a_top5 = compute_retrieval_ranks(a_bn_embs, a_en_embs)

N = len(pool_rows)
print(f"\nOVERALL RANKING ACCURACY (N={N}):")
print(f"  Passages:     Top-1={p_top1}/{N} ({p_top1/N*100:.2f}%), Top-5={p_top5}/{N} ({p_top5/N*100:.2f}%)")
print(f"  Instructions: Top-1={q_top1}/{N} ({q_top1/N*100:.2f}%), Top-5={q_top5}/{N} ({q_top5/N*100:.2f}%)")
print(f"  Responses:    Top-1={a_top1}/{N} ({a_top1/N*100:.2f}%), Top-5={a_top5}/{N} ({a_top5/N*100:.2f}%)")

print("\nPER DOC_ID RANKING ACCURACY (Passages):")
doc_indices = defaultdict(list)
for idx, r in enumerate(pool_rows):
    doc_indices[r.get("doc_id", "UNKNOWN")].append(idx)

for doc_id, indices in sorted(doc_indices.items()):
    d_ranks = p_ranks[indices]
    d_n = len(indices)
    d_t1 = int(np.sum(d_ranks == 1))
    d_t5 = int(np.sum(d_ranks <= 5))
    print(f"  {doc_id:32s}: Top-1={d_t1}/{d_n} ({d_t1/d_n*100:.2f}%), Top-5={d_t5}/{d_n} ({d_t5/d_n*100:.2f}%)")

# Section C.8: Write 40 worst-ranked (or not top-1) passages
worst_indices = np.argsort(-p_ranks)[:40]
misalign_csv_path = AUDIT_DIR / "misaligned_passages.csv"
with misalign_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "doc_id", "rank", "passage_bn", "passage_en"])
    for idx in worst_indices:
        r = pool_rows[idx]
        writer.writerow([r["question_id"], r["doc_id"], p_ranks[idx], r["passage_bn"], r["passage_en"]])
print(f"Wrote 40 worst-ranked passage rows to {misalign_csv_path}")

print("\n" + "=" * 80)
print("SECTION D.9 & D.10: USABLE EVAL ROWS FILTER CASCADE & CHAT LENGTHS")
print("=" * 80)

page_pattern_en = re.compile(r"\bpage\s*\d+\b", re.IGNORECASE)
page_pattern_bn = re.compile(r"(?:পৃষ্ঠা|\d+\s*নং\s*পৃষ্ঠা|পৃষ্ঠা\s*\d+)", re.IGNORECASE)

bn_q_counts = Counter(r.get("instruction_bn", "").strip() for r in pool_rows)
en_q_counts = Counter(r.get("instruction_en", "").strip() for r in pool_rows)

step0_rows = pool_rows # 1813
# Filter (i): no page reference in instruction_bn or instruction_en
step1_rows = [
    r for r in step0_rows 
    if not (page_pattern_en.search(r.get("instruction_en", "")) or page_pattern_bn.search(r.get("instruction_bn", "")))
]

# Filter (ii): instruction_bn and instruction_en each unique in the pool
step2_rows = [
    r for r in step1_rows
    if bn_q_counts[r.get("instruction_bn", "").strip()] == 1 and en_q_counts[r.get("instruction_en", "").strip()] == 1
]

# Filter (iii): both responses at least 5 words
step3_rows = [
    r for r in step2_rows
    if len(r.get("response_bn", "").split()) >= 5 and len(r.get("response_en", "").split()) >= 5
]

# Filter (iv): true-pair passage top-1 in step 7
pool_id_to_rank = {r["question_id"]: p_ranks[i] for i, r in enumerate(pool_rows)}
step4_rows = [
    r for r in step3_rows
    if pool_id_to_rank[r["question_id"]] == 1
]

print("FILTER CASCADE COUNT PROGRESSION:")
print(f"  Step 0: Initial candidate pool:                                           {len(step0_rows)}")
print(f"  Step 1: Filter (i) [No page reference]:                                   {len(step1_rows)}")
print(f"  Step 2: Filter (ii) [Unique instructions in pool]:                        {len(step2_rows)}")
print(f"  Step 3: Filter (iii) [Both responses >= 5 words]:                         {len(step3_rows)}")
print(f"  Step 4: Filter (iv) [True passage pair Top-1 in retrieval ranking]:       {len(step4_rows)}")

# Breakdown per doc_id across the cascade
print("\nBREAKDOWN PER DOC_ID ACROSS THE CASCADE:")
for doc_id in sorted(doc_counts.keys()):
    c0 = sum(1 for r in step0_rows if r["doc_id"] == doc_id)
    c1 = sum(1 for r in step1_rows if r["doc_id"] == doc_id)
    c2 = sum(1 for r in step2_rows if r["doc_id"] == doc_id)
    c3 = sum(1 for r in step3_rows if r["doc_id"] == doc_id)
    c4 = sum(1 for r in step4_rows if r["doc_id"] == doc_id)
    print(f"  {doc_id:32s}: Initial={c0:4d} -> NoPage={c1:4d} -> Unique={c2:4d} -> Resp>=5w={c3:4d} -> Top1={c4:4d}")

usable_csv_path = AUDIT_DIR / "usable_pool.csv"
with usable_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(pool_rows[0].keys()))
    writer.writeheader()
    for r in step4_rows:
        writer.writerow(r)
print(f"\nWritten {len(step4_rows)} final surviving rows to {usable_csv_path}")

# Task 10: Full chat-formatted prompt lengths with Gemma chat template
print("\nTASK 10: GEMMA CHAT-FORMATTED PROMPT LENGTHS (Passage + Instruction)")

def format_prompt(passage, instruction, lang):
    if lang == "bangla":
        content = (
            "অনুগ্রহ করে নিচের অনুচ্ছেদটি মনোযোগ দিয়ে পড়ুন এবং শুধুমাত্র এই অনুচ্ছেদের তথ্যের ভিত্তিতে প্রশ্নের সঠিক উত্তর দিন। অনুচ্ছেদে নেই এমন কোনো তথ্য বা অনুমান যোগ করবেন না।\n\n"
            f"অনুচ্ছেদ:\n\"\"\"\n{passage}\n\"\"\"\n\n"
            f"প্রশ্ন:\n{instruction}\n\n"
            "উত্তর:"
        )
    else:
        content = (
            "Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage. Do not introduce outside knowledge or speculation.\n\n"
            f"Passage:\n\"\"\"\n{passage}\n\"\"\"\n\n"
            f"Question:\n{instruction}\n\n"
            "Answer:"
        )
    messages = [{"role": "user", "content": content}]
    return gemma_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

bn_prompt_lens = []
en_prompt_lens = []

for r in pool_rows:
    p_bn_fmt = format_prompt(r["passage_bn"], r["instruction_bn"], "bangla")
    p_en_fmt = format_prompt(r["passage_en"], r["instruction_en"], "english")
    bn_prompt_lens.append(len(gemma_tok(p_bn_fmt)["input_ids"]))
    en_prompt_lens.append(len(gemma_tok(p_en_fmt)["input_ids"]))

bn_arr = np.array(bn_prompt_lens)
en_arr = np.array(en_prompt_lens)

print("\n--- BANGLA CHAT-FORMATTED PROMPT LENGTHS (N=1,813) ---")
print(f"  p50:   {np.median(bn_arr):.1f}")
print(f"  p90:   {np.percentile(bn_arr, 90):.1f}")
print(f"  p99:   {np.percentile(bn_arr, 99):.1f}")
print(f"  Max:   {np.max(bn_arr)}")
print(f"  Exceeding 1024 tokens: {np.sum(bn_arr > 1024)} ({np.sum(bn_arr > 1024)/len(bn_arr)*100:.2f}%)")

print("\n--- ENGLISH CHAT-FORMATTED PROMPT LENGTHS (N=1,813) ---")
print(f"  p50:   {np.median(en_arr):.1f}")
print(f"  p90:   {np.percentile(en_arr, 90):.1f}")
print(f"  p99:   {np.percentile(en_arr, 99):.1f}")
print(f"  Max:   {np.max(en_arr)}")
print(f"  Exceeding 1024 tokens: {np.sum(en_arr > 1024)} ({np.sum(en_arr > 1024)/len(en_arr)*100:.2f}%)")

print("\nAUDIT3 SCRIPT EXECUTION FINISHED.")
