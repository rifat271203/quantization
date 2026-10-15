import csv
import json
import random
import sys
from pathlib import Path
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_V2 = ROOT / "data_v2"
DATA_V2.mkdir(exist_ok=True)

print("--- Step 1: Loading Belebele for ben_Beng and eng_Latn ---")
ds_bn = load_dataset("facebook/belebele", "ben_Beng", split="test")
ds_en = load_dataset("facebook/belebele", "eng_Latn", split="test")

bn_by_key = {(x["link"], x["question_number"]): x for x in ds_bn}
en_by_key = {(x["link"], x["question_number"]): x for x in ds_en}

common_keys = sorted(list(set(bn_by_key.keys()) & set(en_by_key.keys())))
print(f"Total matched parallel pairs: {len(common_keys)}")
assert len(common_keys) == 900, f"Expected 900 pairs, got {len(common_keys)}"

all_pairs = []
for idx, k in enumerate(common_keys, start=1):
    b = bn_by_key[k]
    e = en_by_key[k]
    assert str(b["correct_answer_num"]) == str(e["correct_answer_num"]), f"Answer mismatch for key {k}"
    
    pair_id = f"bel_{idx:04d}"
    all_pairs.append({
        "pair_id": pair_id,
        "link": k[0],
        "question_number": k[1],
        "correct_answer_num": str(b["correct_answer_num"]),
        "bn": {
            "passage": b["flores_passage"],
            "question": b["question"],
            "options": {
                "1": b["mc_answer1"],
                "2": b["mc_answer2"],
                "3": b["mc_answer3"],
                "4": b["mc_answer4"],
            },
            "correct_answer_num": str(b["correct_answer_num"]),
        },
        "en": {
            "passage": e["flores_passage"],
            "question": e["question"],
            "options": {
                "1": e["mc_answer1"],
                "2": e["mc_answer2"],
                "3": e["mc_answer3"],
                "4": e["mc_answer4"],
            },
            "correct_answer_num": str(e["correct_answer_num"]),
        }
    })

# Sample 300 standard items with fixed seed
rng = random.Random(42)
sampled_300_indices = set(rng.sample(range(len(all_pairs)), 300))

for idx, p in enumerate(all_pairs):
    p["is_primary_300"] = (idx in sampled_300_indices)

# Write full parallel JSONL
parallel_out = DATA_V2 / "belebele_eval_parallel.jsonl"
with parallel_out.open("w", encoding="utf-8") as f:
    for p in all_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Wrote {len(all_pairs)} rows to {parallel_out}")

# Write 300 primary subset formatted for evaluate_models.py (Bangla and English)
eval_bn_300 = DATA_V2 / "belebele_eval_bn_300.jsonl"
eval_en_300 = DATA_V2 / "belebele_eval_en_300.jsonl"

with eval_bn_300.open("w", encoding="utf-8") as f_bn, eval_en_300.open("w", encoding="utf-8") as f_en:
    for p in all_pairs:
        if not p["is_primary_300"]:
            continue
        
        # Build options text
        bn_opts = "\n".join([f"({k}) {v}" for k, v in p["bn"]["options"].items()])
        en_opts = "\n".join([f"({k}) {v}" for k, v in p["en"]["options"].items()])
        
        bn_q = f"{p['bn']['question']}\n\nবিকল্পসমূহ:\n{bn_opts}"
        en_q = f"{p['en']['question']}\n\nOptions:\n{en_opts}"
        
        correct_num = p["correct_answer_num"]
        bn_gold_ans = f"({correct_num}) {p['bn']['options'][correct_num]}"
        en_gold_ans = f"({correct_num}) {p['en']['options'][correct_num]}"
        
        row_bn = {
            "id": p["pair_id"],
            "passage": p["bn"]["passage"],
            "instruction": bn_q,
            "response": bn_gold_ans,
            "correct_option": correct_num,
            "domain": "general_flores",
            "source": "belebele",
        }
        row_en = {
            "id": p["pair_id"],
            "passage": p["en"]["passage"],
            "instruction": en_q,
            "response": en_gold_ans,
            "correct_option": correct_num,
            "domain": "general_flores",
            "source": "belebele",
        }
        
        f_bn.write(json.dumps(row_bn, ensure_ascii=False) + "\n")
        f_en.write(json.dumps(row_en, ensure_ascii=False) + "\n")

print(f"Wrote 300 primary items to {eval_bn_300} and {eval_en_300}")

print("\n--- Step 2: Preparing In-Domain Policy Case Study Manifest ---")
final60_path = DATA_V2 / "final60.csv"
rows_final60 = list(csv.DictReader(final60_path.open(encoding="utf-8-sig")))
print(f"Loaded {len(rows_final60)} passages from {final60_path}")

# Format policy case study rows
policy_rows = []
for r in rows_final60:
    policy_rows.append({
        "case_id": r["scr_id"],
        "domain": r["domain"],
        "doc_id": r["doc_id"],
        "page": r["page"],
        "start_word": r["start_word"],
        "end_word": r["end_word"],
        "words": r["words"],
        "text": r["text"],
        "notes": r.get("notes", "")
    })

policy_out = DATA_V2 / "policy_case_study.jsonl"
with policy_out.open("w", encoding="utf-8") as f:
    for row in policy_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Wrote {len(policy_rows)} in-domain policy passages to {policy_out}")
