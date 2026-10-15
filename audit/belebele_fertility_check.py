import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_V2 = ROOT / "data_v2"

print("Loading Belebele 300 primary items...")
bn_file = DATA_V2 / "belebele_eval_bn_300.jsonl"
en_file = DATA_V2 / "belebele_eval_en_300.jsonl"

bn_rows = [json.loads(line) for line in bn_file.open(encoding="utf-8")]
en_rows = [json.loads(line) for line in en_file.open(encoding="utf-8")]

models = [
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
]

def word_count(text):
    return max(len(text.split()), 1)

print(f"\nEvaluating Tokenizer Fertility over 300 Belebele Passages + Questions...")
print(f"{'Model':<35} | {'BN Fert':<8} | {'EN Fert':<8} | {'Ratio (BN/EN)':<13}")
print("-" * 75)

for model_id in models:
    tok = AutoTokenizer.from_pretrained(model_id)
    
    bn_fertilities = []
    en_fertilities = []
    
    for b, e in zip(bn_rows, en_rows):
        b_text = b["passage"] + " " + b["instruction"]
        e_text = e["passage"] + " " + e["instruction"]
        
        b_toks = len(tok.encode(b_text, add_special_tokens=False))
        b_words = word_count(b_text)
        bn_fertilities.append(b_toks / b_words)
        
        e_toks = len(tok.encode(e_text, add_special_tokens=False))
        e_words = word_count(e_text)
        en_fertilities.append(e_toks / e_words)
        
    avg_bn = sum(bn_fertilities) / len(bn_fertilities)
    avg_en = sum(en_fertilities) / len(en_fertilities)
    ratio = avg_bn / avg_en
    print(f"{model_id:<35} | {avg_bn:<8.4f} | {avg_en:<8.4f} | {ratio:<13.4f}")
