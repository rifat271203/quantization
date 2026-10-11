import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def format_user_prompt(passage: str, instruction: str, lang: str) -> str:
    passage_clean = passage.strip()
    instruction_clean = instruction.strip()
    if lang == "bangla":
        return (
            "অনুগ্রহ করে নিচের অনুচ্ছেদটি মনোযোগ দিয়ে পড়ুন এবং শুধুমাত্র এই অনুচ্ছেদের তথ্যের "
            "ভিত্তিতে প্রশ্নের সঠিক উত্তর দিন। অনুচ্ছেদে নেই এমন কোনো তথ্য বা অনুমান যোগ করবেন না।\n\n"
            f"অনুচ্ছেদ:\n\"\"\"\n{passage_clean}\n\"\"\"\n\n"
            f"প্রশ্ন:\n{instruction_clean}\n\n"
            "উত্তর:"
        )
    else:
        return (
            "Read the following passage carefully and answer the question based ONLY on facts "
            "directly stated in the passage. Do not introduce outside knowledge or speculation.\n\n"
            f"Passage:\n\"\"\"\n{passage_clean}\n\"\"\"\n\n"
            f"Question:\n{instruction_clean}\n\n"
            "Answer:"
        )

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    with open(DATA_DIR / "eval_bangla.jsonl", encoding="utf-8") as f:
        bn_raw = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}
    with open(DATA_DIR / "eval_english.jsonl", encoding="utf-8") as f:
        en_raw = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

    with open(DATA_DIR / "train_bn_only.jsonl", encoding="utf-8") as f:
        bn_train = [json.loads(l) for l in f if l.strip()]
    with open(DATA_DIR / "train_en_only.jsonl", encoding="utf-8") as f:
        en_train = [json.loads(l) for l in f if l.strip()]

    print("=" * 70)
    print("NON-PASSAGE (Instruction + Response + Prompt Template) TOKEN BUDGET")
    print("=" * 70)

    for name, rows, raw_map, lang in [
        ("Bangla", bn_train, bn_raw, "bangla"),
        ("English", en_train, en_raw, "english"),
    ]:
        max_non_passage = 0
        over_1024_non_passage = 0
        total_items = len(rows)

        for r in rows:
            raw = raw_map[r["id"]]
            inst = raw["instruction"].strip()
            resp = raw["response"].strip()

            u_empty = format_user_prompt("", inst, lang)
            msgs = [{"role": "user", "content": u_empty}, {"role": "assistant", "content": resp}]
            txt = tok.apply_chat_template(msgs, tokenize=False)
            tok_len = len(tok.encode(txt, add_special_tokens=False))
            if tok_len > max_non_passage:
                max_non_passage = tok_len
            if tok_len > 1024:
                over_1024_non_passage += 1

        print(f"{name:8s}: Max(Non-Passage)={max_non_passage} tokens. Exceeding 1024={over_1024_non_passage}/{total_items}")

if __name__ == "__main__":
    main()
