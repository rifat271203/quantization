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

    print("=" * 75)
    print("PASSAGE-PRIORITY TRUNCATION SIMULATION")
    print("=" * 75)

    for limit in [1024, 1280, 1536]:
        print(f"\n--- Limit: {limit} tokens ---")
        for name, rows, raw_map, lang in [("Bangla", bn_train, bn_raw, "bangla"), ("English", en_train, en_raw, "english")]:
            passage_truncated = 0
            response_affected = 0
            fully_intact = 0
            total = len(rows)

            for r in rows:
                raw = raw_map[r["id"]]
                inst = raw["instruction"].strip()
                resp = raw["response"].strip()
                passg = raw["passage"].strip()

                # Full length
                full_u = format_user_prompt(passg, inst, lang)
                full_msgs = [{"role": "user", "content": full_u}, {"role": "assistant", "content": resp}]
                full_txt = tok.apply_chat_template(full_msgs, tokenize=False)
                full_tok_len = len(tok.encode(full_txt, add_special_tokens=False))

                if full_tok_len <= limit:
                    fully_intact += 1
                else:
                    # Non-passage length
                    empty_u = format_user_prompt("", inst, lang)
                    empty_msgs = [{"role": "user", "content": empty_u}, {"role": "assistant", "content": resp}]
                    empty_txt = tok.apply_chat_template(empty_msgs, tokenize=False)
                    non_pass_len = len(tok.encode(empty_txt, add_special_tokens=False))

                    if non_pass_len > limit:
                        response_affected += 1
                    else:
                        passage_truncated += 1

            p_pct = (passage_truncated / total) * 100
            r_pct = (response_affected / total) * 100
            intact_pct = (fully_intact / total) * 100
            print(f"  {name:8s} (N={total}):")
            print(f"    100% Intact:         {fully_intact:4d} ({intact_pct:5.2f}%)")
            print(f"    Passage Truncated:   {passage_truncated:4d} ({p_pct:5.2f}%) [Response 100% Preserved]")
            print(f"    Response Affected:   {response_affected:4d} ({r_pct:5.2f}%)")

if __name__ == "__main__":
    main()
