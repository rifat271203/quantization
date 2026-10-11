"""
Build the three training data files with passage-priority truncation.

Principles:
1. Strict document-level scoping (zero dev/eval leakage).
2. Passage-priority truncation:
   - Instruction is NEVER truncated.
   - Response is NEVER truncated (unless non-passage length alone exceeds max_seq_length, affecting <1.6% of BN).
   - If total length exceeds max_seq_length (1024), ONLY the passage is truncated to fit.
3. Formats each record as {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...]}}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BN_INPUT = DATA_DIR / "eval_bangla.jsonl"
EN_INPUT = DATA_DIR / "eval_english.jsonl"

BN_OUTPUT = DATA_DIR / "train_bn_only.jsonl"
EN_OUTPUT = DATA_DIR / "train_en_only.jsonl"
BILINGUAL_OUTPUT = DATA_DIR / "train_bilingual.jsonl"

MAX_SEQ_LENGTH = 1024

TRAIN_DOCS = {
    "agri_mechanization_2020",
    "disaster_sod_2019",
    "disaster_displacement_nap_2022",
}

DEV_EVAL_DOCS = {
    "agri_seed_rules_2020",
    "disaster_displacement_strat_2021",
    "disaster_emcrp_rpf",
    "disaster_emcrp_esmf",
}


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


def truncate_passage_priority(
    tokenizer,
    passage: str,
    instruction: str,
    response: str,
    lang: str,
    max_len: int = MAX_SEQ_LENGTH,
) -> tuple[str, bool, bool]:
    """
    Fits record into max_len by truncating passage first.
    Returns (fitted_passage, was_passage_truncated, was_response_truncated).
    """
    inst_clean = instruction.strip()
    resp_clean = response.strip()
    pass_clean = passage.strip()

    # 1. Check full length
    u_full = format_user_prompt(pass_clean, inst_clean, lang)
    msgs_full = [{"role": "user", "content": u_full}, {"role": "assistant", "content": resp_clean}]
    txt_full = tokenizer.apply_chat_template(msgs_full, tokenize=False)
    tok_full = tokenizer.encode(txt_full, add_special_tokens=False)

    if len(tok_full) <= max_len:
        return pass_clean, False, False

    # 2. Check non-passage overhead (template + prompt text + instruction + response)
    u_empty = format_user_prompt("", inst_clean, lang)
    msgs_empty = [{"role": "user", "content": u_empty}, {"role": "assistant", "content": resp_clean}]
    txt_empty = tokenizer.apply_chat_template(msgs_empty, tokenize=False)
    tok_empty = tokenizer.encode(txt_empty, add_special_tokens=False)
    non_pass_len = len(tok_empty)

    if non_pass_len >= max_len:
        # Extreme case: instruction + response alone exceeds limit
        return "", True, True

    # 3. Truncate passage tokens to exact allowed budget
    allowed_pass_tokens = max_len - non_pass_len
    # Conservative headroom to avoid special token / subword boundary expansion
    budget = max(0, allowed_pass_tokens - 4)

    pass_tokens = tokenizer.encode(pass_clean, add_special_tokens=False)
    truncated_pass_tokens = pass_tokens[:budget]
    truncated_passage = tokenizer.decode(truncated_pass_tokens, skip_special_tokens=True).strip()

    # Verify fit, reduce slightly if still over due to decode re-tokenization
    while truncated_passage:
        u_test = format_user_prompt(truncated_passage, inst_clean, lang)
        msgs_test = [{"role": "user", "content": u_test}, {"role": "assistant", "content": resp_clean}]
        txt_test = tokenizer.apply_chat_template(msgs_test, tokenize=False)
        test_len = len(tokenizer.encode(txt_test, add_special_tokens=False))
        if test_len <= max_len:
            break
        # trim 10 tokens
        budget = max(0, budget - 10)
        truncated_pass_tokens = pass_tokens[:budget]
        truncated_passage = tokenizer.decode(truncated_pass_tokens, skip_special_tokens=True).strip()

    return truncated_passage, True, False


def process_file(
    file_path: Path, lang: str, max_edu_page: int, tokenizer
) -> tuple[list[dict], dict]:
    train_records = []
    stats = {
        "total": 0,
        "fully_intact": 0,
        "passage_truncated": 0,
        "response_affected": 0,
    }

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id", "")
            page_str = str(item.get("source_page", ""))
            page_num = int(page_str) if page_str.isdigit() else 0

            is_train = False
            if doc_id in TRAIN_DOCS:
                is_train = True
            elif doc_id == "edu_nep_2010" and page_num <= max_edu_page:
                is_train = True

            if is_train:
                stats["total"] += 1
                passage = item.get("passage", "")
                instruction = item.get("instruction", "")
                response = item.get("response", "")

                fitted_pass, p_trunc, r_trunc = truncate_passage_priority(
                    tokenizer=tokenizer,
                    passage=passage,
                    instruction=instruction,
                    response=response,
                    lang=lang,
                    max_len=MAX_SEQ_LENGTH,
                )

                if not p_trunc and not r_trunc:
                    stats["fully_intact"] += 1
                elif p_trunc and not r_trunc:
                    stats["passage_truncated"] += 1
                else:
                    stats["response_affected"] += 1

                user_msg = format_user_prompt(
                    passage=fitted_pass,
                    instruction=instruction,
                    lang=lang,
                )
                assistant_msg = response.strip()

                chat_record = {
                    "id": item["id"],
                    "doc_id": doc_id,
                    "domain": item.get("domain", ""),
                    "lang": lang,
                    "passage_truncated": p_trunc,
                    "response_affected": r_trunc,
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                }
                train_records.append(chat_record)

    return train_records, stats


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Loading tokenizer: google/gemma-3-1b-it...")
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    print(f"\nProcessing training splits with Passage-Priority Truncation (Max Seq = {MAX_SEQ_LENGTH})...")
    bn_records, bn_stats = process_file(BN_INPUT, "bangla", max_edu_page=53, tokenizer=tok)
    en_records, en_stats = process_file(EN_INPUT, "english", max_edu_page=57, tokenizer=tok)

    # Save BN only
    with BN_OUTPUT.open("w", encoding="utf-8") as f:
        for r in bn_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(bn_records)} records to {BN_OUTPUT}")

    # Save EN only
    with EN_OUTPUT.open("w", encoding="utf-8") as f:
        for r in en_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(en_records)} records to {EN_OUTPUT}")

    # Interleave for bilingual
    bilingual_records = []
    max_len = max(len(bn_records), len(en_records))
    for i in range(max_len):
        if i < len(bn_records):
            bilingual_records.append(bn_records[i])
        if i < len(en_records):
            bilingual_records.append(en_records[i])

    with BILINGUAL_OUTPUT.open("w", encoding="utf-8") as f:
        for r in bilingual_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(bilingual_records)} records to {BILINGUAL_OUTPUT}")

    # Final Verification
    print("\n" + "=" * 75)
    print("FINAL TRUNCATION RATES AFTER PASSAGE-PRIORITY FIX (Max Length = 1024)")
    print("=" * 75)
    for name, stats in [("Bangla-only", bn_stats), ("English-only", en_stats)]:
        tot = stats["total"]
        intact_p = (stats["fully_intact"] / tot) * 100
        pass_p = (stats["passage_truncated"] / tot) * 100
        resp_p = (stats["response_affected"] / tot) * 100
        print(f"  {name:15s} (N={tot}):")
        print(f"    - 100% Intact:                    {stats['fully_intact']:4d} ({intact_p:5.2f}%)")
        print(f"    - Passage Truncated (Target Safe):{stats['passage_truncated']:4d} ({pass_p:5.2f}%)")
        print(f"    - Response/Target Truncated:      {stats['response_affected']:4d} ({resp_p:5.2f}%)")

    # Bilingual combined
    bi_intact = bn_stats["fully_intact"] + en_stats["fully_intact"]
    bi_pass = bn_stats["passage_truncated"] + en_stats["passage_truncated"]
    bi_resp = bn_stats["response_affected"] + en_stats["response_affected"]
    bi_tot = len(bilingual_records)
    print(f"  {'Bilingual':15s} (N={bi_tot}):")
    print(f"    - 100% Intact:                    {bi_intact:4d} ({bi_intact/bi_tot*100:5.2f}%)")
    print(f"    - Passage Truncated (Target Safe):{bi_pass:4d} ({bi_pass/bi_tot*100:5.2f}%)")
    print(f"    - Response/Target Truncated:      {bi_resp:4d} ({bi_resp/bi_tot*100:5.2f}%)")
    print("=" * 75)

    # Check maximum tokens across all datasets
    max_bn = max(len(tok.encode(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False)) for r in bn_records)
    max_en = max(len(tok.encode(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False)) for r in en_records)
    print(f"\nVerification of maximum tokens in new files:")
    print(f"  Bangla max sequence length:  {max_bn} tokens (limit: {MAX_SEQ_LENGTH})")
    print(f"  English max sequence length: {max_en} tokens (limit: {MAX_SEQ_LENGTH})")


if __name__ == "__main__":
    main()
