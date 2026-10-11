"""
Analyze sequence length breakdown and truncation dynamics for Bangla and English.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BN_TRAIN = DATA_DIR / "train_bn_only.jsonl"
EN_TRAIN = DATA_DIR / "train_en_only.jsonl"

def load_data(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    bn_rows = load_data(BN_TRAIN)
    en_rows = load_data(EN_TRAIN)

    print(f"Loaded {len(bn_rows)} Bangla rows and {len(en_rows)} English rows.\n")

    # Sequence length distribution for both languages
    for max_l in [1024, 1280, 1536, 1792, 2048]:
        bn_ex = sum(1 for r in bn_rows if len(tok.encode(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False)) > max_l)
        en_ex = sum(1 for r in en_rows if len(tok.encode(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False)) > max_l)
        print(f"Threshold {max_l}:")
        print(f"  BN: {bn_ex}/{len(bn_rows)} exceeding ({bn_ex/len(bn_rows)*100:.2f}%)")
        print(f"  EN: {en_ex}/{len(en_rows)} exceeding ({en_ex/len(en_rows)*100:.2f}%)")

    # Now inspect 5 examples from BN that exceed 1024
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN OF 5 BANGLA EXAMPLES EXCEEDING 1024 TOKENS")
    print("=" * 80)

    # Let's inspect raw inputs to get passage, instruction, response separately
    # Or extract from user message
    with (DATA_DIR / "eval_bangla.jsonl").open(encoding="utf-8") as f:
        eval_bn_map = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    exceeding_items = []
    for r in bn_rows:
        raw = eval_bn_map[r["id"]]
        full_text = tok.apply_chat_template(r["messages"], tokenize=False)
        full_toks = tok.encode(full_text, add_special_tokens=False)
        if len(full_toks) > 1024:
            exceeding_items.append((r["id"], raw, len(full_toks)))

    # Select 5 representative items (spread across different lengths)
    # Sort by length
    exceeding_items.sort(key=lambda x: x[2])
    sample_indices = [0, len(exceeding_items)//4, len(exceeding_items)//2, 3*len(exceeding_items)//4, -1]
    samples = [exceeding_items[i] for i in sample_indices]

    for idx, (item_id, raw, total_len) in enumerate(samples, start=1):
        passage = raw.get("passage", "").strip()
        instruction = raw.get("instruction", "").strip()
        response = raw.get("response", "").strip()

        p_toks = len(tok.encode(passage, add_special_tokens=False))
        i_toks = len(tok.encode(instruction, add_special_tokens=False))
        r_toks = len(tok.encode(response, add_special_tokens=False))
        template_overhead = total_len - (p_toks + i_toks + r_toks)

        # What happens if right-truncated to 1024?
        # The tokens beyond 1024 are removed from the END of the sequence.
        lost_tokens = total_len - 1024
        # Since response is at the end:
        response_lost = min(lost_tokens, r_toks)
        response_retained = r_toks - response_lost
        pct_response_lost = (response_lost / r_toks) * 100 if r_toks > 0 else 0
        prompt_lost = max(0, lost_tokens - r_toks)

        print(f"\nExample {idx}: ID={item_id} (doc_id={raw.get('doc_id')})")
        print(f"  Total tokens:        {total_len}")
        print(f"  Token Breakdown:     Passage={p_toks} ({p_toks/total_len*100:.1f}%), "
              f"Instruction={i_toks} ({i_toks/total_len*100:.1f}%), "
              f"Response={r_toks} ({r_toks/total_len*100:.1f}%), "
              f"Overhead={template_overhead}")
        print(f"  Right-truncation to 1024 cuts {lost_tokens} tokens from the end:")
        print(f"    -> Response tokens lost: {response_lost}/{r_toks} ({pct_response_lost:.1f}% of response destroyed!)")
        if prompt_lost > 0:
            print(f"    -> Prompt tokens lost: {prompt_lost} (end of user message also cut!)")
        print(f"  Instruction excerpt: {instruction[:100]}...")
        print(f"  Response:            {response}")

if __name__ == "__main__":
    main()
