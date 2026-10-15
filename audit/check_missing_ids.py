# -*- coding: utf-8 -*-
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

f16_bn_ids = set()
with open('results/llama32_logprob_predictions.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r['tier'] == 'f16' and r['language'] == 'bangla':
            f16_bn_ids.add(r['pair_id'])

all_ids = []
with open('data_v2/belebele_eval_parallel.jsonl', encoding='utf-8') as f:
    for line in f:
        all_ids.append(json.loads(line)['pair_id'])

missing_ids = [pid for pid in all_ids if pid not in f16_bn_ids]
print(f"Missing in f16 Bangla: {len(missing_ids)} items out of 900")
print("First 10 missing IDs:", missing_ids[:10])
print("Last 10 missing IDs:", missing_ids[-10:])
