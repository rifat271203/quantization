# -*- coding: utf-8 -*-
import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open('data_v2/belebele_eval_parallel.jsonl', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item['pair_id'] == 'bel_0010':
            break

sub = item['bn']
instr = '\n'.join([sub['question']] + [f"{k}. {sub['options'][k]}" for k in ['1','2','3','4']])
prompt = f"Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\nPassage:\n\"\"\"\n{sub['passage']}\n\"\"\"\n\n{instr}\n\nAnswer with option number only:"

p = {
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0,
    "max_tokens": 1,
    "logprobs": True,
    "top_logprobs": 20
}
req = urllib.request.Request(
    'http://127.0.0.1:8096/v1/chat/completions',
    data=json.dumps(p).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    print("Response choice:", json.dumps(res['choices'][0], indent=2, ensure_ascii=False))
except Exception as e:
    print("Error on bel_0010:", e)
