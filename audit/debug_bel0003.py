# -*- coding: utf-8 -*-
import json
import subprocess
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open('data_v2/belebele_eval_parallel.jsonl', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item['pair_id'] == 'bel_0003':
            break

sub = item['bn']
instr = '\n'.join([sub['question']] + [f"{k}. {sub['options'][k]}" for k in ['1','2','3','4']])
prompt = f"Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\nPassage:\n\"\"\"\n{sub['passage']}\n\"\"\"\n\n{instr}\n\nAnswer with option number only:"

with open('audit/temp_bel0003.txt', 'w', encoding='utf-8') as f:
    f.write(prompt)

cmd = [
    "llama.cpp/llama-tokenize.exe",
    "-m", "models/gguf/Llama-3.2-1B-Instruct-F16.gguf",
    "-f", "audit/temp_bel0003.txt",
    "--show-count"
]
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out = res.stdout.decode("utf-8", errors="ignore")
m = re.search(r"Total number of tokens:\s*(\d+)", out)
if m:
    print("Total tokens in bel_0003 prompt:", m.group(1))
