# -*- coding: utf-8 -*-
import subprocess
import time
import json
import urllib.request
import sys

sys.stdout.reconfigure(encoding="utf-8")

server_cmd = [
    "llama.cpp/llama-server.exe",
    "-m", "models/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "--port", "8098",
    "-t", "4",
    "-c", "2048",
    "--log-disable"
]
proc = subprocess.Popen(server_cmd)
try:
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8098/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    with open('data_v2/belebele_eval_parallel.jsonl', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['pair_id'] == 'bel_0003':
                break

    sub = item['bn']
    instr = '\n'.join([sub['question']] + [f"{k}. {sub['options'][k]}" for k in ['1','2','3','4']])
    prompt = f"Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\nPassage:\n\"\"\"\n{sub['passage']}\n\"\"\"\n\n{instr}\n\nAnswer with option number only:"

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8098/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    
    c0 = res["choices"][0]
    print("Without assistant prefill on bel_0003:")
    print("Content:", repr(c0["message"]["content"]))
    print("Has logprobs:", "logprobs" in c0 and c0["logprobs"] is not None)
    if "logprobs" in c0 and c0["logprobs"]:
        top_lps = c0["logprobs"]["content"][0]["top_logprobs"]
        for t in top_lps[:5]:
            print("  token:", repr(t["token"]), "lp:", t["logprob"])

finally:
    proc.terminate()
    proc.wait()
