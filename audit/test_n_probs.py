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
    "-c", "4096",
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

    compl_payload = {
        "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n(",
        "temperature": 0.0,
        "n_predict": 1,
        "n_probs": 20
    }
    req2 = urllib.request.Request(
        "http://127.0.0.1:8098/completion",
        data=json.dumps(compl_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req2, timeout=30) as resp:
            compl_res = json.loads(resp.read().decode("utf-8"))
        print("compl_res keys:", list(compl_res.keys()))
        print("compl_res:", json.dumps(compl_res, indent=2, ensure_ascii=False))
    except urllib.error.HTTPError as e:
        print("HTTP ERROR:", e.code, e.read().decode('utf-8'))

finally:
    proc.terminate()
    proc.wait()
