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
    "--port", "8097",
    "-t", "4",
    "-c", "2048",
    "--log-disable"
]
proc = subprocess.Popen(server_cmd)
try:
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8097/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    for pid in ['bel_0001', 'bel_0002', 'bel_0003']:
        with open('data_v2/belebele_eval_parallel.jsonl', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item['pair_id'] == pid:
                    break

        sub = item['bn']
        instr = '\n'.join([sub['question']] + [f"{k}. {sub['options'][k]}" for k in ['1','2','3','4']])
        prompt = f"Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\nPassage:\n\"\"\"\n{sub['passage']}\n\"\"\"\n\n{instr}\n\nAnswer with option number only:"

        # Test both chat and completion
        # Chat test:
        chat_payload = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "("}
            ],
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20
        }
        req = urllib.request.Request(
            "http://127.0.0.1:8097/v1/chat/completions",
            data=json.dumps(chat_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            chat_res = json.loads(resp.read().decode("utf-8"))
        
        c0 = chat_res["choices"][0]
        has_logprobs = "logprobs" in c0 and c0["logprobs"] is not None and "content" in c0["logprobs"]
        print(f"Chat {pid}: has_logprobs={has_logprobs}, finish_reason={c0.get('finish_reason')}, msg_content={repr(c0.get('message', {}).get('content'))}")

        # Completion test:
        compl_payload = {
            "prompt": f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n(",
            "temperature": 0.0,
            "n_predict": 1,
            "logprobs": True,
            "top_logprobs": 20
        }
        req2 = urllib.request.Request(
            "http://127.0.0.1:8097/completion",
            data=json.dumps(compl_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req2, timeout=30) as resp:
            compl_res = json.loads(resp.read().decode("utf-8"))
        
        probs = compl_res.get("completion_probabilities", [])
        has_c_probs = len(probs) > 0 and "top_logprobs" in probs[0]
        top_tokens = [t["token"] for t in probs[0]["top_logprobs"][:3]] if has_c_probs else []
        print(f"Completion {pid}: has_probs={has_c_probs}, top_tokens={top_tokens}")

finally:
    proc.terminate()
    proc.wait()
