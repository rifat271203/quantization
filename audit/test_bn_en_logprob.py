# -*- coding: utf-8 -*-
"""Test prefilled assistant message on Bangla and English items."""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"
MODEL = ROOT / "models" / "gguf" / "gemma-3-1b-it-Q4_K_M.gguf"
DATA_BN = ROOT / "data_v2" / "belebele_eval_bn_300.jsonl"
DATA_EN = ROOT / "data_v2" / "belebele_eval_en_300.jsonl"
PORT = 8098

def main():
    cmd = [str(LLAMA_SERVER), "-m", str(MODEL), "--port", str(PORT), "-t", "4", "-c", "2048", "--log-disable"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    try:
        with open(DATA_EN, encoding="utf-8") as f:
            en_items = [json.loads(line) for idx, line in enumerate(f) if idx < 3]

        candidate_tokens = {"1", "2", "3", "4"}

        print("=== ENGLISH ITEMS WITH PREFILLED ASSISTANT '(' ===")
        for idx, item in enumerate(en_items, 1):
            prompt = (
                "Read the following passage carefully and answer the question based ONLY on facts directly stated "
                "in the passage.\n\n"
                f"Passage:\n\"\"\"\n{item['passage'].strip()}\n\"\"\"\n\n"
                f"{item['instruction'].strip()}\n\n"
                "Answer with option number only:"
            )
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "("}
                    ],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": 20
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            
            top_lp = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            token_lps = {}
            for entry in top_lp:
                t = entry["token"].strip()
                if t in candidate_tokens and t not in token_lps:
                    token_lps[t] = entry["logprob"]
            
            gold = str(item["correct_option"])
            best_token = max(token_lps.items(), key=lambda x: x[1])[0] if token_lps else None
            print(f"EN Item {idx} (ID {item['id']}): Gold={gold}, Best={best_token}, Cand_LPs={token_lps}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
