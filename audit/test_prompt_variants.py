# -*- coding: utf-8 -*-
"""Test different prompt formats to see top 20 tokens."""

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
PORT = 8099

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
        with open(DATA_BN, encoding="utf-8") as f:
            bn_item = json.loads(f.readline())
        with open(DATA_EN, encoding="utf-8") as f:
            en_item = json.loads(f.readline())

        formats = [
            ("Original Ending: 'Correct Answer: '", 
             lambda p, i: f"Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\nPassage:\n\"\"\"\n{p}\n\"\"\"\n\n{i}\n\nCorrect Answer:"),
            ("Ending with 'Correct Answer: ('", 
             lambda p, i: f"Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\nPassage:\n\"\"\"\n{p}\n\"\"\"\n\n{i}\n\nCorrect Answer: ("),
            ("Explicit instruction ending with '(': ", 
             lambda p, i: f"Passage:\n\"\"\"\n{p}\n\"\"\"\n\n{i}\n\nAnswer with option number only (1, 2, 3, or 4): (")
        ]

        for name, fmt in formats:
            print(f"\n=================== {name} ===================")
            # English
            prompt_en = fmt(en_item["passage"], en_item["instruction"])
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps({
                    "messages": [{"role": "user", "content": prompt_en}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": 10
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            top_lp = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            print("EN Top 5 tokens:", [(lp["token"], round(lp["logprob"], 4)) for lp in top_lp[:5]])

            # Bangla
            prompt_bn = fmt(bn_item["passage"], bn_item["instruction"])
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps({
                    "messages": [{"role": "user", "content": prompt_bn}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": 10
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            top_lp = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            print("BN Top 5 tokens:", [(lp["token"], round(lp["logprob"], 4)) for lp in top_lp[:5]])

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
