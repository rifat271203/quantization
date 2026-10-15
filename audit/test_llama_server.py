# -*- coding: utf-8 -*-
"""Test Llama 3.2 1B GGUF with llama-server for Bangla logprob extraction."""

import subprocess
import time
import urllib.request
import json
from pathlib import Path

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"
MODEL = ROOT / "models" / "gguf" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
PORT = 8099

def main():
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(MODEL),
        "--port", str(PORT),
        "-t", "4",
        "-c", "2048",
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd)
    try:
        url = f"http://127.0.0.1:{PORT}/health"
        for _ in range(60):
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.5)

        # Bangla Belebele style prompt
        test_prompt = (
            "Read the following passage carefully and answer the question based ONLY on facts directly stated in the passage.\n\n"
            "Passage:\n\"\"\"\n"
            "ঢাকা বাংলাদেশের রাজধানী এবং সবচেয়ে জনবহুল শহর।\n"
            "\"\"\"\n\n"
            "বাংলাদেশের রাজধানী কোনটি?\n"
            "1. চট্টগ্রাম\n2. ঢাকা\n3. সিলেট\n4. খুলনা\n\n"
            "Answer with option number only:"
        )
        payload = {
            "messages": [
                {"role": "user", "content": test_prompt},
                {"role": "assistant", "content": "("}
            ],
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            
        print("Bangla prompt response received!")
        top_lps = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        for t in top_lps[:10]:
            print(f"  token={repr(t['token'])}, logprob={t['logprob']:.4f}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
