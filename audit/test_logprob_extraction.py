# -*- coding: utf-8 -*-
"""Test logprob extraction on Gemma 3 1B IT."""

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
PORT = 8096

def main():
    cmd = [str(LLAMA_SERVER), "-m", str(MODEL), "--port", str(PORT), "-t", "4", "-c", "2048", "--log-disable"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Starting test server...")
    
    # Wait for ready
    time.sleep(2)
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    print("Server ready!")
                    break
        except Exception:
            time.sleep(0.5)

    try:
        # Load sample English item
        with open(DATA_EN, encoding="utf-8") as f:
            en_item = json.loads(f.readline())
            
        en_prompt = (
            "Read the following passage carefully and answer the question based ONLY on facts directly stated "
            "in the passage.\n\n"
            f"Passage:\n\"\"\"\n{en_item['passage'].strip()}\n\"\"\"\n\n"
            f"{en_item['instruction'].strip()}\n\n"
            "Answer with option number only (1, 2, 3, or 4): ("
        )
        
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": en_prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            
        print("\n--- English Item Result ---")
        print("Gold option:", en_item["correct_option"])
        top_lp = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        for lp in top_lp[:10]:
            print(f"  Token: {repr(lp['token']):<10} | Logprob: {lp['logprob']:.4f}")
            
    finally:
        print("\nStopping server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
