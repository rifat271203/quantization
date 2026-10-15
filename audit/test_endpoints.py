# -*- coding: utf-8 -*-
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
PORT = 8097

def test():
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
        # Test 1: Native /completion endpoint with n_probs
        prompt = "<start_of_turn>user\nWhat is 2+2? Answer with number only:<end_of_turn>\n<start_of_turn>model\n"
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=json.dumps({
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": 10,
                "temperature": 0.0
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                print("--- Native /completion result ---")
                print("Content:", res.get("content"))
                print("Completion probs keys:", res.get("completion_probabilities", [{}])[0].keys() if res.get("completion_probabilities") else "None")
                if res.get("completion_probabilities"):
                    probs = res["completion_probabilities"][0].get("probs", [])
                    for p in probs[:5]:
                        print(" ", p)
        except Exception as e:
            print("Native /completion error:", e)

        # Test 2: OpenAI /v1/chat/completions with logprobs
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "What is 2+2? Answer with number only:"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req2, timeout=10) as resp:
                res2 = json.loads(resp.read().decode("utf-8"))
                print("\n--- /v1/chat/completions result ---")
                print("Choice:", res2["choices"][0]["message"])
                top_lp = res2["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                for lp in top_lp[:5]:
                    print(" ", lp)
        except Exception as e:
            print("/v1/chat/completions error:", e)

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test()
