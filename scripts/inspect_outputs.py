import json
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def check_file(filename):
    print("=" * 60)
    print("Inspecting:", filename)
    print("=" * 60)
    with open(filename, encoding="utf-8") as f:
        for idx in range(3):
            line = f.readline()
            if not line: break
            obj = json.loads(line)
            print(f"ID: {obj['id']} | Lang: {obj['language']} | Pred: {obj['pred_option']} | Gold: {obj['gold_option']}")
            print("Raw text:", repr(obj['raw_response'][:100]))

check_file("results/pred_f16.jsonl")
check_file("results/pred_Q4_K_M.jsonl")
check_file("results/pred_Q2_K.jsonl")
