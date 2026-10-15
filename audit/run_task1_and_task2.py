import os
import re
import csv
import json
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
import pypdf

ROOT = Path("d:/Tokenizer/quant-hallucination-study")

def run_git(cmd):
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, shell=True)
    return res.stdout.strip()

print("=== GIT STATUS FOR BUILD_STUDY_EVAL_SETS ===")
print("git status of file:", run_git("git status data/raw_sources/build_study_eval_sets.py"))
print("git log:", run_git("git log -n 3 --oneline -- data/raw_sources/build_study_eval_sets.py"))

print("\n=== REPO RECENT COMMITS ===")
print(run_git("git log -n 5 --oneline"))

# Let's check headers / comments in build_study_eval_sets.py
script_path = ROOT / "data/raw_sources/build_study_eval_sets.py"
with open(script_path, "r", encoding="utf-8") as f:
    text = f.read()

# Check literal IDs in build_study_eval_sets.py
# In section 1: add_clean_pair
# Let's extract all call arguments to add_clean_pair
add_clean_pat = re.compile(r"add_clean_pair\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'", re.MULTILINE)
matches = add_clean_pat.findall(text)
print("\n=== ADD_CLEAN_PAIR LITERAL ENTRIES ===")
for m in matches:
    print(m)

# In section 2: act_items
act_pat = re.compile(r"\('([^']+)',\s*(\d+),", re.MULTILINE)
# Find in text
lines = text.splitlines()
print("\n=== DISASTER ACT / NCTB ITEMS IN SCRIPT ===")
for i, l in enumerate(lines, 1):
    if "('disaster_act_" in l or "('edu_sci_" in l or "'id': 'disaster_act_low_" in l:
        print(f"L{i}: {l.strip()}")

