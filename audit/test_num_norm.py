import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pypdf

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path("d:/Tokenizer/quant-hallucination-study")

# Bengali digit normalization map
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def norm_digits(s):
    return s.translate(BN_DIGITS)

def extract_numbers(s):
    # Normalize bengali digits first
    s_norm = norm_digits(s)
    # Extract numbers: integers, decimals, comma-separated numbers
    # Ignore footnote glued like 727,0001 or standalone punctuation
    # Regex to find numbers
    nums = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', s_norm)
    # Clean commas inside numbers
    cleaned = []
    for n in nums:
        cleaned.append(n.replace(',', ''))
    return sorted(cleaned)

print("Testing digit normalization and number extraction:")
sample_bn = "ধারা ৩১ এ প্রদত্ত ক্ষমতাবলে সরকার ২,০২০ সনের ৭২৭,০০১"
print("BN sample:", ascii(sample_bn))
print("Extracted:", extract_numbers(sample_bn))
