import sys
import os
import re
import json
import random
import string
import unicodedata
import numpy as np
import pypdfium2 as pdfium
from collections import Counter
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

DOCS = {
    'agri_mechanization_2020': {
        'path': 'data/raw_sources/agriculture/en/agri_mechanization_policy_2020_en.pdf',
        'domain': 'agriculture'
    },
    'agri_seed_rules_2020': {
        'path': 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf',
        'domain': 'agriculture'
    },
    'disaster_sod_2019': {
        'path': 'data/raw_sources/disaster/en/sod_2019_en.pdf',
        'domain': 'disaster'
    },
    'disaster_displacement_nap_2022': {
        'path': 'data/raw_sources/disaster/en/nap_displacement_2022_2042_en.pdf',
        'domain': 'disaster'
    },
    'disaster_displacement_strat_2021': {
        'path': 'data/raw_sources/disaster/en/strategy_displacement_2021_en.pdf',
        'domain': 'disaster'
    },
    'disaster_emcrp_rpf': {
        'path': 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf',
        'domain': 'disaster'
    },
    'disaster_emcrp_esmf': {
        'path': 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf',
        'domain': 'disaster'
    },
    'edu_nep_2010': {
        'path': 'data/raw_sources/education/en/national_education_policy_2010_en.pdf',
        'domain': 'education'
    },
}

# =========================================================================
# STEP 1: CHARACTER EXTRACTION AND UNUSUAL CHARACTER AUDIT
# =========================================================================
curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014

# "characters outside basic Latin, digits, whitespace and common punctuation (including curly quotes and the em dash)"
allowed_chars = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash

print("=" * 80)
print("TASK 1: ENGLISH TEXT EXTRACTION AND CHARACTER AUDIT")
print("=" * 80)

step1_results = {}

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    num_pages = len(pdf)
    total_words = 0
    total_chars = 0
    unusual_counts = Counter()
    
    for page in pdf:
        text = page.get_textpage().get_text_range()
        total_chars += len(text)
        total_words += len(text.split())
        for ch in text:
            if ch not in allowed_chars:
                unusual_counts[ch] += 1
                
    unusual_total = sum(unusual_counts.values())
    unusual_frac = unusual_total / total_chars if total_chars > 0 else 0.0
    top10_unusual = unusual_counts.most_common(10)
    
    step1_results[doc_id] = {
        'pages': num_pages,
        'words': total_words,
        'chars': total_chars,
        'unusual_total': unusual_total,
        'unusual_frac': unusual_frac,
        'top10': top10_unusual
    }
    
    print(f"\nDoc ID: {doc_id}")
    print(f"  Pages: {num_pages}")
    print(f"  Total Words: {total_words:,}")
    print(f"  Total Characters: {total_chars:,}")
    print(f"  Unusual Characters: {unusual_total:,} (Fraction: {unusual_frac:.6f} / {unusual_frac*100:.4f}%)")
    print(f"  Top 10 Unusual Characters:")
    for ch, cnt in top10_unusual:
        cp = ord(ch)
        try:
            name = unicodedata.name(ch)
        except Exception:
            name = "UNKNOWN"
        print(f"    - {ascii(ch)} (U+{cp:04X}, {name}): count={cnt}")

# Collect all unusual characters across all 8 docs to flag in step 2 rule 8
all_unusual_characters = set()
for r in step1_results.values():
    for ch, _ in r['top10']:
        all_unusual_characters.add(ch)

print("\nStep 1 complete.")
