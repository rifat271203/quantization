import sys
import os
import re
import json
import random
import string
import unicodedata
from collections import Counter, defaultdict
import pypdfium2 as pdfium

sys.stdout.reconfigure(encoding="utf-8")

DOCS = {
    'agri_mechanization_2020': {
        'path': 'data/raw_sources/agriculture/en/agri_mechanization_policy_2020_en.pdf',
        'domain': 'agriculture',
        'unofficial_translation': False
    },
    'agri_seed_rules_2020': {
        'path': 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf',
        'domain': 'agriculture',
        'unofficial_translation': True
    },
    'disaster_sod_2019': {
        'path': 'data/raw_sources/disaster/en/sod_2019_en.pdf',
        'domain': 'disaster',
        'unofficial_translation': False
    },
    'disaster_displacement_nap_2022': {
        'path': 'data/raw_sources/disaster/en/nap_displacement_2022_2042_en.pdf',
        'domain': 'disaster',
        'unofficial_translation': False
    },
    'disaster_displacement_strat_2021': {
        'path': 'data/raw_sources/disaster/en/strategy_displacement_2021_en.pdf',
        'domain': 'disaster',
        'unofficial_translation': False
    },
    'disaster_emcrp_rpf': {
        'path': 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf',
        'domain': 'disaster',
        'unofficial_translation': False
    },
    'disaster_emcrp_esmf': {
        'path': 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf',
        'domain': 'disaster',
        'unofficial_translation': False
    },
    'edu_nep_2010': {
        'path': 'data/raw_sources/education/en/national_education_policy_2010_en.pdf',
        'domain': 'education',
        'unofficial_translation': False
    },
}

# =========================================================================
# 2. NORMALISATION TEST
# =========================================================================
print("=" * 80)
print("TESTING NORMALISATION OF U+FFFE, SYMBOL BULLETS, AND EXTENDED CHARACTERS")
print("=" * 80)

# (a) Collect contexts of U+FFFE per doc_id
random.seed(42)
all_fffe_contexts = {}
fffe_replacements_sample = []

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    doc_contexts = []
    for p_no, page in enumerate(pdf):
        text = page.get_textpage().get_text_range()
        for i, ch in enumerate(text):
            if ch == '\ufffe':
                c_before = text[max(0, i-20):i]
                c_after = text[i+1:min(len(text), i+21)]
                doc_contexts.append({
                    'doc_id': doc_id,
                    'page': p_no + 1,
                    'index': i,
                    'before': c_before,
                    'after': c_after,
                    'raw_snippet': text[max(0, i-20):min(len(text), i+21)]
                })
    all_fffe_contexts[doc_id] = doc_contexts
    print(f"{doc_id}: total U+FFFE found = {len(doc_contexts)}")

print("\nSampling 20 contexts of U+FFFE per doc_id (or all available if < 20):")
sampled_fffe_contexts = []
for doc_id, ctxs in all_fffe_contexts.items():
    take_k = min(20, len(ctxs))
    if take_k > 0:
        sampled = random.sample(ctxs, take_k)
        sampled_fffe_contexts.extend(sampled)
        print(f"\n--- {doc_id} ({len(sampled)} contexts) ---")
        for idx, sc in enumerate(sampled[:3], 1): # print first 3 as preview
            print(f"  [{idx}] Page {sc['page']}: {repr(sc['before'])} + U+FFFE + {repr(sc['after'])}")

print(f"\nTotal U+FFFE contexts sampled: {len(sampled_fffe_contexts)}")
