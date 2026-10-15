import csv
import json
import re
import sys
import random
from pathlib import Path
from collections import Counter
import numpy as np
import pypdf

# Ensure stdout encodes to utf-8 safely
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path("d:/Tokenizer/quant-hallucination-study")

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def norm_digits(s):
    return s.translate(BN_DIGITS)

def extract_numbers(s):
    s_norm = norm_digits(s)
    # Find all numeric patterns (integers, decimals)
    # Also handle comma-separated numbers like 1,000 or 1,00,000
    # Flag footnote-glued digits separately: e.g. 727,0001 (where footnote digit glued to number)
    raw_nums = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', s_norm)
    cleaned = []
    glued_flags = []
    for n in raw_nums:
        # Check if glued footnote like 727,0001 (invalid thousand grouping)
        if re.search(r',\d{4,}', n):
            glued_flags.append(n)
        # remove commas
        val = n.replace(',', '')
        cleaned.append(val)
    return sorted(cleaned), glued_flags

# 1. Load BN pages from data_v2/pages_bn.jsonl for the 3 OCR docs
pages_bn_all = {}
if (ROOT / "data_v2/pages_bn.jsonl").exists():
    with open(ROOT / "data_v2/pages_bn.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                d = row['doc_id']
                p = int(row['page'])
                if d not in pages_bn_all:
                    pages_bn_all[d] = {}
                pages_bn_all[d][p] = row['text']

# Also load clean text layers for other docs if needed
# We know the 8 docs:
DOC_PDFS = {
    'agri_mechanization_2020': ('data/raw_sources/agriculture/bn/agri_mechanization_policy_2020_bn.pdf', 'data/raw_sources/agriculture/en/agri_mechanization_policy_2020_en.pdf'),
    'agri_seed_rules_2020': ('data/raw_sources/agriculture/bn/seed_rules_2020_bn.pdf', 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf'),
    'disaster_sod_2019': ('data/raw_sources/disaster/bn/sod_2019_bn.pdf', 'data/raw_sources/disaster/en/sod_2019_en.pdf'),
    'disaster_displacement_nap_2022': ('data/raw_sources/disaster/bn/nap_displacement_2022_2042_bn.pdf', 'data/raw_sources/disaster/en/nap_displacement_2022_2042_en.pdf'),
    'disaster_displacement_strat_2021': ('data/raw_sources/disaster/bn/strategy_displacement_2021_bn.pdf', 'data/raw_sources/disaster/en/strategy_displacement_2021_en.pdf'),
    'disaster_emcrp_rpf': ('data/raw_sources/disaster/bn/rpf_emcrp_bn.pdf', 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf'),
    'disaster_emcrp_esmf': ('data/raw_sources/disaster/bn/esmf_emcrp_bn.pdf', 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf'),
    'edu_nep_2010': ('data/raw_sources/education/bn/national_education_policy_2010_bn.pdf', 'data/raw_sources/education/en/national_education_policy_2010_en.pdf'),
}

pages_en_all = {}
for d, (bn_p, en_p) in DOC_PDFS.items():
    en_full = ROOT / en_p
    if en_full.exists():
        reader = pypdf.PdfReader(str(en_full))
        pages_en_all[d] = {i+1: page.extract_text() or "" for i, page in enumerate(reader.pages)}

# Load anchors.csv
with open(ROOT / "data_v2/anchors.csv", "r", encoding="utf-8") as f:
    anchors_raw = list(csv.DictReader(f))

print(f"Total rows in anchors.csv: {len(anchors_raw)}")
# Check which docs are in anchors.csv
anchors_by_doc = Counter(r['doc_id'] for r in anchors_raw)
for d, cnt in anchors_by_doc.items():
    print(f"  {d}: {cnt} anchors")

# Group anchors by doc_id
docs_anchors = {}
for r in anchors_raw:
    d = r['doc_id']
    if d not in docs_anchors:
        docs_anchors[d] = []
    docs_anchors[d].append(r)

# Process each document
# A matched anchor is one where both bn_page and en_page are non-empty
anchored_sections = []

for d, rows in docs_anchors.items():
    # Keep only matched rows
    matched_rows = [r for r in rows if r['bn_page'] and r['en_page']]
    
    # Sort or keep document order
    # To determine level:
    # Anchor string: e.g. "1" -> level 1, "1.1" -> level 2, "1.1.1" -> level 3
    # If integer like "1", "2" -> level 1
    def get_level(anc_str):
        parts = anc_str.strip().split('.')
        # Filter empty parts
        parts = [p for p in parts if p]
        return len(parts)

    for i, r in enumerate(matched_rows):
        anc = r['anchor']
        lvl = get_level(anc)
        bn_p = int(float(r['bn_page']))
        en_p = int(float(r['en_page']))
        
        # Find next anchor of the same level
        next_r = None
        for j in range(i + 1, len(matched_rows)):
            if get_level(matched_rows[j]['anchor']) == lvl:
                next_r = matched_rows[j]
                break
        
        # BN text extraction:
        # From anchor line on bn_p to next anchor of same level
        # Concatenate pages from bn_p to next_bn_p
        if next_r:
            next_bn_p = int(float(next_r['bn_page']))
            next_en_p = int(float(next_r['en_page']))
        else:
            # Till end of section or doc
            next_bn_p = min(bn_p + 5, max(pages_bn_all.get(d, {bn_p: ""}).keys()))
            next_en_p = min(en_p + 5, max(pages_en_all.get(d, {en_p: ""}).keys()))
        
        # If next page is before current page (out of order in table of contents/annexures), clamp
        if next_bn_p < bn_p:
            next_bn_p = bn_p
        if next_en_p < en_p:
            next_en_p = en_p
            
        bn_pages_span = list(range(bn_p, next_bn_p + 1))
        en_pages_span = list(range(en_p, next_en_p + 1))
        
        bn_combined = "\n".join(pages_bn_all.get(d, {}).get(p, "") for p in bn_pages_span)
        en_combined = "\n".join(pages_en_all.get(d, {}).get(p, "") for p in en_pages_span)
        
        # Locate anchor line in BN
        # Use bn_first_80_chars or regex on anchor
        bn_start_idx = 0
        bn_target = r['bn_first_80_chars'].strip()[:30] if r['bn_first_80_chars'] else ""
        if bn_target and bn_target in bn_combined:
            bn_start_idx = bn_combined.find(bn_target)
        else:
            # fallback regex
            anc_bn = anc.translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
            m = re.search(re.escape(anc_bn) + r'[\।\.\s]', bn_combined)
            if m:
                bn_start_idx = m.start()
                
        # Locate next anchor in BN
        bn_end_idx = len(bn_combined)
        if next_r:
            next_bn_target = next_r['bn_first_80_chars'].strip()[:30] if next_r['bn_first_80_chars'] else ""
            if next_bn_target and next_bn_target in bn_combined[bn_start_idx + 10:]:
                bn_end_idx = bn_start_idx + 10 + bn_combined[bn_start_idx + 10:].find(next_bn_target)
            else:
                next_anc_bn = next_r['anchor'].translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
                m = re.search(re.escape(next_anc_bn) + r'[\।\.\s]', bn_combined[bn_start_idx + 10:])
                if m:
                    bn_end_idx = bn_start_idx + 10 + m.start()
                    
        bn_text = bn_combined[bn_start_idx:bn_end_idx].strip()
        
        # Locate anchor line in EN
        en_start_idx = 0
        en_target = r['en_first_80_chars'].strip()[:30] if r['en_first_80_chars'] else ""
        if en_target and en_target in en_combined:
            en_start_idx = en_combined.find(en_target)
        else:
            m = re.search(re.escape(anc) + r'[\.\s]', en_combined)
            if m:
                en_start_idx = m.start()
                
        # Locate next anchor in EN
        en_end_idx = len(en_combined)
        if next_r:
            next_en_target = next_r['en_first_80_chars'].strip()[:30] if next_r['en_first_80_chars'] else ""
            if next_en_target and next_en_target in en_combined[en_start_idx + 10:]:
                en_end_idx = en_start_idx + 10 + en_combined[en_start_idx + 10:].find(next_en_target)
            else:
                m = re.search(re.escape(next_r['anchor']) + r'[\.\s]', en_combined[en_start_idx + 10:])
                if m:
                    en_end_idx = en_start_idx + 10 + m.start()
                    
        en_text = en_combined[en_start_idx:en_end_idx].strip()
        
        bn_words = len(bn_text.split())
        en_words = len(en_text.split())
        
        nums_bn, glued_bn = extract_numbers(bn_text)
        nums_en, glued_en = extract_numbers(en_text)
        
        # compare multisets
        numbers_equal = (nums_bn == nums_en)
        length_ratio = round(bn_words / en_words, 4) if en_words > 0 else 0.0
        
        item = {
            'doc_id': d,
            'anchor': anc,
            'bn_text': bn_text,
            'en_text': en_text,
            'bn_pages': [bn_p, next_bn_p],
            'en_pages': [en_p, next_en_p],
            'bn_words': bn_words,
            'en_words': en_words,
            'numbers_bn': nums_bn,
            'numbers_en': nums_en,
            'glued_bn': glued_bn,
            'glued_en': glued_en,
            'numbers_equal': numbers_equal,
            'length_ratio': length_ratio,
        }
        anchored_sections.append(item)

# Write to data_v2/anchored_sections.jsonl
out_anchored = ROOT / "data_v2/anchored_sections.jsonl"
with open(out_anchored, "w", encoding="utf-8") as f:
    for item in anchored_sections:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nWrote {len(anchored_sections)} sections to {out_anchored}")

# Report stats for Task 3
print("\n" + "="*80)
print("TASK 3 REPORT: ANCHORED SECTIONS PER DOC_ID")
print("="*80)

for d in docs_anchors.keys():
    sec_d = [s for s in anchored_sections if s['doc_id'] == d]
    n_sec = len(sec_d)
    n_eq = sum(1 for s in sec_d if s['numbers_equal'])
    n_no_num = sum(1 for s in sec_d if len(s['numbers_bn']) == 0 and len(s['numbers_en']) == 0)
    n_diff_num = sum(1 for s in sec_d if not s['numbers_equal'])
    
    ratios = [s['length_ratio'] for s in sec_d if s['length_ratio'] > 0]
    p5 = np.percentile(ratios, 5) if ratios else 0.0
    med = np.median(ratios) if ratios else 0.0
    p95 = np.percentile(ratios, 95) if ratios else 0.0
    
    print(f"\nDocument: {d}")
    print(f"  Total anchored sections: {n_sec}")
    print(f"  numbers_equal is True   : {n_eq}")
    print(f"  No numbers in either    : {n_no_num}")
    print(f"  Different numbers       : {n_diff_num}")
    print(f"  length_ratio distribution (p5, median, p95): {p5:.3f}, {med:.3f}, {p95:.3f}")

# Task 4: Candidate Passages Filtering
print("\n" + "="*80)
print("TASK 4 REPORT: CANDIDATE PASSAGES FILTERING SURVIVORS")
print("="*80)

survivors = []
for d in docs_anchors.keys():
    sec_d = [s for s in anchored_sections if s['doc_id'] == d]
    initial_cnt = len(sec_d)
    
    # Condition 1: 60 <= bn_words <= 250
    c1 = [s for s in sec_d if 60 <= s['bn_words'] <= 250]
    
    # Condition 2: 60 <= en_words <= 250
    c2 = [s for s in c1 if 60 <= s['en_words'] <= 250]
    
    # Condition 3: length_ratio between 0.6 and 1.6
    c3 = [s for s in c2 if 0.6 <= s['length_ratio'] <= 1.6]
    
    # Condition 4: numbers_equal is True OR no numbers in either
    c4 = [s for s in c3 if s['numbers_equal'] or (len(s['numbers_bn']) == 0 and len(s['numbers_en']) == 0)]
    
    print(f"\nDocument: {d}")
    print(f"  Initial matched sections  : {initial_cnt}")
    print(f"  Survive 60 <= bn_words <= 250: {len(c1)}")
    print(f"  Survive 60 <= en_words <= 250: {len(c2)}")
    print(f"  Survive 0.6 <= length_ratio <= 1.6: {len(c3)}")
    print(f"  Survive numbers_equal OR no numbers: {len(c4)}")
    print(f"  Final survivor count      : {len(c4)}")
    
    for idx, s in enumerate(c4):
        item = dict(s)
        item['candidate_id'] = f"{d}_cand_{idx+1:03d}"
        survivors.append(item)

out_cand = ROOT / "data_v2/candidate_passages.jsonl"
with open(out_cand, "w", encoding="utf-8") as f:
    for s in survivors:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nWrote {len(survivors)} survivor passages to {out_cand}")

# Task 5: Gemma Token Check
print("\n" + "="*80)
print("TASK 5 REPORT: GEMMA TOKEN CHECK ON SURVIVORS")
print("="*80)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

for d in docs_anchors.keys():
    surv_d = [s for s in survivors if s['doc_id'] == d]
    if not surv_d:
        print(f"\nDocument: {d} - No survivors")
        continue
    
    bn_tok_lens = [len(tokenizer.encode(s['bn_text'])) for s in surv_d]
    en_tok_lens = [len(tokenizer.encode(s['en_text'])) for s in surv_d]
    ratios = [b / e for b, e in zip(bn_tok_lens, en_tok_lens)]
    
    print(f"\nDocument: {d} (N={len(surv_d)})")
    print(f"  BN Gemma tokens - Mean: {np.mean(bn_tok_lens):.1f}, p90: {np.percentile(bn_tok_lens, 90):.1f}")
    print(f"  EN Gemma tokens - Mean: {np.mean(en_tok_lens):.1f}, p90: {np.percentile(en_tok_lens, 90):.1f}")
    print(f"  BN/EN token ratio - Mean: {np.mean(ratios):.3f}, Median: {np.median(ratios):.3f}")

# Task 6: Random Sampling
print("\n" + "="*80)
print("TASK 6 REPORT: 5 RANDOMLY CHOSEN SURVIVORS PER DOC_ID (SEED 42)")
print("="*80)

random.seed(42)
for d in docs_anchors.keys():
    surv_d = [s for s in survivors if s['doc_id'] == d]
    sample_size = min(5, len(surv_d))
    sample = random.sample(surv_d, sample_size)
    print(f"\nDOCUMENT: {d} (Sample of {sample_size})")
    for idx, s in enumerate(sample):
        print(f"\n--- [{idx+1}] Candidate ID: {s['candidate_id']} | Anchor: {s['anchor']} | BN words: {s['bn_words']} | EN words: {s['en_words']} | Ratio: {s['length_ratio']} ---")
        print("  BN TEXT (ascii-escaped):")
        print(f"    {ascii(s['bn_text'][:300])}...")
        print("  EN TEXT (ascii-escaped):")
        print(f"    {ascii(s['en_text'][:300])}...")

