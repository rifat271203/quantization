import sys
import os
import re
import csv
import string
import random
import numpy as np
import pypdfium2 as pdfium
from collections import Counter, defaultdict
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014
allowed_chars_old = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash
allowed_chars_new = allowed_chars_old | {'\u2013', '\u2022', '\u2010', '\u2026'}

DOCS = {
    'agri_mechanization_2020': {'path': 'data/raw_sources/agriculture/en/agri_mechanization_policy_2020_en.pdf', 'domain': 'agriculture', 'unofficial_translation': False},
    'agri_seed_rules_2020': {'path': 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf', 'domain': 'agriculture', 'unofficial_translation': True},
    'disaster_sod_2019': {'path': 'data/raw_sources/disaster/en/sod_2019_en.pdf', 'domain': 'disaster', 'unofficial_translation': False},
    'disaster_displacement_nap_2022': {'path': 'data/raw_sources/disaster/en/nap_displacement_2022_2042_en.pdf', 'domain': 'disaster', 'unofficial_translation': False},
    'disaster_displacement_strat_2021': {'path': 'data/raw_sources/disaster/en/strategy_displacement_2021_en.pdf', 'domain': 'disaster', 'unofficial_translation': False},
    'disaster_emcrp_rpf': {'path': 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf', 'domain': 'disaster', 'unofficial_translation': False},
    'disaster_emcrp_esmf': {'path': 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf', 'domain': 'disaster', 'unofficial_translation': False},
    'edu_nep_2010': {'path': 'data/raw_sources/education/en/national_education_policy_2010_en.pdf', 'domain': 'education', 'unofficial_translation': False},
}

def normalize_text(text):
    # (a) U+FFFE
    res = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch == '\ufffe':
            l_char = text[i-1] if i > 0 else ''
            r_char = text[i+1] if i < n - 1 else ''
            if l_char.isalpha() and r_char.isalpha():
                pass
            else:
                res.append(' ')
        else:
            res.append(ch)
    text = ''.join(res)
    # (b) Bullets
    for b in ['\uf0b7', '\uf0a7', '\uf0d8', '\uf0fc']:
        text = text.replace(b, '\u2022')
    return text

def count_sentences(text):
    cleaned = re.sub(r'\b(e\.g\.|i\.e\.|viz\.|etc\.|al\.|Mr\.|Mrs\.|Dr\.|No\.|Fig\.|Vol\.|pp\.|sec\.)', 
                     lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d+\.\d+\b', lambda m: m.group(0).replace('.', '<DOT>'), cleaned)
    cleaned = re.sub(r'(?<=\s)\d+\.(?=\s)', lambda m: m.group(0).replace('.', '<DOT>'), cleaned)
    sentences = re.split(r'[.!?]+[\'"’”)]?(?:\s+|$)', cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def split_into_sentences(text):
    masked = list(text)
    for m in re.finditer(r'\b(e\.g\.|i\.e\.|viz\.|etc\.|al\.|Mr\.|Mrs\.|Dr\.|No\.|Fig\.|Vol\.|pp\.|sec\.)', text, re.I):
        for idx in range(m.start(), m.end()):
            if masked[idx] == '.':
                masked[idx] = '@'
    for m in re.finditer(r'\b\d+\.\d+\b', text):
        for idx in range(m.start(), m.end()):
            if masked[idx] == '.':
                masked[idx] = '@'
    masked_str = ''.join(masked)
    splits = [m.end() for m in re.finditer(r'[.!?]+[\'"’”)]?(?:\s+|$)', masked_str)]
    if not splits or splits[-1] < len(text):
        splits.append(len(text))
    sentences = []
    prev = 0
    for idx in splits:
        s = text[prev:idx].strip()
        if s:
            sentences.append(s)
        prev = idx
    return sentences

def extract_paragraphs(page_text):
    raw_blocks = [b.strip() for b in re.split(r'\n\s*\n', page_text) if b.strip()]
    paragraphs = []
    for block in raw_blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        cur = [lines[0]]
        for line in lines[1:]:
            prev = cur[-1]
            is_list = bool(re.match(r'^(?:\(?\d+[\.\)]|\(?[a-zA-Z]\)|\d+\.\d+(\.\d+)*|[•▪\-\*]|Chapter\s+\d+|Section\s+\d+)\s+', line))
            prev_end = bool(re.search(r'[\.\?!:]["\'\)]?$', prev))
            starts_upper = bool(re.match(r'^[A-Z0-9"\'•]', line))
            if is_list or (prev_end and starts_upper and len(prev) < 65):
                paragraphs.append((' '.join(cur), list(cur)))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            paragraphs.append((' '.join(cur), list(cur)))
    return paragraphs

def has_footnote_glue(text):
    return bool(re.search(r'[a-zA-Z,]\d{1,2}(?:\s|$)', text))

def compute_flags(c):
    text = c['text']
    lines = c['lines']
    
    # 1. question_marks
    q_marks = text.count('?')
    
    # 2. short_line_fraction (< 6 words)
    if lines:
        short_lines = sum(1 for l in lines if len(l.split()) < 6)
        short_frac = short_lines / len(lines)
    else:
        short_frac = 0.0
        
    # 3. bullet_count
    b_count = text.count('\u2022')
    
    # 4. section_markers (tokens like 6.4.1, 7.0, or "Section N")
    sec_markers = len(re.findall(r'\bSection\s+\d+(?:\.\d+)*\b|\b\d+\.\d+(?:\.\d+)*\b', text, re.IGNORECASE))
    
    # 5. capital_start_fraction
    s_list = split_into_sentences(text)
    if s_list:
        cap_starts = sum(1 for s in s_list if s.lstrip(' "\'“‘(\\[')[:1].isupper())
        cap_frac = cap_starts / len(s_list)
    else:
        cap_frac = 1.0
        
    table_like = (short_frac > 0.4 or q_marks >= 3 or sec_markers >= 4 or cap_frac < 0.7)
    
    c['question_marks'] = q_marks
    c['short_line_fraction'] = short_frac
    c['bullet_count'] = b_count
    c['section_markers'] = sec_markers
    c['capital_start_fraction'] = cap_frac
    c['table_like'] = table_like
    return c

print("Loading Gemma-3-1b-it tokenizer...")
tok = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
print("Tokenizer loaded successfully.")
