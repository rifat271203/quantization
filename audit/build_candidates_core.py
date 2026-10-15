import sys
import os
import re
import string
import pypdfium2 as pdfium
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

DOCS = {
    'agri_mechanization_2020': 'data/raw_sources/agriculture/en/agri_mechanization_policy_2020_en.pdf',
    'agri_seed_rules_2020': 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf',
    'disaster_sod_2019': 'data/raw_sources/disaster/en/sod_2019_en.pdf',
    'disaster_displacement_nap_2022': 'data/raw_sources/disaster/en/nap_displacement_2022_2042_en.pdf',
    'disaster_displacement_strat_2021': 'data/raw_sources/disaster/en/strategy_displacement_2021_en.pdf',
    'disaster_emcrp_rpf': 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf',
    'disaster_emcrp_esmf': 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf',
    'edu_nep_2010': 'data/raw_sources/education/en/national_education_policy_2010_en.pdf',
}

curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014

allowed_chars = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash

def has_footnote_glue(text):
    return bool(re.search(r'\b\d{1,3},\d{4,}\b', text))

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
            starts_upper = bool(re.match(r'^[A-Z0-9"\']', line))
            if is_list or (prev_end and starts_upper and len(prev) < 65):
                paragraphs.append((' '.join(cur), list(cur)))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            paragraphs.append((' '.join(cur), list(cur)))
    return paragraphs

print("Setup completed successfully.")
