import sys
import os
import re
import string
import pypdfium2 as pdfium
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

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

def normalize_text_v2(text):
    # 1. U+FFFE -> plain hyphen '-'
    text = text.replace('\ufffe', '-')
    # 2. Bullets
    for b in ['\uf0b7', '\uf0a7', '\uf0d8', '\uf0fc']:
        text = text.replace(b, '\u2022')
    return text

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

# Clause marker stripping regex
CLAUSE_STRIP_REGEX = re.compile(
    r'^(?:[•▪■\*\-–—]\s*)*(?:(?:\d+(?:\.\d+)*\.?|\([0-9a-zA-ZivxIVX]+\)|[0-9a-zA-ZivxIVX]+\))\s*)*(?:[•▪■\*\-–—]\s*)*'
)

def check_capital_or_digit_start(s):
    s_clean = s.lstrip()
    stripped = CLAUSE_STRIP_REGEX.sub('', s_clean).lstrip(' "\'“‘(\\[')
    if not stripped:
        return True
    c = stripped[0]
    return c.isupper() or c.isdigit()

def count_section_markers_v2(c):
    # Section markers at start of a line or sentence
    # Pattern: \b\d+\.\d+(\.\d+)*\b or \bSection\s+\d+(\.\d+)*\b
    # Check start of each line
    cnt = 0
    for l in c['lines']:
        l_s = l.strip().lstrip('•▪■*-–— ').strip()
        if re.match(r'^(?:Section\s+\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*)\b', l_s, re.I):
            cnt += 1
    # Also check sentences that might not be at start of a line
    s_list = split_into_sentences(c['text'])
    for s in s_list:
        s_s = s.strip().lstrip('•▪■*-–— ').strip()
        # if this sentence does not start at line start, check it
        # Actually counting across sentences:
        pass
    return cnt

def count_section_markers_in_text(text):
    # Matches at start of text, start of line, or right after sentence end
    pattern = r'(?:(?<=\n)|(?<=\A)|(?<=[.!?]\s))(?:[•▪■\*\-–—]\s*)?(?:Section\s+\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*)\b'
    return len(re.findall(pattern, text, re.I))

print("Testing U+FFFE replacement with '-' and checking sample contexts...")
for doc_id in ['agri_seed_rules_2020', 'edu_nep_2010']:
    pdf = pdfium.PdfDocument(DOCS[doc_id]['path'])
    found = []
    for p_no, page in enumerate(pdf):
        raw_t = page.get_textpage().get_text_range()
        for i, ch in enumerate(raw_t):
            if ch == '\ufffe':
                norm_t = normalize_text_v2(raw_t)
                c_before = norm_t[max(0, i-15):i]
                c_after = norm_t[i+1:min(len(norm_t), i+16)]
                found.append((p_no+1, repr(norm_t[max(0, i-15):min(len(norm_t), i+16)])))
    print(f"{doc_id} contexts with '-': {len(found)}")
    for p, snip in found[:5]:
        print(f"  p.{p}: {snip}")
