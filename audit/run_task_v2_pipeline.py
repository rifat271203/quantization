import sys
import os
import re
import csv
import json
import string
import random
import numpy as np
import pypdfium2 as pdfium
from collections import Counter, defaultdict
from transformers import AutoTokenizer

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
# TASK 1: U+FFFE FIX & 30 RESULTING CONTEXTS
# =========================================================================
print("=" * 90)
print("TASK 1: U+FFFE NORMALISATION FIX (REPLACE WITH '-') & 30 RESULTING CONTEXTS")
print("=" * 90)

def normalize_text_v2(text):
    # Replace U+FFFE with plain hyphen "-"
    text = text.replace('\ufffe', '-')
    # Wingdings bullets U+F0B7, U+F0A7, U+F0D8, U+F0FC -> U+2022
    for b in ['\uf0b7', '\uf0a7', '\uf0d8', '\uf0fc']:
        text = text.replace(b, '\u2022')
    return text

# Collect all occurrences and extract contexts
doc_hyphen_contexts = {}
random.seed(42)

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    contexts = []
    for p_no, page in enumerate(pdf):
        raw_text = page.get_textpage().get_text_range()
        for i, ch in enumerate(raw_text):
            if ch == '\ufffe':
                norm_page = normalize_text_v2(raw_text)
                c_before = norm_page[max(0, i-15):i]
                c_after = norm_page[i+1:min(len(norm_page), i+16)]
                snippet = norm_page[max(0, i-15):min(len(norm_page), i+16)]
                contexts.append({
                    'doc_id': doc_id,
                    'page': p_no + 1,
                    'before': c_before,
                    'after': c_after,
                    'snippet': snippet
                })
    doc_hyphen_contexts[doc_id] = contexts

# Selection of 30 contexts: 5 from agri_seed_rules_2020, 5 from edu_nep_2010, spread across docs
sampled_30_contexts = []

# Exact allocation across docs:
# agri_seed_rules_2020: 5
# edu_nep_2010: 5
# agri_mechanization_2020: 1 (only 1 available)
# disaster_sod_2019: 4
# disaster_displacement_nap_2022: 4
# disaster_displacement_strat_2021: 4
# disaster_emcrp_rpf: 3
# disaster_emcrp_esmf: 4
# Total: 5 + 5 + 1 + 4 + 4 + 4 + 3 + 4 = 30
allocations = {
    'agri_seed_rules_2020': 5,
    'edu_nep_2010': 5,
    'agri_mechanization_2020': 1,
    'disaster_sod_2019': 4,
    'disaster_displacement_nap_2022': 4,
    'disaster_displacement_strat_2021': 4,
    'disaster_emcrp_rpf': 3,
    'disaster_emcrp_esmf': 4
}

for doc_id, k in allocations.items():
    ctxs = doc_hyphen_contexts[doc_id]
    sampled = random.sample(ctxs, min(k, len(ctxs)))
    sampled_30_contexts.extend(sampled)

print(f"Total resulting contexts printed: {len(sampled_30_contexts)}")
for idx, c in enumerate(sampled_30_contexts, 1):
    print(f"[{idx:2d}] {c['doc_id']:<32} p.{c['page']:<3} Context: {repr(c['snippet'])}")


# =========================================================================
# TASK 2 & 3: EXTRA EXCLUSIONS & FIXED TABLE_LIKE FLAG
# =========================================================================
print("\n" + "=" * 90)
print("TASK 2 & 3: FIXED TABLE_LIKE FLAG & EXTRA EXCLUSIONS")
print("=" * 90)

curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014
allowed_chars = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash | {'\u2013', '\u2022', '\u2010', '\u2026'}

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

# Regex for stripping leading clause markers:
# patterns like "6.2.1", "7.0", "1.1", "17.", "(1)", "(a)", "(ii)", "a)", and bullets
CLAUSE_STRIP_REGEX = re.compile(
    r'^(?:[•▪■\*\-–—\u2022]\s*)*(?:(?:\d+(?:\.\d+)*\.?|\([0-9a-zA-ZivxIVX]+\)|[0-9a-zA-ZivxIVX]+\))\s*)*(?:[•▪■\*\-–—\u2022]\s*)*'
)

def compute_flags_v2(c):
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
    
    # 4. section_markers: only markers at the start of a line or sentence (threshold >= 8)
    sec_markers_cnt = 0
    # Line starts
    for l in lines:
        l_s = l.strip().lstrip('•▪■*-–—\u2022 ').strip()
        if re.match(r'^(?:Section\s+\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*)\b', l_s, re.I):
            sec_markers_cnt += 1
            
    # 5. capital_start_fraction after stripping leading clause marker
    s_list = split_into_sentences(text)
    if s_list:
        cap_starts = 0
        for s in s_list:
            s_clean = s.lstrip()
            stripped = CLAUSE_STRIP_REGEX.sub('', s_clean).lstrip(' "\'“‘(\\[')
            if stripped:
                c_char = stripped[0]
                if c_char.isupper() or c_char.isdigit():
                    cap_starts += 1
            else:
                cap_starts += 1
        cap_frac = cap_starts / len(s_list)
    else:
        cap_frac = 1.0
        
    # Table-like triggers
    t_short = (short_frac > 0.4)
    t_qm = (q_marks >= 3)
    t_sec = (sec_markers_cnt >= 8)
    t_cap = (cap_frac < 0.7)
    table_like = (t_short or t_qm or t_sec or t_cap)
    
    c['question_marks'] = q_marks
    c['short_line_fraction'] = short_frac
    c['bullet_count'] = b_count
    c['section_markers'] = sec_markers_cnt
    c['capital_start_fraction'] = cap_frac
    c['table_like'] = table_like
    c['t_short'] = t_short
    c['t_qm'] = t_qm
    c['t_sec'] = t_sec
    c['t_cap'] = t_cap
    return c

# Extra Exclusions:
# (a) any passage in which the same sentence (normalised whitespace) appears twice
def check_excl_a(c):
    s_list = split_into_sentences(c['text'])
    norm_s = [' '.join(s.split()) for s in s_list if s.strip()]
    return len(norm_s) != len(set(norm_s))

# (b) any passage with more than 25 percent of its words inside parentheses or square brackets
def check_excl_b(c):
    text = c['text']
    total_w = len(text.split())
    if total_w == 0:
        return False
    bracket_matches = re.findall(r'\([^)]*\)|\[[^\]]*\]', text)
    bracket_w = sum(len(m.split()) for m in bracket_matches)
    return (bracket_w / total_w) > 0.25

# (c) any passage that starts with a lowercase letter or a bare bullet fragment of fewer than 6 words
def check_excl_c(c):
    text = c['text'].strip()
    if not text:
        return True
    if text[0].islower():
        return True
    # bare bullet fragment of fewer than 6 words
    if text.startswith(('•', '▪', '■', '-', '*', '\u2022')):
        s_list = split_into_sentences(text)
        if s_list and len(s_list[0].split()) < 6:
            return True
        if c['lines'] and len(c['lines'][0].split()) < 6:
            return True
    return False

# (d) any passage where more than 3 sentences consist of fewer than 6 words
def check_excl_d(c):
    s_list = split_into_sentences(c['text'])
    short_sentences = sum(1 for s in s_list if len(s.split()) < 6)
    return short_sentences > 3


# Run candidate generation across all docs
doc_candidates_all = {}
doc_norm_pages = {}

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    pages_text = [normalize_text_v2(p.get_textpage().get_text_range()) for p in pdf]
    doc_norm_pages[doc_id] = pages_text
    
    line_page_counts = Counter()
    doc_page_paras = []
    for p_no, p_text in enumerate(pages_text):
        p_lines = set(l.strip() for l in p_text.splitlines() if l.strip())
        for l in p_lines:
            line_page_counts[l] += 1
        paras = extract_paragraphs(p_text)
        doc_page_paras.append((p_no + 1, paras))
        
    repeated_lines = {l for l, c in line_page_counts.items() if c > 3 and len(l) > 2}
    
    raw_candidates = []
    for page_num, paras in doc_page_paras:
        sections = []
        cur_sec = 'default'
        cur_p = []
        for ptxt, plines in paras:
            words = ptxt.split()
            is_heading = False
            if len(words) <= 12 and not ptxt.endswith('.'):
                if re.match(r'^(?:Chapter\s+\d+|Section\s+\d+|\d+(\.\d+)*\s+[A-Z]|[A-Z\s]{4,})', ptxt):
                    is_heading = True
            if is_heading:
                if cur_p:
                    sections.append((cur_sec, cur_p))
                    cur_p = []
                cur_sec = ptxt
            else:
                cur_p.append((ptxt, plines))
        if cur_p:
            sections.append((cur_sec, cur_p))
            
        for sec_name, sec_paras in sections:
            n = len(sec_paras)
            for i in range(n):
                start_first_char = sec_paras[i][0].lstrip()[:1]
                if start_first_char.islower():
                    continue
                cur_w = 0
                cur_ptxt = []
                cur_plines = []
                for j in range(i, n):
                    ptxt, plines = sec_paras[j]
                    w = len(ptxt.split())
                    cur_w += w
                    cur_ptxt.append(ptxt)
                    cur_plines.extend(plines)
                    if 80 <= cur_w <= 180:
                        full_txt = ' '.join(cur_ptxt)
                        if full_txt.rstrip().endswith(('.', '!', '?')):
                            raw_candidates.append({
                                'doc_id': doc_id, 'domain': info['domain'], 'page': page_num,
                                'section': sec_name, 'text': full_txt, 'lines': list(cur_plines),
                                'words': cur_w, 'footnote_glue': has_footnote_glue(full_txt),
                                'unofficial_translation': info['unofficial_translation']
                            })
                    elif cur_w > 180:
                        break
                if len(sec_paras[i][0].split()) > 180:
                    s_list_i = split_into_sentences(sec_paras[i][0])
                    cur_s = []
                    for s in s_list_i:
                        cur_s.append(s)
                        sub_txt = ' '.join(cur_s)
                        w = len(sub_txt.split())
                        if 80 <= w <= 180 and sub_txt.rstrip().endswith(('.', '!', '?')):
                            raw_candidates.append({
                                'doc_id': doc_id, 'domain': info['domain'], 'page': page_num,
                                'section': sec_name, 'text': sub_txt, 'lines': list(sec_paras[i][1]),
                                'words': w, 'footnote_glue': has_footnote_glue(sub_txt),
                                'unofficial_translation': info['unofficial_translation']
                            })
                            
    # Standard 9 exclusions
    r1 = [c for c in raw_candidates if not ('table of contents' in c['section'].lower() or c['section'].strip().lower() == 'contents' or any(re.search(r'(\.{3,}|…{3,}|…\s*\d+|\.{2,}\s*\d+|\.\s*\.\s*\.)', l) for l in c['lines']))]
    r2 = [c for c in r1 if not any(sum(ch.isdigit() for ch in l.strip()) / len(l.strip()) > 0.25 for l in c['lines'] if l.strip() and not has_footnote_glue(l.strip()))]
    r3 = [c for c in r2 if not any(l in repeated_lines for l in c['lines'])]
    r4 = [c for c in r3 if not (any(t in c['section'].lower() for t in ['acronym', 'abbreviation']) or (len(c['lines']) > 3 and sum(1 for l in c['lines'] if re.match(r'^[A-Z0-9]{2,}\s+[A-Z]', l)) / len(c['lines']) > 0.5))]
    r5 = [c for c in r4 if not any(t in c['section'].lower() for t in ['reference', 'bibliography', 'literature cited'])]
    r6 = [c for c in r5 if not any(re.search(r'(_{3,}|\.{5,})', l) for l in c['lines'])]
    r7 = [c for c in r6 if count_sentences(c['text']) >= 3]
    r8 = [c for c in r7 if not any(ch not in allowed_chars for ch in c['text'])]
    seen = set()
    r9 = []
    for c in r8:
        norm = ' '.join(c['text'].split())
        if norm not in seen:
            seen.add(norm)
            r9.append(c)
            
    # Compute flags v2 and word offsets
    for c in r9:
        compute_flags_v2(c)
        p_text = pages_text[c['page'] - 1]
        p_words = p_text.split()
        c_words = c['text'].split()
        k = len(c_words)
        m_idx = -1
        for i in range(len(p_words) - k + 1):
            if p_words[i:i+k] == c_words:
                m_idx = i
                break
        assert m_idx != -1, f"Could not locate candidate on page {c['page']}"
        c['start_word'] = m_idx
        c['end_word'] = m_idx + k
        c['context_before'] = ' '.join(p_words[max(0, m_idx - 30):m_idx]) if m_idx > 0 else ""
        
        # Check extra exclusions
        c['excl_a'] = check_excl_a(c)
        c['excl_b'] = check_excl_b(c)
        c['excl_c'] = check_excl_c(c)
        c['excl_d'] = check_excl_d(c)
        c['extra_excluded'] = (c['excl_a'] or c['excl_b'] or c['excl_c'] or c['excl_d'])
        
    doc_candidates_all[doc_id] = r9

print("\n--- TABLE_LIKE CANDIDATE REPORT (NEW RULES) ---")
print(f"{'Doc ID':<34} {'Total':<7} {'Table-Like':<12} {'ShortLine>0.4':<14} {'QM>=3':<7} {'Sec>=8':<8} {'Cap<0.7':<8}")
print("-" * 90)
for doc_id, cands in doc_candidates_all.items():
    tot = len(cands)
    tbl = sum(1 for c in cands if c['table_like'])
    sh = sum(1 for c in cands if c['t_short'])
    qm = sum(1 for c in cands if c['t_qm'])
    sec = sum(1 for c in cands if c['t_sec'])
    cap = sum(1 for c in cands if c['t_cap'])
    print(f"{doc_id:<34} {tot:<7} {tbl:<12} {sh:<14} {qm:<7} {sec:<8} {cap:<8}")

print("\n--- EXTRA EXCLUSIONS BREAKDOWN PER DOC_ID ---")
print(f"{'Doc ID':<34} {'Total':<7} {'(a)DupSent':<12} {'(b)Brackets>25%':<16} {'(c)Lower/Bullet':<16} {'(d)ShortSent>3':<16} {'Any Extra Excl':<14}")
print("-" * 115)
for doc_id, cands in doc_candidates_all.items():
    tot = len(cands)
    ea = sum(1 for c in cands if c['excl_a'])
    eb = sum(1 for c in cands if c['excl_b'])
    ec = sum(1 for c in cands if c['excl_c'])
    ed = sum(1 for c in cands if c['excl_d'])
    any_e = sum(1 for c in cands if c['extra_excluded'])
    print(f"{doc_id:<34} {tot:<7} {ea:<12} {eb:<16} {ec:<16} {ed:<16} {any_e:<14}")


# =========================================================================
# TASK 4: AGRICULTURE SOURCES IN MANIFEST.CSV
# =========================================================================
print("\n" + "=" * 90)
print("TASK 4: EN AGRICULTURE DOCUMENTS IN data/raw_sources/manifest.csv")
print("=" * 90)

agri_manifest_docs = []
with open('data/raw_sources/manifest.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        dom = r.get('domain', '').strip().lower()
        fn = r.get('filename', '').strip()
        lang = r.get('language', '').strip().lower()
        if (dom == 'agriculture' or 'agriculture' in fn.lower()) and lang == 'en':
            agri_manifest_docs.append({
                'doc_id': r.get('doc_id', ''),
                'pair_id': r.get('pair_id', ''),
                'filename': fn,
                'pages': r.get('pages', ''),
                'title': r.get('title', '')
            })

print(f"{'Doc ID':<46} {'Filename':<55} {'Pages':<6}")
print("-" * 110)
for d in agri_manifest_docs:
    print(f"{d['doc_id']:<46} {d['filename']:<55} {d['pages']:<6}")


# =========================================================================
# TASK 5: INTERVAL SCHEDULING SELECTION & RESAMPLING (random.seed=42)
# =========================================================================
print("\n" + "=" * 90)
print("TASK 5: INTERVAL SCHEDULING & RESAMPLING (random.seed=42)")
print("=" * 90)

doc_clean_cands = {}
doc_max_intervals = {}

for doc_id, cands in doc_candidates_all.items():
    # Passages that are non-table_like and pass all extra exclusions
    clean = [c for c in cands if not c['table_like'] and not c['extra_excluded']]
    doc_clean_cands[doc_id] = clean
    
    # Interval scheduling:
    # Sort by end offset: (page, end_word, start_word)
    # Take the earliest-ending passage that starts after the last chosen end
    clean_sorted = sorted(clean, key=lambda c: (c['page'], c['end_word'], c['start_word']))
    chosen_intervals = []
    last_page = -1
    last_end = -1
    for c in clean_sorted:
        if c['page'] > last_page:
            chosen_intervals.append(c)
            last_page = c['page']
            last_end = c['end_word']
        elif c['page'] == last_page:
            if c['start_word'] >= last_end:
                chosen_intervals.append(c)
                last_end = c['end_word']
                
    doc_max_intervals[doc_id] = chosen_intervals

print(f"{'Doc ID':<34} {'Domain':<14} {'Clean Candidates':<18} {'Max Non-Overlapping':<20}")
print("-" * 90)
for doc_id, info in DOCS.items():
    cl_cnt = len(doc_clean_cands[doc_id])
    mx_cnt = len(doc_max_intervals[doc_id])
    print(f"{doc_id:<34} {info['domain']:<14} {cl_cnt:<18} {mx_cnt:<20}")

# Resampling rules with random.seed(42):
# Agriculture: up to 40 (agri_mechanization_2020 first; add agri_seed_rules_2020 only if fewer than 40, with unofficial_translation=True)
# Disaster: up to 45 with at most 12 per doc_id
# Education: up to 45

random.seed(42)

# Agriculture sampling
agri_mech_avail = list(doc_max_intervals['agri_mechanization_2020'])
agri_seed_avail = list(doc_max_intervals['agri_seed_rules_2020'])

sampled_agri = []
if len(agri_mech_avail) >= 40:
    sampled_agri = random.sample(agri_mech_avail, 40)
    for c in sampled_agri:
        c['unofficial_translation'] = False
else:
    sampled_agri.extend(agri_mech_avail)
    for c in sampled_agri:
        c['unofficial_translation'] = False
    needed = 40 - len(sampled_agri)
    add_k = min(needed, len(agri_seed_avail))
    if add_k > 0:
        seed_sample = random.sample(agri_seed_avail, add_k) if len(agri_seed_avail) > add_k else list(agri_seed_avail)
        for c in seed_sample:
            c['unofficial_translation'] = True
        sampled_agri.extend(seed_sample)

# Disaster sampling (up to 45 with at most 12 per doc_id)
disaster_doc_ids = [
    'disaster_sod_2019',
    'disaster_displacement_nap_2022',
    'disaster_displacement_strat_2021',
    'disaster_emcrp_rpf',
    'disaster_emcrp_esmf'
]

disaster_per_doc_pool = {}
for d_id in disaster_doc_ids:
    avail = list(doc_max_intervals[d_id])
    take_k = min(12, len(avail))
    if len(avail) > take_k:
        disaster_per_doc_pool[d_id] = random.sample(avail, take_k)
    else:
        disaster_per_doc_pool[d_id] = list(avail)

disaster_pool_flat = []
for d_id in disaster_doc_ids:
    disaster_pool_flat.extend(disaster_per_doc_pool[d_id])

if len(disaster_pool_flat) > 45:
    sampled_disaster = random.sample(disaster_pool_flat, 45)
else:
    sampled_disaster = list(disaster_pool_flat)

# Education sampling (up to 45 from edu_nep_2010)
edu_avail = list(doc_max_intervals['edu_nep_2010'])
sampled_edu = random.sample(edu_avail, min(45, len(edu_avail)))

print(f"\nFinal Resample Counts:")
print(f"  Agriculture: {len(sampled_agri)} sampled (mech={sum(1 for c in sampled_agri if c['doc_id']=='agri_mechanization_2020')}, seed={sum(1 for c in sampled_agri if c['doc_id']=='agri_seed_rules_2020')})")
print(f"  Disaster: {len(sampled_disaster)} sampled across {len(disaster_doc_ids)} docs:")
for d_id in disaster_doc_ids:
    print(f"    - {d_id}: {sum(1 for c in sampled_disaster if c['doc_id']==d_id)} sampled (pool was {len(disaster_per_doc_pool[d_id])})")
print(f"  Education: {len(sampled_edu)} sampled (available was {len(edu_avail)})")


# =========================================================================
# TASK 6: OUTPUT data_v2/screening_v2.csv (UTF-8 WITH BOM)
# =========================================================================
print("\n" + "=" * 90)
print("TASK 6: OUTPUT data_v2/screening_v2.csv (UTF-8 WITH BOM)")
print("=" * 90)

tok = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

all_sampled_v2 = []
scr_counter = 1

for c in sampled_agri:
    c['scr_id'] = f"scr_{scr_counter:04d}"
    c['gemma_tokens'] = len(tok.encode(c['text'], add_special_tokens=False))
    scr_counter += 1
    all_sampled_v2.append(c)

for c in sampled_disaster:
    c['scr_id'] = f"scr_{scr_counter:04d}"
    c['gemma_tokens'] = len(tok.encode(c['text'], add_special_tokens=False))
    scr_counter += 1
    all_sampled_v2.append(c)

for c in sampled_edu:
    c['scr_id'] = f"scr_{scr_counter:04d}"
    c['gemma_tokens'] = len(tok.encode(c['text'], add_special_tokens=False))
    scr_counter += 1
    all_sampled_v2.append(c)

out_v2_path = 'data_v2/screening_v2.csv'
fieldnames_v2 = [
    'scr_id', 'domain', 'doc_id', 'page', 'start_word', 'end_word', 'words',
    'gemma_tokens', 'table_like', 'footnote_glue', 'unofficial_translation',
    'context_before', 'text', 'keep', 'reject_reason', 'notes'
]

with open(out_v2_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_v2)
    writer.writeheader()
    for c in all_sampled_v2:
        writer.writerow({
            'scr_id': c['scr_id'],
            'domain': c['domain'],
            'doc_id': c['doc_id'],
            'page': c['page'],
            'start_word': c['start_word'],
            'end_word': c['end_word'],
            'words': c['words'],
            'gemma_tokens': c['gemma_tokens'],
            'table_like': c['table_like'],
            'footnote_glue': c['footnote_glue'],
            'unofficial_translation': c['unofficial_translation'],
            'context_before': c['context_before'],
            'text': c['text'],
            'keep': '',
            'reject_reason': '',
            'notes': ''
        })

print(f"Saved {len(all_sampled_v2)} passages to {out_v2_path} (UTF-8 with BOM)")


# =========================================================================
# TASK 7: TOKEN STATISTICS & 5 SAMPLED AGRICULTURE PASSAGES
# =========================================================================
print("\n" + "=" * 90)
print("TASK 7: TOKEN STATISTICS & 5 SAMPLED AGRICULTURE PASSAGES")
print("=" * 90)

print("\n--- TOKEN STATISTICS PER DOC_ID ---")
print(f"{'Doc ID':<34} {'Domain':<14} {'Sampled':<8} {'Mean Words':<12} {'Mean Gemma':<12} {'p90 Gemma':<12}")
print("-" * 92)
for doc_id, info in DOCS.items():
    doc_cands = [c for c in all_sampled_v2 if c['doc_id'] == doc_id]
    cnt = len(doc_cands)
    if cnt > 0:
        w_arr = [c['words'] for c in doc_cands]
        t_arr = [c['gemma_tokens'] for c in doc_cands]
        m_w = np.mean(w_arr)
        m_t = np.mean(t_arr)
        p90_t = np.percentile(t_arr, 90)
        print(f"{doc_id:<34} {info['domain']:<14} {cnt:<8} {m_w:<12.1f} {m_t:<12.1f} {p90_t:<12.1f}")
    else:
        print(f"{doc_id:<34} {info['domain']:<14} {cnt:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("\n--- TOKEN STATISTICS PER DOMAIN ---")
print(f"{'Domain':<16} {'Sampled':<8} {'Mean Words':<12} {'Mean Gemma':<12} {'p90 Gemma':<12}")
print("-" * 60)
for dom in ['agriculture', 'disaster', 'education']:
    dom_cands = [c for c in all_sampled_v2 if c['domain'] == dom]
    cnt = len(dom_cands)
    if cnt > 0:
        w_arr = [c['words'] for c in dom_cands]
        t_arr = [c['gemma_tokens'] for c in dom_cands]
        m_w = np.mean(w_arr)
        m_t = np.mean(t_arr)
        p90_t = np.percentile(t_arr, 90)
        print(f"{dom:<16} {cnt:<8} {m_w:<12.1f} {m_t:<12.1f} {p90_t:<12.1f}")
    else:
        print(f"{dom:<16} {cnt:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("\n" + "=" * 90)
print("5 SAMPLED AGRICULTURE PASSAGES IN FULL")
print("=" * 90)

agri_sampled_all = [c for c in all_sampled_v2 if c['domain'] == 'agriculture']
for idx, c in enumerate(agri_sampled_all[:5], 1):
    print(f"\n###############################################################################")
    print(f"AGRICULTURE PASSAGE {idx}: [{c['scr_id']}] {c['doc_id']} (Page {c['page']}, Words: {c['words']}, Gemma Tokens: {c['gemma_tokens']})")
    print(f"Offsets: start_word={c['start_word']}, end_word={c['end_word']}, footnote_glue={c['footnote_glue']}, unofficial={c['unofficial_translation']}")
    print(f"Context Before: {repr(c['context_before'])}")
    print(f"-------------------------------------------------------------------------------")
    print(c['text'])
