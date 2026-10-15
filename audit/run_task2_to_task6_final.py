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
# TASK 1: OVERLAP ANALYSIS OF 220 SAMPLED PASSAGES
# =========================================================================
print("=" * 90)
print("TASK 1: OVERLAP ANALYSIS OF 220 SAMPLED PASSAGES IN data_v2/en_candidates.jsonl")
print("=" * 90)

with open('data_v2/en_candidates.jsonl', 'r', encoding='utf-8') as f:
    sampled_220 = [json.loads(line) for line in f]

by_doc_220 = defaultdict(list)
for c in sampled_220:
    by_doc_220[c['doc_id']].append(c)

task1_stats = {}

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    passages = by_doc_220.get(doc_id, [])
    
    located = []
    for c in passages:
        p_no = c['page'] - 1
        page_text = pdf[p_no].get_textpage().get_text_range()
        p_words = page_text.split()
        c_words = c['text'].split()
        k = len(c_words)
        match_idx = -1
        for i in range(len(p_words) - k + 1):
            if p_words[i:i+k] == c_words:
                match_idx = i
                break
        assert match_idx != -1, f"Could not locate {c['candidate_id']} on page {c['page']}"
        located.append({
            'candidate_id': c['candidate_id'],
            'page': c['page'],
            'start': match_idx,
            'end': match_idx + k,
            'words': k
        })
        
    num_passages = len(located)
    overlapping_pairs = 0
    pair_details = []
    for i in range(num_passages):
        for j in range(i + 1, num_passages):
            pA = located[i]
            pB = located[j]
            if pA['page'] == pB['page']:
                ov_start = max(pA['start'], pB['start'])
                ov_end = min(pA['end'], pB['end'])
                if ov_start < ov_end:
                    overlapping_pairs += 1
                    pair_details.append((pA['candidate_id'], pB['candidate_id'], pA['page'], ov_end - ov_start))
                    
    page_word_coverage = defaultdict(lambda: defaultdict(int))
    total_sampled_words = sum(p['words'] for p in located)
    
    for p in located:
        for w_pos in range(p['start'], p['end']):
            page_word_coverage[p['page']][w_pos] += 1
            
    multicovered_token_instances = 0
    for p in located:
        for w_pos in range(p['start'], p['end']):
            if page_word_coverage[p['page']][w_pos] > 1:
                multicovered_token_instances += 1
                
    frac_multicovered = (multicovered_token_instances / total_sampled_words) if total_sampled_words > 0 else 0.0
    
    task1_stats[doc_id] = {
        'sampled_count': num_passages,
        'total_words': total_sampled_words,
        'overlapping_pairs': overlapping_pairs,
        'multicovered_token_instances': multicovered_token_instances,
        'frac_multicovered': frac_multicovered,
        'pair_details': pair_details
    }

print(f"{'Doc ID':<34} {'Sampled':<8} {'Overlap Pairs':<15} {'Total Words':<13} {'Multi-cov Words':<17} {'Overlap Frac':<14}")
print("-" * 105)
for doc_id, s in task1_stats.items():
    print(f"{doc_id:<34} {s['sampled_count']:<8} {s['overlapping_pairs']:<15} {s['total_words']:<13} {s['multicovered_token_instances']:<17} {s['frac_multicovered']*100:.2f}%")


# =========================================================================
# TASK 2: NORMALISATION OF PAGE TEXT
# =========================================================================
print("\n" + "=" * 90)
print("TASK 2: PAGE TEXT NORMALISATION & RECOVERY COMPARISON")
print("=" * 90)

random.seed(42)

# (a) Contexts of U+FFFE per doc_id
doc_fffe_all = {}
doc_fffe_sampled = {}

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    ctxs = []
    for p_no, page in enumerate(pdf):
        text = page.get_textpage().get_text_range()
        for i, ch in enumerate(text):
            if ch == '\ufffe':
                c_before = text[max(0, i-20):i]
                c_after = text[i+1:min(len(text), i+21)]
                ctxs.append({
                    'doc_id': doc_id,
                    'page': p_no + 1,
                    'index': i,
                    'before': c_before,
                    'after': c_after,
                    'raw': text[max(0, i-20):min(len(text), i+21)]
                })
    doc_fffe_all[doc_id] = ctxs
    take_k = min(20, len(ctxs))
    if take_k > 0:
        doc_fffe_sampled[doc_id] = random.sample(ctxs, take_k)
    else:
        doc_fffe_sampled[doc_id] = []

print("\n--- U+FFFE OCCURRENCES PER DOC_ID ---")
for doc_id, ctxs in doc_fffe_all.items():
    sampled_cnt = len(doc_fffe_sampled[doc_id])
    print(f"  {doc_id:<34}: {len(ctxs):>3} total occurrences (sampled {sampled_cnt})")

total_sampled_fffe = sum(len(v) for v in doc_fffe_sampled.values())
print(f"Total sampled contexts across docs: {total_sampled_fffe}")

print("\n--- PRINTING ALL SAMPLED U+FFFE CONTEXTS (20 chars either side, repr) ---")
for doc_id, ctx_list in doc_fffe_sampled.items():
    print(f"\n>>> Doc: {doc_id} ({len(ctx_list)} contexts)")
    for idx, c in enumerate(ctx_list, 1):
        print(f"  [{idx:2d}] p.{c['page']:<3} {repr(c['before'])} + U+FFFE + {repr(c['after'])}")

def normalize_fffe(text):
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
    return ''.join(res)

print("\n--- 20 RESULTING NORMALIZED CONTEXTS OF U+FFFE REPLACEMENT ---")
all_sampled_fffe_flat = [c for cl in doc_fffe_sampled.values() for c in cl]
random.seed(42)
sample_20_fffe = random.sample(all_sampled_fffe_flat, 20)
for idx, c in enumerate(sample_20_fffe, 1):
    raw_snip = c['raw']
    norm_snip = normalize_fffe(raw_snip)
    print(f"  [{idx:2d}] {c['doc_id']} p.{c['page']:<3} Raw: {repr(raw_snip)}  ==>  Norm: {repr(norm_snip)}")

def normalize_full_page_text(text):
    # (a) U+FFFE
    text = normalize_fffe(text)
    # (b) Bullets U+F0B7, U+F0A7, U+F0D8, U+F0FC -> U+2022
    for b in ['\uf0b7', '\uf0a7', '\uf0d8', '\uf0fc']:
        text = text.replace(b, '\u2022')
    return text

# Character sets
curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014
allowed_chars_old = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash
allowed_chars_new = allowed_chars_old | {'\u2013', '\u2022', '\u2010', '\u2026'}

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

def run_extraction_and_exclusions(doc_id, info, normalize=False):
    pdf = pdfium.PdfDocument(info['path'])
    pages_text = []
    for page in pdf:
        t = page.get_textpage().get_text_range()
        if normalize:
            t = normalize_full_page_text(t)
        pages_text.append(t)
        
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
                                'doc_id': doc_id,
                                'domain': info['domain'],
                                'page': page_num,
                                'section': sec_name,
                                'text': full_txt,
                                'lines': list(cur_plines),
                                'words': cur_w,
                                'footnote_glue': has_footnote_glue(full_txt),
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
                                'doc_id': doc_id,
                                'domain': info['domain'],
                                'page': page_num,
                                'section': sec_name,
                                'text': sub_txt,
                                'lines': list(sec_paras[i][1]),
                                'words': w,
                                'footnote_glue': has_footnote_glue(sub_txt),
                                'unofficial_translation': info['unofficial_translation']
                            })
                            
    allowed = allowed_chars_new if normalize else allowed_chars_old
    
    # 9 exclusion rules
    r1 = []
    for c in raw_candidates:
        if 'table of contents' in c['section'].lower() or c['section'].strip().lower() == 'contents':
            continue
        if any(re.search(r'(\.{3,}|…{3,}|…\s*\d+|\.{2,}\s*\d+|\.\s*\.\s*\.)', l) for l in c['lines']):
            continue
        r1.append(c)
        
    r2 = []
    for c in r1:
        high_digit = False
        for l in c['lines']:
            stripped = l.strip()
            if not stripped or has_footnote_glue(stripped):
                continue
            if sum(ch.isdigit() for ch in stripped) / len(stripped) > 0.25:
                high_digit = True
                break
        if not high_digit:
            r2.append(c)
            
    r3 = [c for c in r2 if not any(l in repeated_lines for l in c['lines'])]
    
    r4 = []
    for c in r3:
        sec = c['section'].lower()
        if any(t in sec for t in ['acronym', 'abbreviation']):
            continue
        acro_lines = sum(1 for l in c['lines'] if re.match(r'^[A-Z0-9]{2,}\s+[A-Z]', l))
        if len(c['lines']) > 3 and acro_lines / len(c['lines']) > 0.5:
            continue
        r4.append(c)
        
    r5 = [c for c in r4 if not any(t in c['section'].lower() for t in ['reference', 'bibliography', 'literature cited'])]
    r6 = [c for c in r5 if not any(re.search(r'(_{3,}|\.{5,})', l) for l in c['lines'])]
    r7 = [c for c in r6 if count_sentences(c['text']) >= 3]
    r8 = [c for c in r7 if not any(ch not in allowed for ch in c['text'])]
    
    seen = set()
    r9 = []
    for c in r8:
        norm = ' '.join(c['text'].split())
        if norm not in seen:
            seen.add(norm)
            r9.append(c)
            
    return r9, pages_text

print("\n--- BEFORE VS AFTER NORMALISATION CANDIDATE RECOVERY ---")
cands_before = {}
cands_after = {}
norm_pages_by_doc = {}

for doc_id, info in DOCS.items():
    c_b, _ = run_extraction_and_exclusions(doc_id, info, normalize=False)
    c_a, p_texts = run_extraction_and_exclusions(doc_id, info, normalize=True)
    cands_before[doc_id] = c_b
    cands_after[doc_id] = c_a
    norm_pages_by_doc[doc_id] = p_texts

print(f"{'Doc ID':<34} {'Before Norm':<14} {'After Norm':<14} {'Recovered Count':<16}")
print("-" * 80)
for doc_id in DOCS:
    cb_len = len(cands_before[doc_id])
    ca_len = len(cands_after[doc_id])
    diff = ca_len - cb_len
    print(f"{doc_id:<34} {cb_len:<14} {ca_len:<14} +{diff:<15}")


# =========================================================================
# TASK 3: TABLE AND CHECKLIST FLAGS
# =========================================================================
print("\n" + "=" * 90)
print("TASK 3: TABLE AND CHECKLIST FLAGS")
print("=" * 90)

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

table_flag_stats = {}
table_like_cands_by_doc = {}
non_table_cands_by_doc = {}

for doc_id in DOCS:
    cands = cands_after[doc_id]
    p_texts = norm_pages_by_doc[doc_id]
    
    # Compute flags and locate word offsets
    for c in cands:
        compute_flags(c)
        p_words = p_texts[c['page'] - 1].split()
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
        
    tbl_like = [c for c in cands if c['table_like']]
    non_tbl = [c for c in cands if not c['table_like']]
    table_like_cands_by_doc[doc_id] = tbl_like
    non_table_cands_by_doc[doc_id] = non_tbl
    
    table_flag_stats[doc_id] = {
        'total': len(cands),
        'table_like': len(tbl_like),
        'non_table': len(non_tbl),
        'short_lines_trigger': sum(1 for c in cands if c['short_line_fraction'] > 0.4),
        'qm_trigger': sum(1 for c in cands if c['question_marks'] >= 3),
        'sec_trigger': sum(1 for c in cands if c['section_markers'] >= 4),
        'cap_trigger': sum(1 for c in cands if c['capital_start_fraction'] < 0.7),
    }

print(f"{'Doc ID':<34} {'Total Cands':<12} {'Table-Like':<12} {'Non-Table':<12} {'ShortLine>0.4':<14} {'QM>=3':<7} {'Sec>=4':<8} {'Cap<0.7':<8}")
print("-" * 110)
for doc_id, s in table_flag_stats.items():
    print(f"{doc_id:<34} {s['total']:<12} {s['table_like']:<12} {s['non_table']:<12} {s['short_lines_trigger']:<14} {s['qm_trigger']:<7} {s['sec_trigger']:<8} {s['cap_trigger']:<8}")


# =========================================================================
# TASK 4: GREEDY NON-OVERLAPPING SELECTION & RESAMPLING
# =========================================================================
print("\n" + "=" * 90)
print("TASK 4: NON-OVERLAPPING CANDIDATE BUILDING & SAMPLING (random.seed=42)")
print("=" * 90)

greedy_available_by_doc = {}

for doc_id in DOCS:
    non_tbl = list(non_table_cands_by_doc[doc_id])
    # Sort greedily in page order
    non_tbl.sort(key=lambda c: (c['page'], c['start_word'], c['end_word']))
    
    chosen = []
    last_page = -1
    last_end = -1
    for c in non_tbl:
        if c['page'] > last_page:
            chosen.append(c)
            last_page = c['page']
            last_end = c['end_word']
        elif c['page'] == last_page:
            if c['start_word'] >= last_end:
                chosen.append(c)
                last_end = c['end_word']
                
    greedy_available_by_doc[doc_id] = chosen

print("\n--- MAXIMUM NON-OVERLAPPING NON-TABLE-LIKE PASSAGES AVAILABLE PER DOC_ID ---")
print(f"{'Doc ID':<34} {'Domain':<14} {'Available Non-Overlapping Passages':<36}")
print("-" * 85)
for doc_id, info in DOCS.items():
    avail_cnt = len(greedy_available_by_doc[doc_id])
    print(f"{doc_id:<34} {info['domain']:<14} {avail_cnt:<36}")

# Resampling rules:
# Agriculture: up to 45 (agri_mechanization_2020 first; add agri_seed_rules_2020 only if fewer than 45, unofficial_translation=True)
# Disaster: up to 80 with at most 20 per doc_id
# Education: up to 60 from edu_nep_2010

random.seed(42)

# Agriculture sampling
agri_mech_avail = list(greedy_available_by_doc['agri_mechanization_2020'])
agri_seed_avail = list(greedy_available_by_doc['agri_seed_rules_2020'])

sampled_agri = []
if len(agri_mech_avail) >= 45:
    sampled_agri = random.sample(agri_mech_avail, 45)
    for c in sampled_agri:
        c['unofficial_translation'] = False
else:
    sampled_agri.extend(agri_mech_avail)
    for c in sampled_agri:
        c['unofficial_translation'] = False
    needed = 45 - len(sampled_agri)
    add_k = min(needed, len(agri_seed_avail))
    if add_k > 0:
        seed_sample = random.sample(agri_seed_avail, add_k) if len(agri_seed_avail) > add_k else list(agri_seed_avail)
        for c in seed_sample:
            c['unofficial_translation'] = True
        sampled_agri.extend(seed_sample)

# Disaster sampling (up to 80, at most 20 per doc_id)
disaster_doc_ids = [
    'disaster_sod_2019',
    'disaster_displacement_nap_2022',
    'disaster_displacement_strat_2021',
    'disaster_emcrp_rpf',
    'disaster_emcrp_esmf'
]

disaster_per_doc_pool = {}
for d_id in disaster_doc_ids:
    avail = list(greedy_available_by_doc[d_id])
    take_k = min(20, len(avail))
    if len(avail) > take_k:
        disaster_per_doc_pool[d_id] = random.sample(avail, take_k)
    else:
        disaster_per_doc_pool[d_id] = list(avail)

disaster_pool_flat = []
for d_id in disaster_doc_ids:
    disaster_pool_flat.extend(disaster_per_doc_pool[d_id])

if len(disaster_pool_flat) > 80:
    sampled_disaster = random.sample(disaster_pool_flat, 80)
else:
    sampled_disaster = list(disaster_pool_flat)

# Education sampling (up to 60 from edu_nep_2010)
edu_avail = list(greedy_available_by_doc['edu_nep_2010'])
sampled_edu = random.sample(edu_avail, min(60, len(edu_avail)))

print(f"\nSampled Agriculture: {len(sampled_agri)} (mech={sum(1 for c in sampled_agri if c['doc_id']=='agri_mechanization_2020')}, seed={sum(1 for c in sampled_agri if c['doc_id']=='agri_seed_rules_2020')})")
print(f"Sampled Disaster: {len(sampled_disaster)} across {len(disaster_doc_ids)} docs")
for d_id in disaster_doc_ids:
    print(f"  - {d_id}: {sum(1 for c in sampled_disaster if c['doc_id']==d_id)} sampled (pool was {len(disaster_per_doc_pool[d_id])})")
print(f"Sampled Education: {len(sampled_edu)} (available was {len(edu_avail)})")


# =========================================================================
# TASK 5: TOKENIZATION & OUTPUT CSVS
# =========================================================================
print("\n" + "=" * 90)
print("TASK 5: TOKENIZATION (google/gemma-3-1b-it) & CSV WRITING")
print("=" * 90)

tok = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

# Tokenize all sampled passages
all_sampled_final = []
cand_id_counter = 1

for c in sampled_agri:
    c['candidate_id'] = f"en_cand_{cand_id_counter:04d}"
    c['gemma_tokens'] = len(tok.encode(c['text'], add_special_tokens=False))
    cand_id_counter += 1
    all_sampled_final.append(c)

for c in sampled_disaster:
    c['candidate_id'] = f"en_cand_{cand_id_counter:04d}"
    c['gemma_tokens'] = len(tok.encode(c['text'], add_special_tokens=False))
    cand_id_counter += 1
    all_sampled_final.append(c)

for c in sampled_edu:
    c['candidate_id'] = f"en_cand_{cand_id_counter:04d}"
    c['gemma_tokens'] = len(tok.encode(c['text'], add_special_tokens=False))
    cand_id_counter += 1
    all_sampled_final.append(c)

# Write data_v2/screening.csv
os.makedirs('data_v2', exist_ok=True)
screening_csv_path = 'data_v2/screening.csv'
fieldnames = [
    'candidate_id', 'domain', 'doc_id', 'page', 'words', 'gemma_tokens',
    'table_like', 'footnote_glue', 'unofficial_translation', 'text',
    'keep', 'reject_reason', 'notes'
]

with open(screening_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for c in all_sampled_final:
        writer.writerow({
            'candidate_id': c['candidate_id'],
            'domain': c['domain'],
            'doc_id': c['doc_id'],
            'page': c['page'],
            'words': c['words'],
            'gemma_tokens': c['gemma_tokens'],
            'table_like': c['table_like'],
            'footnote_glue': c['footnote_glue'],
            'unofficial_translation': c['unofficial_translation'],
            'text': c['text'],
            'keep': '',
            'reject_reason': '',
            'notes': ''
        })

print(f"Saved {len(all_sampled_final)} sampled passages to {screening_csv_path} (UTF-8 with BOM)")

# Rejected table-like passages
all_table_like_rejected = []
rej_id_counter = 1
for doc_id, tbl_list in table_like_cands_by_doc.items():
    for c in tbl_list:
        c['candidate_id'] = f"rej_table_{rej_id_counter:04d}"
        c['gemma_tokens'] = len(tok.encode(c['text'], add_special_tokens=False))
        rej_id_counter += 1
        all_table_like_rejected.append(c)

rejected_csv_path = 'data_v2/screening_rejected_table_like.csv'
with open(rejected_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for c in all_table_like_rejected:
        writer.writerow({
            'candidate_id': c['candidate_id'],
            'domain': c['domain'],
            'doc_id': c['doc_id'],
            'page': c['page'],
            'words': c['words'],
            'gemma_tokens': c['gemma_tokens'],
            'table_like': c['table_like'],
            'footnote_glue': c['footnote_glue'],
            'unofficial_translation': c['unofficial_translation'],
            'text': c['text'],
            'keep': '',
            'reject_reason': '',
            'notes': ''
        })

print(f"Saved {len(all_table_like_rejected)} rejected table_like passages to {rejected_csv_path} (UTF-8 with BOM)")


# =========================================================================
# TASK 6: REPORTING & PRINTING 3 SAMPLED PASSAGES PER DOMAIN
# =========================================================================
print("\n" + "=" * 90)
print("TASK 6: TOKEN STATISTICS & FULL PASSAGES PER DOMAIN")
print("=" * 90)

print("\n--- REPORT PER DOC_ID ---")
print(f"{'Doc ID':<34} {'Domain':<14} {'Sampled':<8} {'Mean Words':<12} {'Mean Gemma':<12} {'p90 Gemma':<12}")
print("-" * 92)
for doc_id, info in DOCS.items():
    doc_cands = [c for c in all_sampled_final if c['doc_id'] == doc_id]
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

print("\n--- REPORT PER DOMAIN ---")
print(f"{'Domain':<16} {'Sampled':<8} {'Mean Words':<12} {'Mean Gemma':<12} {'p90 Gemma':<12}")
print("-" * 60)
for dom in ['agriculture', 'disaster', 'education']:
    dom_cands = [c for c in all_sampled_final if c['domain'] == dom]
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
print("3 SAMPLED PASSAGES PER DOMAIN IN FULL")
print("=" * 90)

for dom in ['agriculture', 'disaster', 'education']:
    dom_cands = [c for c in all_sampled_final if c['domain'] == dom]
    print(f"\n###############################################################################")
    print(f"DOMAIN: {dom.upper()} (Showing 3 sampled passages in full)")
    print(f"###############################################################################")
    for idx, c in enumerate(dom_cands[:3], 1):
        print(f"\n--- Passage {idx}: [{c['candidate_id']}] Doc: {c['doc_id']} (Page {c['page']}, Words: {c['words']}, Gemma Tokens: {c['gemma_tokens']}) ---")
        print(c['text'])
