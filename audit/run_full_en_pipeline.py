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

# Target documents
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
# STEP 1: EXTRACT ENGLISH TEXT PER PAGE & CHARACTER AUDIT
# =========================================================================
curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014

# "characters outside basic Latin, digits, whitespace and common punctuation (including curly quotes and the em dash)"
allowed_chars = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash

print("=" * 80)
print("TASK 1: ENGLISH TEXT EXTRACTION AND CHARACTER AUDIT")
print("=" * 80)

step1_stats = {}
doc_pages_text = {}

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    num_pages = len(pdf)
    total_words = 0
    total_chars = 0
    unusual_counts = Counter()
    pages_text = []
    
    for page in pdf:
        text = page.get_textpage().get_text_range()
        pages_text.append(text)
        total_chars += len(text)
        total_words += len(text.split())
        for ch in text:
            if ch not in allowed_chars:
                unusual_counts[ch] += 1
                
    doc_pages_text[doc_id] = pages_text
    unusual_total = sum(unusual_counts.values())
    unusual_frac = unusual_total / total_chars if total_chars > 0 else 0.0
    top10_unusual = unusual_counts.most_common(10)
    
    step1_stats[doc_id] = {
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

# =========================================================================
# STEP 2: BUILD ENGLISH CANDIDATE PASSAGES & SEQUENTIAL EXCLUSIONS
# =========================================================================
print("\n" + "=" * 80)
print("TASK 2: BUILD ENGLISH CANDIDATE PASSAGES & SEQUENTIAL EXCLUSIONS")
print("=" * 80)

def has_footnote_glue(text):
    # Footnote digit glued to a number: e.g. 727,0001 or 919,0002
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

# Process candidate passages per document
candidates_by_doc = {}
attrition_by_doc = {}

for doc_id, info in DOCS.items():
    pages_text = doc_pages_text[doc_id]
    
    # 1. Identify headers and footers repeated on > 3 pages
    line_page_counts = Counter()
    doc_page_paras = []
    for p_no, p_text in enumerate(pages_text):
        p_lines = set(l.strip() for l in p_text.splitlines() if l.strip())
        for l in p_lines:
            line_page_counts[l] += 1
        paras = extract_paragraphs(p_text)
        doc_page_paras.append((p_no + 1, paras))
        
    repeated_lines = {l for l, c in line_page_counts.items() if c > 3 and len(l) > 2}
    
    # 2. Build candidate passages
    # For each page, group paragraphs into sections
    raw_candidates = []
    for page_num, paras in doc_page_paras:
        sections = []
        cur_sec = "default"
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
                # Paragraph start must not be a dangling lowercase fragment carried over from previous page
                start_first_char = sec_paras[i][0].lstrip()[:1]
                if start_first_char.islower():
                    continue

                # Form candidate passage starting at paragraph i start
                # Merge neighbouring paragraphs from same page and section into 80-180 words ending at sentence end
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
                                'source_file': info['path'],
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
                        
                # Also handle long paragraphs (> 180 words) where slicing at sentence boundary fits 80-180 words
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
                                'source_file': info['path'],
                                'page': page_num,
                                'section': sec_name,
                                'text': sub_txt,
                                'lines': list(sec_paras[i][1]),
                                'words': w,
                                'footnote_glue': has_footnote_glue(sub_txt),
                                'unofficial_translation': info['unofficial_translation']
                            })

    initial_count = len(raw_candidates)
    
    # ---------------------------------------------------------------------
    # Sequential Exclusion Pipeline:
    # ---------------------------------------------------------------------
    # Rule 1: Table of contents lines (dot leaders or trailing page numbers)
    def is_toc_candidate(c):
        if 'table of contents' in c['section'].lower() or c['section'].strip().lower() == 'contents':
            return True
        for l in c['lines']:
            if re.search(r'(\.{3,}|…{2,}|\.\s*\.\s*\.)', l):
                return True
        return False
        
    after_r1 = [c for c in raw_candidates if not is_toc_candidate(c)]
    
    # Rule 2: Lines where digits exceed 25 percent of characters
    def has_high_digit_line(c):
        for l in c['lines']:
            stripped = l.strip()
            if not stripped:
                continue
            # If line has glued footnote digit (like 727,0001), do NOT exclude (keep passage, footnote_glue=True)
            if has_footnote_glue(stripped):
                continue
            digit_cnt = sum(ch.isdigit() for ch in stripped)
            if (digit_cnt / len(stripped)) > 0.25:
                return True
        return False
        
    after_r2 = [c for c in after_r1 if not has_high_digit_line(c)]
    
    # Rule 3: Headers and footers repeated on more than 3 pages
    def has_repeated_header_footer(c):
        return any(l in repeated_lines for l in c['lines'])
        
    after_r3 = [c for c in after_r2 if not has_repeated_header_footer(c)]
    
    # Rule 4: Acronym lists
    def is_acronym_candidate(c):
        sec = c['section'].lower()
        if any(term in sec for term in ['acronym', 'abbreviation']):
            return True
        acro_lines = sum(1 for l in c['lines'] if re.match(r'^[A-Z0-9]{2,}\s+[A-Z]', l))
        if len(c['lines']) > 3 and acro_lines / len(c['lines']) > 0.5:
            return True
        return False
        
    after_r4 = [c for c in after_r3 if not is_acronym_candidate(c)]
    
    # Rule 5: Reference or bibliography sections
    def is_reference_candidate(c):
        sec = c['section'].lower()
        return any(term in sec for term in ['reference', 'bibliography', 'literature cited'])
        
    after_r5 = [c for c in after_r4 if not is_reference_candidate(c)]
    
    # Rule 6: Forms with blank fill-in lines
    def has_blank_fill_in(c):
        for l in c['lines']:
            if re.search(r'(_{3,}|\.{5,})', l):
                return True
        return False
        
    after_r6 = [c for c in after_r5 if not has_blank_fill_in(c)]
    
    # Rule 7: Passages with fewer than 3 sentences
    def has_fewer_than_3_sentences(c):
        return count_sentences(c['text']) < 3
        
    after_r7 = [c for c in after_r6 if not has_fewer_than_3_sentences(c)]
    
    # Rule 8: Any passage containing characters flagged as unusual in step 1
    def has_unusual_characters(c):
        return any(ch not in allowed_chars for ch in c['text'])
        
    after_r8 = [c for c in after_r7 if not has_unusual_characters(c)]
    
    # Rule 9: Duplicates
    seen_texts = set()
    after_r9 = []
    for c in after_r8:
        norm_txt = ' '.join(c['text'].split())
        if norm_txt not in seen_texts:
            seen_texts.add(norm_txt)
            after_r9.append(c)
            
    candidates_by_doc[doc_id] = after_r9
    
    attrition_by_doc[doc_id] = {
        'initial': initial_count,
        'after_r1': len(after_r1),
        'after_r2': len(after_r2),
        'after_r3': len(after_r3),
        'after_r4': len(after_r4),
        'after_r5': len(after_r5),
        'after_r6': len(after_r6),
        'after_r7': len(after_r7),
        'after_r8': len(after_r8),
        'after_r9': len(after_r9),
    }

print("Candidate generation and sequential filtering complete.")

# =========================================================================
# STEP 3: SAMPLING CANDIDATES WITH random.seed(42)
# =========================================================================
print("\n" + "=" * 80)
print("TASK 3: CANDIDATE SAMPLING WITH random.seed(42)")
print("=" * 80)

agri_mech_pool = list(candidates_by_doc['agri_mechanization_2020'])
agri_seed_pool = list(candidates_by_doc['agri_seed_rules_2020'])

# Seed for sampling
random.seed(42)
sampled_agri = []
if len(agri_mech_pool) >= 60:
    sampled_agri = random.sample(agri_mech_pool, 60)
    for c in sampled_agri:
        c['unofficial_translation'] = False
else:
    sampled_agri.extend(agri_mech_pool)
    for c in sampled_agri:
        c['unofficial_translation'] = False
    needed = 60 - len(sampled_agri)
    add_count = min(needed, len(agri_seed_pool))
    sampled_seed = random.sample(agri_seed_pool, add_count)
    for c in sampled_seed:
        c['unofficial_translation'] = True
    sampled_agri.extend(sampled_seed)

# Disaster: up to 100 with at most 25 per doc_id
disaster_doc_ids = [
    'disaster_sod_2019',
    'disaster_displacement_nap_2022',
    'disaster_displacement_strat_2021',
    'disaster_emcrp_rpf',
    'disaster_emcrp_esmf'
]

disaster_pool = []
for d_id in disaster_doc_ids:
    doc_cand = list(candidates_by_doc[d_id])
    take_k = min(25, len(doc_cand))
    disaster_pool.extend(random.sample(doc_cand, take_k))

if len(disaster_pool) > 100:
    sampled_disaster = random.sample(disaster_pool, 100)
else:
    sampled_disaster = list(disaster_pool)

# Education: up to 60 from edu_nep_2010
edu_pool = list(candidates_by_doc['edu_nep_2010'])
sampled_edu = random.sample(edu_pool, min(60, len(edu_pool)))

# Assign candidate IDs and write to data_v2/en_candidates.jsonl
os.makedirs('data_v2', exist_ok=True)
all_sampled = []
cand_idx = 1

for c in sampled_agri:
    c['candidate_id'] = f"en_cand_{cand_idx:04d}"
    cand_idx += 1
    all_sampled.append(c)

for c in sampled_disaster:
    c['candidate_id'] = f"en_cand_{cand_idx:04d}"
    cand_idx += 1
    all_sampled.append(c)

for c in sampled_edu:
    c['candidate_id'] = f"en_cand_{cand_idx:04d}"
    cand_idx += 1
    all_sampled.append(c)

output_jsonl_path = 'data_v2/en_candidates.jsonl'
with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    for c in all_sampled:
        row = {
            'candidate_id': c['candidate_id'],
            'doc_id': c['doc_id'],
            'domain': c['domain'],
            'source_file': c['source_file'],
            'page': c['page'],
            'text': c['text'],
            'words': c['words'],
            'footnote_glue': c['footnote_glue'],
            'unofficial_translation': c['unofficial_translation']
        }
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f"Successfully wrote {len(all_sampled)} sampled candidates to {output_jsonl_path}")

# =========================================================================
# STEP 4: GEMMA-3-1B-IT TOKENIZATION & ATTRITION REPORTING
# =========================================================================
print("\n" + "=" * 80)
print("TASK 4: TOKENIZATION & ATTRITION REPORT")
print("=" * 80)

tok = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

# Compute token counts per sampled passage
for c in all_sampled:
    token_ids = tok.encode(c['text'], add_special_tokens=False)
    c['tokens'] = len(token_ids)

print("\n--- ATTRITION REPORT PER DOC_ID ---")
print(f"{'Doc ID':<32} {'Initial':<8} {'R1(TOC)':<8} {'R2(Digit)':<9} {'R3(H/F)':<8} {'R4(Acro)':<8} {'R5(Ref)':<8} {'R6(Form)':<8} {'R7(<3S)':<8} {'R8(Char)':<8} {'R9(Dup)':<8}")
print("-" * 115)
for doc_id, att in attrition_by_doc.items():
    print(f"{doc_id:<32} {att['initial']:<8} {att['after_r1']:<8} {att['after_r2']:<8} {att['after_r3']:<8} {att['after_r4']:<8} {att['after_r5']:<8} {att['after_r6']:<8} {att['after_r7']:<8} {att['after_r8']:<8} {att['after_r9']:<8}")

print("\n--- FINAL SAMPLED COUNT & GEMMA TOKENS PER DOC_ID ---")
print(f"{'Doc ID':<32} {'Domain':<12} {'Sampled':<8} {'Mean Words':<12} {'Mean Tokens':<12} {'p90 Tokens':<12}")
print("-" * 88)

for doc_id, info in DOCS.items():
    doc_passages = [c for c in all_sampled if c['doc_id'] == doc_id]
    cnt = len(doc_passages)
    if cnt > 0:
        w_vals = [c['words'] for c in doc_passages]
        t_vals = [c['tokens'] for c in doc_passages]
        mean_w = np.mean(w_vals)
        mean_t = np.mean(t_vals)
        p90_t = np.percentile(t_vals, 90)
        print(f"{doc_id:<32} {info['domain']:<12} {cnt:<8} {mean_w:<12.1f} {mean_t:<12.1f} {p90_t:<12.1f}")
    else:
        print(f"{doc_id:<32} {info['domain']:<12} {cnt:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("\n--- FINAL SAMPLED COUNT & GEMMA TOKENS PER DOMAIN ---")
print(f"{'Domain':<16} {'Sampled':<8} {'Mean Words':<12} {'Mean Tokens':<12} {'p90 Tokens':<12}")
print("-" * 60)
for dom in ['agriculture', 'disaster', 'education']:
    dom_passages = [c for c in all_sampled if c['domain'] == dom]
    w_vals = [c['words'] for c in dom_passages]
    t_vals = [c['tokens'] for c in dom_passages]
    print(f"{dom:<16} {len(dom_passages):<8} {np.mean(w_vals):<12.1f} {np.mean(t_vals):<12.1f} {np.percentile(t_vals, 90):<12.1f}")

# =========================================================================
# STEP 5: PRINT 5 RANDOM SAMPLED PASSAGES PER DOMAIN
# =========================================================================
print("\n" + "=" * 80)
print("TASK 5: SAMPLE PASSAGE INSPECTION (5 PER DOMAIN)")
print("=" * 80)

# Set seed 42 to sample 5 passages per domain
random.seed(42)
for dom in ['agriculture', 'disaster', 'education']:
    dom_passages = [c for c in all_sampled if c['domain'] == dom]
    picked = random.sample(dom_passages, 5)
    print(f"\n==================================================================")
    print(f"DOMAIN: {dom.upper()} (5 Sampled Passages)")
    print(f"==================================================================")
    for idx, p in enumerate(picked, 1):
        print(f"\n[{dom.upper()} SAMPLE {idx}]")
        print(f"  Candidate ID:           {p['candidate_id']}")
        print(f"  Doc ID:                 {p['doc_id']}")
        print(f"  Source File:            {p['source_file']}")
        print(f"  Page:                   {p['page']}")
        print(f"  Word Count:             {p['words']}")
        print(f"  Gemma Tokens:           {p['tokens']}")
        print(f"  Footnote Glue:          {p['footnote_glue']}")
        print(f"  Unofficial Translation: {p['unofficial_translation']}")
        print(f"  Text:")
        print(f"    {p['text']}")

print("\n" + "=" * 80)
print("PIPELINE EXECUTION COMPLETE")
print("=" * 80)
