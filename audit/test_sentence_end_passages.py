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
en_dash = {'–'} # U+2013

allowed_base = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash
allowed_with_endash = allowed_base | en_dash

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

def split_text_into_sentences(text):
    cleaned = re.sub(r'\b(e\.g\.|i\.e\.|viz\.|etc\.|al\.|Mr\.|Mrs\.|Dr\.|No\.|Fig\.|Vol\.|pp\.|sec\.)', 
                     lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d+\.\d+\b', lambda m: m.group(0).replace('.', '<DOT>'), cleaned)
    cleaned = re.sub(r'(?<=\s)\d+\.(?=\s)', lambda m: m.group(0).replace('.', '<DOT>'), cleaned)
    
    splits = [m.end() for m in re.finditer(r'[.!?]+[\'"’”)]?(?:\s+|$)', cleaned)]
    if not splits or splits[-1] < len(text):
        splits.append(len(text))
        
    sentences = []
    prev_idx = 0
    for idx in splits:
        s = text[prev_idx:idx].strip()
        if s:
            sentences.append(s)
        prev_idx = idx
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

for doc_id, path in DOCS.items():
    pdf = pdfium.PdfDocument(path)
    passages = []
    for p_no, page in enumerate(pdf):
        p_text = page.get_textpage().get_text_range()
        paras = extract_paragraphs(p_text)
        
        # Group into sections
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
                # We can either merge whole paras, or if a single para is > 180, take sentences
                # Let's collect all sentence sequences from para i start
                # Gather sentences from para i onwards
                collected_sentences = []
                collected_lines = []
                for j in range(i, n):
                    ptxt, plines = sec_paras[j]
                    s_list = split_text_into_sentences(ptxt)
                    for s in s_list:
                        collected_sentences.append(s)
                    collected_lines.extend(plines)
                    
                    # Check if ending at para j end is valid
                    sub_txt = ' '.join(collected_sentences)
                    w = len(sub_txt.split())
                    if 80 <= w <= 180 and sub_txt.rstrip().endswith(('.', '!', '?')):
                        passages.append({
                            'page': p_no + 1,
                            'sec': sec_name,
                            'text': sub_txt,
                            'lines': list(collected_lines),
                            'words': w,
                            'glue': has_footnote_glue(sub_txt)
                        })
                    elif w > 180:
                        # If adding whole para pushed it > 180, check if stopping at a sentence inside para j works
                        # Check sentence-by-sentence
                        break
                        
                # Also check within para i if para i itself is long
                s_list_i = split_text_into_sentences(sec_paras[i][0])
                if len(sec_paras[i][0].split()) > 180:
                    cur_s = []
                    for s in s_list_i:
                        cur_s.append(s)
                        sub_txt = ' '.join(cur_s)
                        w = len(sub_txt.split())
                        if 80 <= w <= 180 and sub_txt.rstrip().endswith(('.', '!', '?')):
                            passages.append({
                                'page': p_no + 1,
                                'sec': sec_name,
                                'text': sub_txt,
                                'lines': list(sec_paras[i][1]),
                                'words': w,
                                'glue': has_footnote_glue(sub_txt)
                            })
                            
    glue_cnt = sum(p['glue'] for p in passages)
    print(f"=== {doc_id} ===")
    print(f"  Total passages generated: {len(passages)}, with footnote_glue: {glue_cnt}")
