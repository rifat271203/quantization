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
    text = text.replace('\ufffe', '-')
    for b in ['\uf0b7', '\uf0a7', '\uf0d8', '\uf0fc']:
        text = text.replace(b, '\u2022')
    return text

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
    
    # 4. section_markers at start of a line or sentence (threshold >= 8)
    # Check start of lines and start of sentences for patterns like:
    # 6.4.1, 7.0, 1.1, Section N
    sec_markers_cnt = 0
    # Line starts
    for l in lines:
        l_s = l.strip().lstrip('•▪■*-–—\u2022 ').strip()
        if re.match(r'^(?:Section\s+\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*)\b', l_s, re.I):
            sec_markers_cnt += 1
    # Check sentence starts that do not start a line
    s_list = split_into_sentences(text)
    # 5. capital_start_fraction after stripping leading clause marker
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
                cap_starts += 1 # if only clause marker was present
        cap_frac = cap_starts / len(s_list)
    else:
        cap_frac = 1.0
        
    # New table_like rule:
    # short_line_fraction > 0.4 or question_marks >= 3 or section_markers >= 8 or capital_start_fraction < 0.7
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
        # check first sentence or first line
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

print("Running pipeline on all docs with new normalization, table_like flags, and extra exclusions...")

for doc_id, info in DOCS.items():
    pdf = pdfium.PdfDocument(info['path'])
    pages_text = [normalize_text_v2(p.get_textpage().get_text_range()) for p in pdf]
    
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
            
    # Compute flags v2
    for c in r9:
        compute_flags_v2(c)
        
    cnt_table = sum(1 for c in r9 if c['table_like'])
    cnt_short = sum(1 for c in r9 if c['t_short'])
    cnt_qm = sum(1 for c in r9 if c['t_qm'])
    cnt_sec = sum(1 for c in r9 if c['t_sec'])
    cnt_cap = sum(1 for c in r9 if c['t_cap'])
    
    # Extra exclusions on all r9 candidates (or non-table candidates)
    ex_a = sum(1 for c in r9 if check_excl_a(c))
    ex_b = sum(1 for c in r9 if check_excl_b(c))
    ex_c = sum(1 for c in r9 if check_excl_c(c))
    ex_d = sum(1 for c in r9 if check_excl_d(c))
    
    # Candidates passing table_like AND extra exclusions
    clean_cands = [c for c in r9 if not c['table_like'] and not (check_excl_a(c) or check_excl_b(c) or check_excl_c(c) or check_excl_d(c))]
    
    print(f"\n{doc_id} (total={len(r9)}):")
    print(f"  table_like={cnt_table} (short={cnt_short}, qm={cnt_qm}, sec={cnt_sec}, cap={cnt_cap})")
    print(f"  extra_excl: a(dup_sent)={ex_a}, b(brackets>25%)={ex_b}, c(lower/bare_bullet)={ex_c}, d(short_sent>3)={ex_d}")
    print(f"  Clean candidates (non-table & non-excluded): {len(clean_cands)}")
