import sys
import re
import string
import unicodedata
import pypdfium2 as pdfium
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014
allowed_chars_old = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash

# New allowed chars:
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

def run_pipeline(doc_id, path, info, normalize=False):
    pdf = pdfium.PdfDocument(path)
    pages_text = []
    for page in pdf:
        t = page.get_textpage().get_text_range()
        if normalize:
            t = normalize_text(t)
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
                            
    allowed = allowed_chars_new if normalize else allowed_chars_old
    
    # 9 exclusion rules
    r1 = []
    for c in raw_candidates:
        if 'table of contents' in c['section'].lower() or c['section'].strip().lower() == 'contents':
            continue
        # dot leaders: 3 or more in a row or followed by page number
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
    return r9

for doc_id, info in DOCS.items():
    c_old = run_pipeline(doc_id, info['path'], info, normalize=False)
    c_new = run_pipeline(doc_id, info['path'], info, normalize=True)
    print(f'{doc_id}: old={len(c_old)}, new={len(c_new)}, diff={len(c_new)-len(c_old)}')
