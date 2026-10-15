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

# Prompt Step 1: "basic Latin, digits, whitespace and common punctuation (including curly quotes and the em dash)"
allowed_base = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash
allowed_with_endash = allowed_base | en_dash

def count_sentences(text):
    cleaned = re.sub(r'\b(e\.g\.|i\.e\.|viz\.|etc\.|al\.|Mr\.|Mrs\.|Dr\.|No\.|Fig\.|Vol\.|pp\.|sec\.)', 
                     lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d+\.\d+\b', lambda m: m.group(0).replace('.', '<DOT>'), cleaned)
    cleaned = re.sub(r'(?<=\s)\d+\.(?=\s)', lambda m: m.group(0).replace('.', '<DOT>'), cleaned)
    # Sentence end: terminal punctuation followed by space or end of text
    sentences = re.split(r'[.!?]+[\'"’”)]?(?:\s+|$)', cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def has_footnote_glue(text):
    # Footnote digit glued to a number: e.g. 727,0001 or 919,0002
    return bool(re.search(r'\b\d{1,3},\d{4,}\b', text))

def extract_page_paragraphs_with_lines(page_text):
    """
    Extract paragraphs, keeping track of the individual raw lines that make up each paragraph.
    """
    raw_blocks = [b.strip() for b in re.split(r'\n\s*\n', page_text) if b.strip()]
    paragraphs = [] # list of (para_text, list_of_raw_lines)
    
    for block in raw_blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        cur_lines = [lines[0]]
        for line in lines[1:]:
            prev = cur_lines[-1]
            is_list = bool(re.match(r'^(?:\(?\d+[\.\)]|\(?[a-zA-Z]\)|\d+\.\d+(\.\d+)*|[•▪\-\*]|Chapter\s+\d+|Section\s+\d+)\s+', line))
            prev_end = bool(re.search(r'[\.\?!:]["\'\)]?$', prev))
            starts_upper = bool(re.match(r'^[A-Z0-9"\']', line))
            if is_list or (prev_end and starts_upper and len(prev) < 65):
                paragraphs.append((' '.join(cur_lines), list(cur_lines)))
                cur_lines = [line]
            else:
                cur_lines.append(line)
        if cur_lines:
            paragraphs.append((' '.join(cur_lines), list(cur_lines)))
    return paragraphs

def run_test(use_endash_in_allowed=False):
    allowed = allowed_with_endash if use_endash_in_allowed else allowed_base
    print(f"\n=======================================================")
    print(f"RUNNING TEST (allowed includes en-dash: {use_endash_in_allowed})")
    print(f"=======================================================")

    for doc_id, path in DOCS.items():
        pdf = pdfium.PdfDocument(path)
        
        # 1. Collect repeated lines (> 3 pages)
        line_page_counts = Counter()
        doc_page_paras = []
        for p_no, page in enumerate(pdf):
            p_text = page.get_textpage().get_text_range()
            p_lines = set(l.strip() for l in p_text.splitlines() if l.strip())
            for l in p_lines:
                line_page_counts[l] += 1
            paras = extract_page_paragraphs_with_lines(p_text)
            doc_page_paras.append((p_no + 1, paras))
            
        repeated_lines = {l for l, c in line_page_counts.items() if c > 3 and len(l) > 2}
        
        # 2. Build candidate passages
        # On each page, merge neighbouring paragraphs from the same section/page into 80-180 words
        candidates = []
        for page_num, paras in doc_page_paras:
            # Group paras into sections
            sections = []
            cur_sec = "default"
            cur_p = []
            for p_text, p_lines in paras:
                # check if section header
                words = p_text.split()
                is_heading = False
                if len(words) <= 12 and not p_text.endswith('.'):
                    if re.match(r'^(?:Chapter\s+\d+|Section\s+\d+|\d+(\.\d+)*\s+[A-Z]|[A-Z\s]{4,})', p_text):
                        is_heading = True
                if is_heading:
                    if cur_p:
                        sections.append((cur_sec, cur_p))
                        cur_p = []
                    cur_sec = p_text
                else:
                    cur_p.append((p_text, p_lines))
            if cur_p:
                sections.append((cur_sec, cur_p))
                
            for sec_name, sec_paras in sections:
                n = len(sec_paras)
                for i in range(n):
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
                                candidates.append({
                                    'doc_id': doc_id,
                                    'page': page_num,
                                    'section': sec_name,
                                    'text': full_txt,
                                    'lines': list(cur_plines),
                                    'words': cur_w,
                                    'footnote_glue': has_footnote_glue(full_txt)
                                })
                        elif cur_w > 180:
                            break
                            
        # Now apply the 9 exclusion rules in order and count:
        total_initial = len(candidates)
        
        # Rule 1: TOC lines (dot leaders or trailing page numbers)
        # Dot leaders: \.{3,} or …{2,} or \.\s*\.\s*\. or index listing lines
        def is_toc_line(line):
            if re.search(r'(\.{3,}|…{2,}|\.\s*\.\s*\.)', line):
                return True
            if re.search(r'^\s*(?:Chapter|\d+\.|\w+).*?\s+\d+$', line) and len(line) < 60:
                # possible TOC entry
                pass
            return False
            
        r1_survivors = []
        for c in candidates:
            # check if any line has dot leaders
            has_toc = any(is_toc_line(l) for l in c['lines']) or ('table of contents' in c['section'].lower()) or ('contents' == c['section'].strip().lower())
            if not has_toc:
                r1_survivors.append(c)
                
        # Rule 2: lines where digits exceed 25 percent of characters
        def has_heavy_digits(line):
            l = line.strip()
            if not l:
                return False
            digit_cnt = sum(ch.isdigit() for ch in l)
            return (digit_cnt / len(l)) > 0.25
            
        r2_survivors = []
        for c in r1_survivors:
            if not any(has_heavy_digits(l) for l in c['lines']):
                r2_survivors.append(c)
                
        # Rule 3: headers and footers repeated on more than 3 pages
        r3_survivors = []
        for c in r2_survivors:
            # check if any line in c['lines'] is in repeated_lines
            # note: ignore single numbers / short strings if needed, or exact match
            if not any(l in repeated_lines for l in c['lines']):
                r3_survivors.append(c)
                
        # Rule 4: acronym lists
        def is_acronym_candidate(c):
            sec = c['section'].lower()
            if any(term in sec for term in ['acronym', 'abbreviation']):
                return True
            # check if majority of lines match ACRONYM: Description or ACRONYM Description
            acro_lines = sum(1 for l in c['lines'] if re.match(r'^[A-Z0-9]{2,}\s+[A-Z]', l))
            if len(c['lines']) > 3 and acro_lines / len(c['lines']) > 0.5:
                return True
            return False
            
        r4_survivors = []
        for c in r3_survivors:
            if not is_acronym_candidate(c):
                r4_survivors.append(c)
                
        # Rule 5: reference or bibliography sections
        def is_reference_section(c):
            sec = c['section'].lower()
            return any(term in sec for term in ['reference', 'bibliography', 'literature cited'])
            
        r5_survivors = []
        for c in r4_survivors:
            if not is_reference_section(c):
                r5_survivors.append(c)
                
        # Rule 6: forms with blank fill-in lines
        def has_blank_fill_in(c):
            for l in c['lines']:
                if re.search(r'(_{3,}|\.{5,})', l):
                    return True
            return False
            
        r6_survivors = []
        for c in r5_survivors:
            if not has_blank_fill_in(c):
                r6_survivors.append(c)
                
        # Rule 7: passages with fewer than 3 sentences
        r7_survivors = []
        for c in r6_survivors:
            s_cnt = count_sentences(c['text'])
            if s_cnt >= 3:
                r7_survivors.append(c)
                
        # Rule 8: any passage containing characters flagged as unusual in step 1
        r8_survivors = []
        for c in r7_survivors:
            has_unusual = any(ch not in allowed for ch in c['text'])
            if not has_unusual:
                r8_survivors.append(c)
                
        # Rule 9: duplicates
        r9_survivors = []
        seen_texts = set()
        for c in r8_survivors:
            norm_txt = ' '.join(c['text'].split())
            if norm_txt not in seen_texts:
                seen_texts.add(norm_txt)
                r9_survivors.append(c)
                
        print(f"=== {doc_id} ===")
        print(f"  Initial candidates: {total_initial}")
        print(f"  After Rule 1 (TOC): {len(r1_survivors)} (-{total_initial - len(r1_survivors)})")
        print(f"  After Rule 2 (Digits > 25%): {len(r2_survivors)} (-{len(r1_survivors) - len(r2_survivors)})")
        print(f"  After Rule 3 (Repeated H/F): {len(r3_survivors)} (-{len(r2_survivors) - len(r3_survivors)})")
        print(f"  After Rule 4 (Acronyms): {len(r4_survivors)} (-{len(r3_survivors) - len(r4_survivors)})")
        print(f"  After Rule 5 (References): {len(r5_survivors)} (-{len(r4_survivors) - len(r5_survivors)})")
        print(f"  After Rule 6 (Blank forms): {len(r6_survivors)} (-{len(r5_survivors) - len(r6_survivors)})")
        print(f"  After Rule 7 (< 3 sentences): {len(r7_survivors)} (-{len(r6_survivors) - len(r7_survivors)})")
        print(f"  After Rule 8 (Unusual chars): {len(r8_survivors)} (-{len(r7_survivors) - len(r8_survivors)})")
        print(f"  After Rule 9 (Duplicates): {len(r9_survivors)} (-{len(r8_survivors) - len(r9_survivors)})")
        print(f"  Footnote glue count in survivors: {sum(c['footnote_glue'] for c in r9_survivors)}")

run_test(use_endash_in_allowed=False)
run_test(use_endash_in_allowed=True)
