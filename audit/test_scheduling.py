import sys
import os
import re
import random
import pypdfium2 as pdfium
from collections import Counter, defaultdict
from test_task2_task3_exclusions import (
    DOCS, normalize_text_v2, count_sentences, split_into_sentences,
    extract_paragraphs, has_footnote_glue, compute_flags_v2,
    check_excl_a, check_excl_b, check_excl_c, check_excl_d, allowed_chars
)

sys.stdout.reconfigure(encoding="utf-8")

scheduling_stats = {}

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
        
    # Extra exclusions
    clean_cands = []
    for c in r9:
        if c['table_like']:
            continue
        if check_excl_a(c) or check_excl_b(c) or check_excl_c(c) or check_excl_d(c):
            continue
        clean_cands.append(c)
        
    # Interval scheduling:
    # Sort by end offset: (page, end_word, start_word)
    # Take the earliest-ending passage that starts after the last chosen end
    clean_cands.sort(key=lambda c: (c['page'], c['end_word'], c['start_word']))
    
    chosen_intervals = []
    last_page = -1
    last_end = -1
    for c in clean_cands:
        if c['page'] > last_page:
            chosen_intervals.append(c)
            last_page = c['page']
            last_end = c['end_word']
        elif c['page'] == last_page:
            if c['start_word'] >= last_end:
                chosen_intervals.append(c)
                last_end = c['end_word']
                
    scheduling_stats[doc_id] = {
        'total_r9': len(r9),
        'clean_count': len(clean_cands),
        'max_non_overlapping': len(chosen_intervals),
        'chosen': chosen_intervals
    }
    print(f"{doc_id:<32}: clean={len(clean_cands):<4} max_non_overlapping={len(chosen_intervals):<4}")
