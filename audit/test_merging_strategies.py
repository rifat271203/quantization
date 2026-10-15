import sys
import os
import re
import pypdfium2 as pdfium

sys.stdout.reconfigure(encoding="utf-8")

from test_paragraph_splitting import split_page_into_paragraphs, DOCS

def get_page_sections_and_paras(page_text):
    """
    Identify sections and paragraphs on a page.
    A section header is a short line (< 80 chars) that matches heading patterns:
    e.g. '3.0 Challenges...', '3.1 Small Farms...', 'Chapter 1: ...', '1. Education...',
    or standalone all-caps / title-case headers.
    """
    raw_paras = split_page_into_paragraphs(page_text)
    
    sections = [] # list of (section_name, list_of_paras)
    cur_section = "default"
    cur_paras = []
    
    for p in raw_paras:
        # Check if p is a section header:
        # short line, no sentence ending period, or explicit section numbering
        is_heading = False
        words = p.split()
        if len(words) <= 12 and not p.endswith('.'):
            if re.match(r'^(?:Chapter\s+\d+|Section\s+\d+|\d+(\.\d+)*\s+[A-Z]|[A-Z\s]{4,})', p):
                is_heading = True
        if is_heading:
            if cur_paras:
                sections.append((cur_section, cur_paras))
                cur_paras = []
            cur_section = p
        else:
            cur_paras.append(p)
            
    if cur_paras:
        sections.append((cur_section, cur_paras))
        
    return sections

# Test strategy 1: Whole paragraphs only (start at para i, end at para j)
# Test strategy 2: Sliding window of whole paragraphs
# Test strategy 3: Whole paragraphs + sentence-level stop if para exceeds 180
for doc_id, path in DOCS.items():
    pdf = pdfium.PdfDocument(path)
    passages_whole = []
    passages_non_overlapping = []
    
    for p_no, page in enumerate(pdf):
        text = page.get_textpage().get_text_range()
        sections = get_page_sections_and_paras(text)
        
        for sec_name, paras in sections:
            # sliding window across paras
            n = len(paras)
            for i in range(n):
                cur_words = 0
                cur_text = []
                for j in range(i, n):
                    p_text = paras[j]
                    p_w = len(p_text.split())
                    cur_words += p_w
                    cur_text.append(p_text)
                    if 80 <= cur_words <= 180:
                        # Check if ends at sentence end (. ! ?)
                        full_txt = ' '.join(cur_text)
                        if full_txt.rstrip().endswith(('.', '!', '?')):
                            passages_whole.append((p_no + 1, sec_name, full_txt, cur_words))
                    elif cur_words > 180:
                        break
                        
    print(f"=== {doc_id} ===")
    print(f"  Whole paragraph merged passages (80-180w): {len(passages_whole)}")
