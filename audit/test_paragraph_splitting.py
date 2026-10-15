import sys
import os
import re
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

def split_page_into_paragraphs(page_text):
    """
    Split page text into paragraphs:
    1. First split by blank lines (\n\s*\n).
    2. Within blocks that have multiple lines without blank lines:
       Determine if a line break is a hard-wrapped line or a new paragraph/list-item/heading.
    """
    raw_blocks = [b.strip() for b in re.split(r'\n\s*\n', page_text) if b.strip()]
    paragraphs = []
    
    for block in raw_blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
            
        cur_para = [lines[0]]
        for line in lines[1:]:
            prev_line = cur_para[-1]
            
            # Indicators that `line` starts a new paragraph / list item / section:
            # - Starts with list marker: (1), (a), 1., 1.1, [A-Z]., bullet, etc.
            is_list_or_heading = bool(re.match(r'^(?:\(?\d+[\.\)]|\(?[a-zA-Z]\)|\d+\.\d+(\.\d+)*|[•▪\-\*]|Chapter\s+\d+|Section\s+\d+)\s+', line))
            
            # - Previous line ends with terminal punctuation (. ? ! : or ." etc.) AND this line starts with uppercase
            prev_ends_sentence = bool(re.search(r'[\.\?!:]["\'\)]?$', prev_line))
            starts_upper = bool(re.match(r'^[A-Z0-9"\']', line))
            
            # If it's a list/heading OR (prev ends sentence and line starts with upper and prev line looks like end of block/short line)
            # In PDFs, hard wraps can happen after period only if sentence ended mid-paragraph, but if prev_line is short (< 60 chars) or list item, it's a new paragraph.
            is_new = False
            if is_list_or_heading:
                is_new = True
            elif prev_ends_sentence and starts_upper and len(prev_line) < 65:
                is_new = True
                
            if is_new:
                paragraphs.append(' '.join(cur_para))
                cur_para = [line]
            else:
                cur_para.append(line)
        if cur_para:
            paragraphs.append(' '.join(cur_para))
            
    return paragraphs

for doc_id, path in DOCS.items():
    pdf = pdfium.PdfDocument(path)
    doc_paras = []
    for p_no, page in enumerate(pdf):
        text = page.get_textpage().get_text_range()
        paras = split_page_into_paragraphs(text)
        for p in paras:
            doc_paras.append((p_no + 1, p))
            
    words_per_para = [len(p[1].split()) for p in doc_paras]
    print(f"=== {doc_id} ===")
    print(f"  Total paragraphs: {len(doc_paras)}")
    print(f"  Word count min: {min(words_per_para) if words_per_para else 0}, "
          f"max: {max(words_per_para) if words_per_para else 0}, "
          f"mean: {sum(words_per_para)/len(words_per_para) if words_per_para else 0:.1f}")
    print(f"  Paras 80-180 words: {sum(80 <= w <= 180 for w in words_per_para)}")
    print(f"  Paras < 80 words: {sum(w < 80 for w in words_per_para)}")
    print(f"  Paras > 180 words: {sum(w > 180 for w in words_per_para)}")
