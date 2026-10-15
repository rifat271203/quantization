import sys
import os
import string
import unicodedata
from collections import Counter
import pypdfium2 as pdfium

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

# Definition A: basic Latin, digits, whitespace, string.punctuation, curly quotes, em dash (U+2014)
curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014
en_dash = {'–'} # U+2013

allowed_base = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash
allowed_with_endash = allowed_base | en_dash

for doc_id, path in DOCS.items():
    pdf = pdfium.PdfDocument(path)
    total_chars = 0
    cnt_base = Counter()
    cnt_endash = Counter()
    pages = len(pdf)
    words = 0
    for page in pdf:
        text = page.get_textpage().get_text_range()
        total_chars += len(text)
        words += len(text.split())
        for ch in text:
            if ch not in allowed_base:
                cnt_base[ch] += 1
            if ch not in allowed_with_endash:
                cnt_endash[ch] += 1
    print(f"=== {doc_id} ===")
    print(f"  Pages: {pages}, Total Words: {words}, Total Chars: {total_chars}")
    print(f"  Outside base: count={sum(cnt_base.values())}, frac={sum(cnt_base.values())/total_chars:.6e}")
    print(f"  Outside with en-dash: count={sum(cnt_endash.values())}, frac={sum(cnt_endash.values())/total_chars:.6e}")
    print(f"  Top 10 outside base:")
    for ch, cnt in cnt_base.most_common(10):
        try:
            name = unicodedata.name(ch)
        except Exception:
            name = "UNKNOWN"
        print(f"    {repr(ch)} (U+{ord(ch):04X}, {name}): {cnt}")
