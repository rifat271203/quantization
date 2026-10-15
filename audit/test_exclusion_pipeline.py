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

# Step 1 Character sets:
curly_quotes = {'“', '”', '‘', '’'}
em_dash = {'—'} # U+2014
en_dash = {'–'} # U+2013

# Definition 1: Strict (no en-dash in allowed)
allowed_chars_strict = set(string.ascii_letters) | set(string.digits) | set(string.whitespace) | set(string.punctuation) | curly_quotes | em_dash
# Definition 2: Common punctuation including en-dash
allowed_chars_with_endash = allowed_chars_strict | en_dash

print("Testing exclusion pipeline...")
