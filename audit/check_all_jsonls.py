import os
import re
import json
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("d:/Tokenizer/quant-hallucination-study")

# Literal IDs from build_study_eval_sets.py:
LITERAL_PAIRS = [
    ('agri_mech_01', 77, 'agriculture', 'agri_mechanization_2020'),
    ('agri_mech_02', 91, 'agriculture', 'agri_mechanization_2020'),
    ('agri_seed_01', 107, 'agriculture', 'agri_seed_rules_2020'),
    ('agri_seed_02', 121, 'agriculture', 'agri_seed_rules_2020'),
    ('disaster_sod_01', 137, 'disaster', 'disaster_sod_2019'),
    ('disaster_sod_02', 151, 'disaster', 'disaster_sod_2019'),
    ('disaster_nap_01', 167, 'disaster', 'disaster_displacement_nap_2022'),
    ('disaster_strat_01', 183, 'disaster', 'disaster_displacement_strat_2021'),
    ('disaster_emcrp_01', 199, 'disaster', 'disaster_emcrp_rpf'),
    ('disaster_emcrp_02', 215, 'disaster', 'disaster_emcrp_esmf'),
    ('edu_nep_01', 232, 'education', 'edu_nep_2010'),
    ('edu_nep_02', 247, 'education', 'edu_nep_2010'),
]

LITERAL_ACT_ITEMS = [
    ('disaster_act_01', 300, 'disaster', 'disaster_act_2012'),
    ('disaster_act_02', 304, 'disaster', 'disaster_act_2012'),
    ('disaster_act_03', 308, 'disaster', 'disaster_act_2012'),
    ('disaster_act_04', 312, 'disaster', 'disaster_act_2012'),
    ('disaster_act_05', 316, 'disaster', 'disaster_act_2012'),
    ('disaster_act_06', 320, 'disaster', 'disaster_act_2012'),
    ('disaster_act_07', 324, 'disaster', 'disaster_act_2012'),
    ('disaster_act_low_01', 348, 'disaster', 'disaster_act_2012'),
    ('disaster_act_low_04', 360, 'disaster', 'disaster_act_2012'),
]

LITERAL_NCTB_ITEMS = [
    ('edu_sci_clean_01', 376, 'education', 'edu_nctb_science_class9'),
    ('edu_sci_clean_02', 381, 'education', 'edu_nctb_science_class9'),
    ('edu_sci_clean_03', 386, 'education', 'edu_nctb_science_class9'),
    ('edu_sci_low_01', 392, 'education', 'edu_nctb_science_class9'),
    ('edu_sci_low_02', 397, 'education', 'edu_nctb_science_class9'),
    ('edu_sci_low_03', 402, 'education', 'edu_nctb_science_class9'),
]

all_literals = LITERAL_PAIRS + LITERAL_ACT_ITEMS + LITERAL_NCTB_ITEMS
all_base_ids = {x[0] for x in all_literals}
print(f"Total base literal IDs defined in build_study_eval_sets.py: {len(all_base_ids)}")

# Check every jsonl file in ROOT and ROOT / "data"
jsonl_files = list(ROOT.glob("*.jsonl")) + list((ROOT / "data").glob("*.jsonl"))
print(f"\nChecking occurrences across {len(jsonl_files)} jsonl files:")

for jf in jsonl_files:
    total = 0
    matched = 0
    matched_ids = []
    by_doc = Counter()
    with open(jf, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            rid = row.get("id", "")
            base_rid = rid.split("__")[0]
            if base_rid in all_base_ids:
                matched += 1
                doc = row.get("doc_id", "none")
                by_doc[doc] += 1
                matched_ids.append((rid, doc))
    if matched > 0:
        print(f"\nFILE: {jf.relative_to(ROOT)} -> {matched} / {total} rows matched:")
        for doc, c in sorted(by_doc.items()):
            print(f"   {doc}: {c}")
        print(f"   Sample IDs: {[x[0] for x in matched_ids[:4]]}")
    else:
        print(f"FILE: {jf.relative_to(ROOT)} -> 0 / {total} matched")

