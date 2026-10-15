import sys
import os
import re
from test_run_pipeline_norm import run_pipeline, DOCS, split_into_sentences
from test_task3_task4 import compute_flags

sys.stdout.reconfigure(encoding="utf-8")

cands = run_pipeline('agri_mechanization_2020', DOCS['agri_mechanization_2020']['path'], DOCS['agri_mechanization_2020'], normalize=True)
for i, c in enumerate(cands[:10]):
    compute_flags(c)
    print(f"Cand {i+1}: table_like={c['table_like']}, short={c['short_line_fraction']:.2f}, qm={c['question_marks']}, sec={c['section_markers']}, cap={c['capital_start_fraction']:.2f}")
