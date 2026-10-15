import sys
import json
import re
from pathlib import Path
import pypdf

ROOT = Path("d:/Tokenizer/quant-hallucination-study")

DOCS_INFO = [
    {
        'doc_id': 'agri_mechanization_2020',
        'bn_pdf': 'data/raw_sources/agriculture/bn/agri_mechanization_policy_2020_bn.pdf',
        'en_pdf': 'data/raw_sources/agriculture/en/agri_mechanization_policy_2020_en.pdf',
        'bn_clean': None,
        'en_clean': None,
    },
    {
        'doc_id': 'agri_seed_rules_2020',
        'bn_pdf': 'data/raw_sources/agriculture/bn/seed_rules_2020_bn.pdf',
        'en_pdf': 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf',
        'bn_clean': 'data/raw_sources/converted_text/agri_seed_rules_2020_converted.txt',
        'en_clean': None,
    },
    {
        'doc_id': 'disaster_sod_2019',
        'bn_pdf': 'data/raw_sources/disaster/bn/sod_2019_bn.pdf',
        'en_pdf': 'data/raw_sources/disaster/en/sod_2019_en.pdf',
        'bn_clean': None,
        'en_clean': None,
    },
    {
        'doc_id': 'disaster_displacement_nap_2022',
        'bn_pdf': 'data/raw_sources/disaster/bn/nap_displacement_2022_2042_bn.pdf',
        'en_pdf': 'data/raw_sources/disaster/en/nap_displacement_2022_2042_en.pdf',
        'bn_clean': None,
        'en_clean': None,
    },
    {
        'doc_id': 'disaster_displacement_strat_2021',
        'bn_pdf': 'data/raw_sources/disaster/bn/strategy_displacement_2021_bn.pdf',
        'en_pdf': 'data/raw_sources/disaster/en/strategy_displacement_2021_en.pdf',
        'bn_clean': 'data/raw_sources/converted_text/disaster_displacement_strat_2021_converted.txt',
        'en_clean': None,
    },
    {
        'doc_id': 'disaster_emcrp_rpf',
        'bn_pdf': 'data/raw_sources/disaster/bn/rpf_emcrp_bn.pdf',
        'en_pdf': 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf',
        'bn_clean': None,
        'en_clean': None,
    },
    {
        'doc_id': 'disaster_emcrp_esmf',
        'bn_pdf': 'data/raw_sources/disaster/bn/esmf_emcrp_bn.pdf',
        'en_pdf': 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf',
        'bn_clean': None,
        'en_clean': None,
    },
    {
        'doc_id': 'edu_nep_2010',
        'bn_pdf': 'data/raw_sources/education/bn/national_education_policy_2010_bn.pdf',
        'en_pdf': 'data/raw_sources/education/en/national_education_policy_2010_en.pdf',
        'bn_clean': 'data/raw_sources/ocr_samples/national_education_policy_2010_bn',
        'en_clean': None,
    },
]

summary_text = (ROOT / "data/raw_sources/SUMMARY.md").read_text(encoding="utf-8")

def get_pdf_info(rel_path):
    p = ROOT / rel_path
    if not p.exists():
        return None, 0, "FILE NOT FOUND"
    try:
        reader = pypdf.PdfReader(str(p))
        num_pages = len(reader.pages)
        p1 = reader.pages[0].extract_text() or ""
        return str(rel_path).replace("\\", "/"), num_pages, p1[:400]
    except Exception as e:
        return str(rel_path).replace("\\", "/"), 0, f"ERROR: {e}"

report_lines = []
report_lines.append("="*80)
report_lines.append("TASK 2: DOCUMENT VERSIONS AUDIT (ALL 8 DOCS)")
report_lines.append("="*80)

for d in DOCS_INFO:
    doc_id = d['doc_id']
    bn_file, bn_pages, bn_p1 = get_pdf_info(d['bn_pdf'])
    en_file, en_pages, en_p1 = get_pdf_info(d['en_pdf'])
    
    report_lines.append(f"\n--- {doc_id} ---")
    report_lines.append(f"BN Source Path: {bn_file}")
    report_lines.append(f"EN Source Path: {en_file}")
    report_lines.append(f"BN Page Count : {bn_pages}")
    report_lines.append(f"EN Page Count : {en_pages}")
    report_lines.append(f"BN Page 1 (first 400 chars, ascii-escaped):")
    report_lines.append(f"  {ascii(bn_p1)}")
    report_lines.append(f"EN Page 1 (first 400 chars, ascii-escaped):")
    report_lines.append(f"  {ascii(en_p1)}")
    
    # Check SUMMARY.md for this document
    summary_lines_for_doc = []
    # Match lines containing doc_id or specific doc title
    for line in summary_text.splitlines():
        if doc_id in line:
            summary_lines_for_doc.append(line)
            
    report_lines.append("SUMMARY.md mentions:")
    for sm in summary_lines_for_doc[:6]:
        report_lines.append(f"  {ascii(sm)}")

out_path = ROOT / "audit/task2_report.txt"
out_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Wrote full report to {out_path}")
