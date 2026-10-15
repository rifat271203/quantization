"""
audit/run_ocr_pipeline.py:
Performs 300 DPI OCR extraction on disaster_emcrp_esmf, disaster_emcrp_rpf, and agri_seed_rules_2020.
Saves per-page text to data_v2/ocr/<doc_id>/page_NNN.txt and data_v2/pages_bn.jsonl.
Also saves ben+eng text for the emcrp PDFs to data_v2/ocr/<doc_id>/page_NNN_ben_eng.txt.
"""

import sys
import os
import json
import csv
import time
import subprocess
import unicodedata
from pathlib import Path
import pypdfium2 as pdfium

ROOT = Path(r"d:\Tokenizer\quant-hallucination-study")
DATA_V2 = ROOT / "data_v2"
OCR_DIR = DATA_V2 / "ocr"

tess_exe = r"C:\Users\Tanjid\.gemini\antigravity-ide\brain\9665d0d8-f813-491f-9752-0404d32efc11\scratch\tesseract\tesseract.exe"
tessdata_best = r"C:\Users\Tanjid\.gemini\antigravity-ide\brain\9665d0d8-f813-491f-9752-0404d32efc11\scratch\tesseract\tessdata_best"

DOCS = [
    {
        "doc_id": "disaster_emcrp_esmf",
        "pdf_path": ROOT / "data" / "raw_sources" / "disaster" / "bn" / "esmf_emcrp_bn.pdf",
        "also_ben_eng": True,
    },
    {
        "doc_id": "disaster_emcrp_rpf",
        "pdf_path": ROOT / "data" / "raw_sources" / "disaster" / "bn" / "rpf_emcrp_bn.pdf",
        "also_ben_eng": True,
    },
    {
        "doc_id": "agri_seed_rules_2020",
        "pdf_path": ROOT / "data" / "raw_sources" / "agriculture" / "bn" / "seed_rules_2020_bn.pdf",
        "also_ben_eng": False,
    },
]

TEMP_IMG_DIR = ROOT / "audit" / "temp_ocr_imgs"
TEMP_IMG_DIR.mkdir(parents=True, exist_ok=True)

pages_records = []
t_start = time.time()

for doc_cfg in DOCS:
    doc_id = doc_cfg["doc_id"]
    pdf_path = doc_cfg["pdf_path"]
    also_ben_eng = doc_cfg["also_ben_eng"]
    
    out_dir = OCR_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    doc = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(doc)
    print(f"\n>>> Starting OCR for {doc_id} ({num_pages} pages)...")
    
    for p_idx in range(num_pages):
        page_num = p_idx + 1
        page = doc[p_idx]
        
        # 1. PDF text layer check for garble (Cn)
        pdf_text = page.get_textpage().get_text_range()
        pdf_text_layer_garbled = any(
            unicodedata.category(c) == "Cn" for c in pdf_text if 0x0980 <= ord(c) <= 0x09FF
        )
        
        # Render at 300 DPI (scale = 300 / 72 = 4.16667)
        img = page.render(scale=300 / 72).to_pil()
        temp_img_path = TEMP_IMG_DIR / f"{doc_id}_p{page_num}.png"
        img.save(temp_img_path)
        
        # Run tesseract -l ben with both txt and tsv output
        base_out = out_dir / f"page_{page_num:03d}"
        cmd_ben = [
            tess_exe,
            "--tessdata-dir", tessdata_best,
            str(temp_img_path),
            str(base_out),
            "-l", "ben",
            "txt", "tsv",
        ]
        subprocess.run(cmd_ben, capture_output=True, check=True)
        
        # Read ben text
        txt_path = out_dir / f"page_{page_num:03d}.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()
            
        # Parse TSV for confidence
        tsv_path = out_dir / f"page_{page_num:03d}.tsv"
        confs = []
        if tsv_path.exists():
            with open(tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for r in reader:
                    c = float(r.get("conf", -1))
                    if c >= 0 and r.get("text", "").strip():
                        confs.append(c)
        mean_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        
        # Also run -l ben+eng for emcrp PDFs
        if also_ben_eng:
            base_out_be = out_dir / f"page_{page_num:03d}_ben_eng"
            cmd_be = [
                tess_exe,
                "--tessdata-dir", tessdata_best,
                str(temp_img_path),
                str(base_out_be),
                "-l", "ben+eng",
                "txt",
            ]
            subprocess.run(cmd_be, capture_output=True, check=True)
            
        # Clean up temp image
        try:
            temp_img_path.unlink()
        except Exception:
            pass
            
        rec = {
            "doc_id": doc_id,
            "page": page_num,
            "text": ocr_text,
            "ocr_conf_mean": mean_conf,
            "pdf_text_layer_garbled": pdf_text_layer_garbled,
        }
        pages_records.append(rec)
        
        if page_num % 10 == 0 or page_num == num_pages:
            elapsed = time.time() - t_start
            print(f"  [{doc_id}] Processed {page_num}/{num_pages} pages (elapsed {elapsed:.1f}s, mean_conf={mean_conf}%)")

# Write data_v2/pages_bn.jsonl
pages_jsonl_path = DATA_V2 / "pages_bn.jsonl"
with open(pages_jsonl_path, "w", encoding="utf-8") as f:
    for rec in pages_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nFinished OCR pipeline! Total pages written to {pages_jsonl_path}: {len(pages_records)}")
