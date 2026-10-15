"""
audit/run_ocr_pipeline_fast.py:
Parallel 300 DPI OCR extraction on disaster_emcrp_esmf, disaster_emcrp_rpf, and agri_seed_rules_2020.
Uses multiprocessing ThreadPoolExecutor / ProcessPool to run Tesseract processes concurrently across CPU cores.
"""

import sys
import os
import json
import csv
import time
import subprocess
import unicodedata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pypdfium2 as pdfium

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\Tokenizer\quant-hallucination-study")
DATA_V2 = ROOT / "data_v2"
OCR_DIR = DATA_V2 / "ocr"
OCR_DIR.mkdir(parents=True, exist_ok=True)

tess_exe = r"C:\Users\Tanjid\.gemini\antigravity-ide\brain\9665d0d8-f813-491f-9752-0404d32efc11\scratch\tesseract\tesseract.exe"
tessdata_best = r"C:\Users\Tanjid\.gemini\antigravity-ide\brain\9665d0d8-f813-491f-9752-0404d32efc11\scratch\tesseract\tessdata_best"

DOCS = [
    {
        "doc_id": "disaster_emcrp_esmf",
        "pdf_path": str(ROOT / "data" / "raw_sources" / "disaster" / "bn" / "esmf_emcrp_bn.pdf"),
        "also_ben_eng": True,
    },
    {
        "doc_id": "disaster_emcrp_rpf",
        "pdf_path": str(ROOT / "data" / "raw_sources" / "disaster" / "bn" / "rpf_emcrp_bn.pdf"),
        "also_ben_eng": True,
    },
    {
        "doc_id": "agri_seed_rules_2020",
        "pdf_path": str(ROOT / "data" / "raw_sources" / "agriculture" / "bn" / "seed_rules_2020_bn.pdf"),
        "also_ben_eng": False,
    },
]

TEMP_IMG_DIR = ROOT / "audit" / "temp_ocr_imgs"
TEMP_IMG_DIR.mkdir(parents=True, exist_ok=True)

def process_single_page(args):
    doc_id, pdf_path, page_num, also_ben_eng = args
    out_dir = OCR_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    txt_path = out_dir / f"page_{page_num:03d}.txt"
    tsv_path = out_dir / f"page_{page_num:03d}.tsv"
    ben_eng_txt_path = out_dir / f"page_{page_num:03d}_ben_eng.txt"
    
    # Check if already processed
    need_ben = not (txt_path.exists() and tsv_path.exists())
    need_ben_eng = also_ben_eng and not ben_eng_txt_path.exists()
    
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_num - 1]
    
    # 1. Check garble
    pdf_text = page.get_textpage().get_text_range()
    pdf_text_layer_garbled = any(
        unicodedata.category(c) == "Cn" for c in pdf_text if 0x0980 <= ord(c) <= 0x09FF
    )
    
    temp_img_path = TEMP_IMG_DIR / f"{doc_id}_p{page_num}_{os.getpid()}.png"
    img_rendered = False
    
    if need_ben or need_ben_eng:
        img = page.render(scale=300 / 72).to_pil()
        img.save(temp_img_path)
        img_rendered = True
        
    if need_ben:
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
        
    if need_ben_eng:
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
        
    if img_rendered and temp_img_path.exists():
        try:
            temp_img_path.unlink()
        except Exception:
            pass
            
    with open(txt_path, "r", encoding="utf-8") as f:
        ocr_text = f.read()
        
    confs = []
    if tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                c = float(r.get("conf", -1))
                if c >= 0 and r.get("text", "").strip():
                    confs.append(c)
    mean_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
    
    return {
        "doc_id": doc_id,
        "page": page_num,
        "text": ocr_text,
        "ocr_conf_mean": mean_conf,
        "pdf_text_layer_garbled": pdf_text_layer_garbled,
    }

def main():
    tasks = []
    for doc_cfg in DOCS:
        doc_id = doc_cfg["doc_id"]
        pdf_path = doc_cfg["pdf_path"]
        also_ben_eng = doc_cfg["also_ben_eng"]
        doc = pdfium.PdfDocument(pdf_path)
        for p in range(1, len(doc) + 1):
            tasks.append((doc_id, pdf_path, p, also_ben_eng))
            
    total_tasks = len(tasks)
    print(f"Total pages to process across 3 docs: {total_tasks} (using 8 worker processes)...")
    
    results = []
    completed = 0
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_page, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 10 == 0 or completed == total_tasks:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"  Completed {completed}/{total_tasks} pages ({completed/total_tasks*100:.1f}%) [Rate: {rate:.2f} pgs/s, ETA: {eta:.1f}s]")
                
    # Sort results by doc_id and page
    doc_order = {d["doc_id"]: i for i, d in enumerate(DOCS)}
    results.sort(key=lambda r: (doc_order.get(r["doc_id"], 99), r["page"]))
    
    # Save data_v2/pages_bn.jsonl
    jsonl_path = DATA_V2 / "pages_bn.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nALL {len(results)} PAGES OCR'D AND SAVED TO {jsonl_path} successfully in {time.time()-t0:.1f}s!")

if __name__ == "__main__":
    main()
