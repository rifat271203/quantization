import os
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
LLAMA_DIR = ROOT / "llama.cpp"
LLAMA_DIR.mkdir(exist_ok=True)

TAG = "b11577"
CUDA_ZIP = f"llama-{TAG}-bin-win-cuda-12.4-x64.zip"
CUDART_ZIP = "cudart-llama-bin-win-cuda-12.4-x64.zip"

BASE_URL = f"https://github.com/ggerganov/llama.cpp/releases/download/{TAG}"

def download_and_extract(filename):
    url = f"{BASE_URL}/{filename}"
    zip_path = LLAMA_DIR / filename
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting {filename} to {LLAMA_DIR} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(LLAMA_DIR)
    zip_path.unlink()
    print(f"Done with {filename}!")

print(f"Setting up llama.cpp {TAG} in {LLAMA_DIR} ...")
download_and_extract(CUDART_ZIP)
download_and_extract(CUDA_ZIP)

# Also download convert_hf_to_gguf.py from the main repo
converter_url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
print(f"Downloading convert_hf_to_gguf.py ...")
urllib.request.urlretrieve(converter_url, LLAMA_DIR / "convert_hf_to_gguf.py")

print("\nVerifying executables in llama.cpp directory:")
for f in LLAMA_DIR.iterdir():
    if f.suffix in [".exe", ".py"]:
        print(f" - {f.name}")
