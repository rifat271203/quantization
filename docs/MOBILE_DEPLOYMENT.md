# Mobile On-Device Deployment Guide (4-bit Compressed Model)

This guide documents the procedure for deploying and running the project's fine-tuned, 4-bit quantized ~1B parameter model (`gemma3-1b-bilingual-q4_k_m.gguf`) on a mobile Android device.

---

## 1. Prerequisites & Target Specifications

- **Target Architecture**: ARM64 (aarch64) Android smartphone (Android 10+ recommended).
- **RAM Requirement**: ~1.2 GiB to 1.8 GiB free RAM (model size is ~700 MB – 1 GB in 4-bit Q4_K_M).
- **Inference Runtime**: `llama.cpp` compiled for Android via Termux or standalone NDK binary.

---

## 2. Model Export to GGUF Q4_K_M

From the host machine, ensure the bilingual model is merged and quantized:

```powershell
# 1. Merge LoRA adapter into base model
python scripts/merge_lora.py `
  --adapter-dir models/gemma3-1b-bilingual-lora-smoke `
  --output-dir models/gemma3-1b-bilingual-merged

# 2. Convert to GGUF F16 and Quantize to Q4_K_M
python scripts/quantize_models.py `
  --model-dir models/gemma3-1b-bilingual-merged `
  --tiers f16 q8_0 q4_k_m
```

The resulting 4-bit mobile binary is located at:
`models/gguf/gemma3-1b-bilingual-merged-q4_k_m.gguf`

---

## 3. On-Device Execution via Termux (Android)

### Step 3.1: Install Termux & Build llama.cpp
On the Android device, open Termux and run:

```bash
pkg update && pkg upgrade -y
pkg install clang cmake make git -y

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Compile optimized for mobile ARM NEON
cmake -B build
cmake --build build --config Release -j$(nproc)
```

### Step 3.2: Transfer the 4-bit Model to Android
Transfer the `.gguf` file via USB ADB, Termux SSH, or HTTP transfer:

```bash
# Using ADB from PC:
adb push models/gguf/gemma3-1b-bilingual-merged-q4_k_m.gguf /sdcard/Download/

# In Termux:
termux-setup-storage
cp /sdcard/Download/gemma3-1b-bilingual-merged-q4_k_m.gguf ~/
```

### Step 3.3: Interactive CLI Inference on Mobile

Run interactive generation in Termux:

```bash
./build/bin/llama-cli \
  -m ~/gemma3-1b-bilingual-merged-q4_k_m.gguf \
  -n 128 \
  -t 4 \
  --temp 0.2 \
  -p "User: জাতীয় দুর্যোগ ব্যবস্থাপনা কাউন্সিলের সভাপতি কে?\nAssistant:"
```

---

## 4. Alternative: Lightweight Local Web Server Demo

For live live demonstrations to colleagues or evaluators without compiling C++ on the device, launch the bundled lightweight server on the host machine and access it directly from the phone's mobile browser over local Wi-Fi:

```powershell
python scripts/demo_mobile_inference.py --port 8080 --precision 4bit
```

Open `http://<YOUR_PC_LOCAL_IP>:8080` on the mobile phone browser to test Bangla and English prompt queries interactively with real-time latency and token generation metrics.
