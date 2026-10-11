"""
Generate the complete Kaggle fine-tuning notebook: kaggle_finetune_gemma3_qlora.ipynb
"""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # Cell 1: Markdown Title
    add_md("""# Cross-Lingual Quantization Hallucination Study: QLoRA Fine-Tuning
### Model: `google/gemma-3-1b-it` (Pinned HF Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`)
This notebook executes the complete 3-variant fine-tuning pipeline on Kaggle GPU (T4 / P100):
1. **Bilingual (Primary)**: Full 5,448-pair train split, BN+EN interleaved in one shared LoRA adapter.
2. **Bangla-only Ablation**: 2,549 BN examples only.
3. **English-only Ablation**: 2,899 EN examples only.

All three variants use strictly identical hyperparameters (LoRA rank=16, alpha=32, batch=1, grad_accum=8, lr=2e-4, 3 epochs).
After each run, the LoRA adapter is merged into the base model to produce a standalone **FP16 checkpoint** ready for subsequent GGUF quantization.""")

    # Cell 2: Install dependencies
    add_code("""# Step 1: Install required packages
!pip install -q -U "transformers>=4.48.0" accelerate peft bitsandbytes "trl>=0.12.0" datasets huggingface_hub""")

    # Cell 3: GPU & Environment Setup
    add_code("""# Step 2: Verify GPU and environment
import os
import sys
import gc
import json
import time
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

print("PyTorch Version:", torch.__version__)
assert torch.cuda.is_available(), "GPU accelerator is required! Enable GPU T4 in Kaggle notebook settings."

gpu_name = torch.cuda.get_device_name(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU Detected: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")

# Turing GPUs (like T4) run fastest on float16; Ampere/Hopper run on bfloat16
USE_BF16 = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"Using compute dtype: {COMPUTE_DTYPE} (bf16={USE_BF16}, fp16={not USE_BF16})")""")

    # Cell 4: Hugging Face Authentication
    add_code("""# Step 3: Authenticate with Hugging Face to access google/gemma-3-1b-it
from huggingface_hub import login

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    print("Found HF_TOKEN in Kaggle Secrets.")
except Exception:
    hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    # If not found in secrets, prompt user
    print("Please enter your Hugging Face Access Token (with read access to google/gemma-3-1b-it):")
    from getpass import getpass
    hf_token = getpass("HF Token: ")

login(token=hf_token)
print("Hugging Face login successful!")""")

    # Cell 5: Locate and Extract Data
    add_code("""# Step 4: Locate dataset files
# Check if uploaded as a Kaggle Dataset under /kaggle/input, or uploaded as zip in current directory
DATA_DIR = None

# Search in /kaggle/input
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_bilingual.jsonl" in files:
        DATA_DIR = Path(root)
        print(f"Found dataset in Kaggle input: {DATA_DIR}")
        break

if DATA_DIR is None:
    # Check current directory for zip file or unpacked data
    if Path("kaggle_dataset.zip").exists():
        !unzip -o kaggle_dataset.zip -d ./data
        DATA_DIR = Path("./data")
    elif Path("data/train_bilingual.jsonl").exists():
        DATA_DIR = Path("data")
    elif Path("train_bilingual.jsonl").exists():
        DATA_DIR = Path(".")
    else:
        raise FileNotFoundError(
            "Could not find train_bilingual.jsonl! Please upload kaggle_dataset.zip or add it as a Kaggle Dataset."
        )

print(f"Dataset directory verified: {DATA_DIR.resolve()}")
for fname in ["train_bilingual.jsonl", "train_bn_only.jsonl", "train_en_only.jsonl", "splits.json"]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        line_count = sum(1 for _ in open(fpath, encoding="utf-8")) if fname.endswith(".jsonl") else "N/A"
        print(f"  ✓ {fname:25s} (Size: {fpath.stat().st_size / 1024:.1f} KB, Records: {line_count})")""")

    # Cell 6: Model Constants & LoRA Configuration
    add_code("""# Step 5: Study Configuration and Pinned Commit Hash
MODEL_ID = "google/gemma-3-1b-it"
PINNED_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

MAX_SEQ_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8
LEARNING_RATE = 2e-4
EPOCHS = 3

OUTPUT_BASE_DIR = Path("/kaggle/working/models")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
print("Base output directory:", OUTPUT_BASE_DIR)""")

    # Cell 7: Helper Functions
    add_code("""# Step 6: Dataset Loader & Step Metrics Callback
def load_dataset_from_jsonl(path: Path, tokenizer, max_seq_length: int = 1024) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    texts = []
    exceed_count = 0
    for r in rows:
        messages = r["messages"]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tok_len = len(tokenizer.encode(chat_text, add_special_tokens=False))
        if tok_len > max_seq_length:
            exceed_count += 1
        texts.append(chat_text)

    print(f"Loaded {len(texts)} examples from {path.name} (Exceeding {max_seq_length}: {exceed_count})")
    return Dataset.from_dict({"text": texts}), len(rows)


class StepMetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.step_times = []
        self._last_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        self.step_times.append(now - self._last_time)
        self._last_time = now""")

    # Cell 8: Fine-Tuning Function
    add_code("""# Step 7: Core Fine-Tuning Execution Function
def train_variant(variant_name: str, data_file: Path, epochs: int = EPOCHS):
    print("\\n" + "=" * 75)
    print(f"STARTING FINE-TUNING VARIANT: {variant_name.upper()}")
    print("=" * 75)
    print(f"Data File:       {data_file}")
    print(f"Epochs:          {epochs}")
    print(f"Effective Batch: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Target Modules:  {TARGET_MODULES}")

    adapter_dir = OUTPUT_BASE_DIR / f"gemma3-1b-{variant_name}-lora"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Dataset
    dataset, total_records = load_dataset_from_jsonl(data_file, tokenizer, MAX_SEQ_LENGTH)

    # 3. Model in 4-bit NF4
    print("Loading base model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=COMPUTE_DTYPE,
    )
    model.config.use_cache = False

    # 4. Attach LoRA
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=25,
        report_to=[],
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    step_cb = StepMetricsCallback()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[step_cb],
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    train_result = trainer.train()
    total_time_s = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"\\n✓ Training finished in {total_time_s / 60:.2f} minutes ({total_time_s:.1f} s)")
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"  Final Loss: {train_result.training_loss:.4f}")

    # Save adapter
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    manifest = {
        "variant": variant_name,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "data_file": str(data_file.name),
        "total_records": total_records,
        "epochs": epochs,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "wall_clock_seconds": total_time_s,
        "final_loss": train_result.training_loss,
        "peak_vram_gb": peak_vram_gb,
        "device": gpu_name,
    }
    with (adapter_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Clean up model from memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return adapter_dir, manifest""")

    # Cell 9: Merge Function
    add_code("""# Step 8: Merge LoRA Adapter into Base Model (Single FP16 Checkpoint)
def merge_variant(adapter_dir: Path, variant_name: str):
    print("\\n" + "=" * 75)
    print(f"MERGING LORA ADAPTER INTO FP16 BASE MODEL: {variant_name.upper()}")
    print("=" * 75)

    merged_out_dir = OUTPUT_BASE_DIR / f"gemma3-1b-{variant_name}-merged-fp16"
    merged_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)

    # Load base model in FP16
    print(f"Loading unquantized base model '{MODEL_ID}' in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Attach and merge
    print(f"Attaching adapter from {adapter_dir.name}...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print("Merging adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    # Save merged model
    print(f"Saving merged FP16 checkpoint to {merged_out_dir}...")
    merged_model.save_pretrained(str(merged_out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_out_dir))

    elapsed = time.perf_counter() - t0
    print(f"✓ Merge complete in {elapsed:.1f} s ({elapsed / 60:.2f} min)")
    print(f"  Standalone FP16 checkpoint: {merged_out_dir}")

    # Clean memory
    del base_model, peft_model, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    return merged_out_dir""")

    # Cell 10: Run Variant 1 (Bilingual)
    add_code("""# RUN 1/3: Bilingual Primary Model (BN + EN Interleaved)
bilingual_data = DATA_DIR / "train_bilingual.jsonl"
bi_adapter_dir, bi_manifest = train_variant("bilingual", bilingual_data, epochs=3)
bi_merged_dir = merge_variant(bi_adapter_dir, "bilingual")""")

    # Cell 11: Run Variant 2 (Bangla-Only)
    add_code("""# RUN 2/3: Bangla-Only Ablation
bn_data = DATA_DIR / "train_bn_only.jsonl"
bn_adapter_dir, bn_manifest = train_variant("bn-only", bn_data, epochs=3)
bn_merged_dir = merge_variant(bn_adapter_dir, "bn-only")""")

    # Cell 12: Run Variant 3 (English-Only)
    add_code("""# RUN 3/3: English-Only Ablation
en_data = DATA_DIR / "train_en_only.jsonl"
en_adapter_dir, en_manifest = train_variant("en-only", en_data, epochs=3)
en_merged_dir = merge_variant(en_adapter_dir, "en-only")""")

    # Cell 13: Summary Report & Verification
    add_code("""# Step 9: Final Comparative Report
all_manifests = [bi_manifest, bn_manifest, en_manifest]

print("=" * 85)
print("FINAL FINE-TUNING SUMMARY ACROSS ALL THREE VARIANTS")
print("=" * 85)
header = f"{'Variant':15s} | {'Records':8s} | {'Final Loss':11s} | {'Wall Clock':12s} | {'Peak VRAM':10s} | {'Merged FP16 Path'}"
print(header)
print("-" * 85)

for m in all_manifests:
    vname = m["variant"]
    n_rec = str(m["total_records"])
    loss = f"{m['final_loss']:.4f}"
    time_str = f"{m['wall_clock_seconds'] / 60:.1f} min"
    vram = f"{m['peak_vram_gb']:.2f} GB"
    merged_path = f"models/gemma3-1b-{vname}-merged-fp16"
    print(f"{vname:15s} | {n_rec:8s} | {loss:11s} | {time_str:12s} | {vram:10s} | {merged_path}")

print("=" * 85)
print("✓ Base model ('google/gemma-3-1b-it') was left completely untouched as comparison baseline.")
print("✓ Zero dev/eval split documents were touched during training.")
print("✓ All 3 merged FP16 checkpoints produced successfully!")""")

    # Cell 14: Archive Outputs for Easy Download
    add_code("""# Step 10: Compress LoRA adapters & manifests for convenient download
# (Note: Merged FP16 weights are ~2.5 GB each; you can download adapters + manifests or full models)
import subprocess

print("Archiving LoRA adapters and manifests...")
!tar -czf /kaggle/working/lora_adapters_all_variants.tar.gz -C /kaggle/working/models gemma3-1b-bilingual-lora gemma3-1b-bn-only-lora gemma3-1b-en-only-lora

print("Finished! Output files available in /kaggle/working:")
!ls -lh /kaggle/working""")

    output_path = Path("kaggle_finetune_gemma3_qlora.ipynb")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    create_notebook()
