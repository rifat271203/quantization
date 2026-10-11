import json
from pathlib import Path

VARIANTS = [
    {
        "filename": "gemma3-qlora-bilingual.ipynb",
        "variant_key": "bilingual",
        "title": "Bilingual (Primary)",
        "name": "Bilingual (Primary)",
        "data_file": "train_bilingual.jsonl",
        "record_count": 5448,
        "desc": "Full BN+EN interleaved dataset",
    },
    {
        "filename": "gemma3-qlora-bn-only.ipynb",
        "variant_key": "bn-only",
        "title": "Bangla-Only Ablation",
        "name": "Bangla-Only Ablation",
        "data_file": "train_bn_only.jsonl",
        "record_count": 2549,
        "desc": "Bangla-only subset",
    },
    {
        "filename": "gemma3-qlora-en-only.ipynb",
        "variant_key": "en-only",
        "title": "English-Only Ablation",
        "name": "English-Only Ablation",
        "data_file": "train_en_only.jsonl",
        "record_count": 2899,
        "desc": "English-only subset",
    },
]

def make_notebook(v):
    md_header = f"""# Cross-Lingual Quantization Hallucination Study: QLoRA Fine-Tuning
## Variant: {v['title']} (Standalone Single-GPU Execution)
### Base Model: `google/gemma-3-1b-it` (Pinned HF Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`)

This standalone notebook fine-tunes the **{v['name']}** variant independently on Kaggle GPU:
- **Dataset**: `{v['data_file']}` ({v['record_count']:,} pairs, {v['desc']})
- **Hyperparameters**: LoRA rank $r=16$, $\\alpha=32$, dropout $0.05$, per-device batch size 1, gradient accumulation 8 (effective batch size 8), lr $2\\times 10^{{-4}}$, 3 epochs.
- **Hardware Strategy**: Explicit single-GPU binding (`device_map={{\"\": 0}}`) to prevent cross-GPU communication overhead on T4 x2 accelerators.
- **Checkpoint Resilience**: Step-based checkpointing (`save_steps=200`, `save_total_limit=3`) with automatic resumption if a prior checkpoint exists.
- **Time Budget Monitor**: Logs wall-clock time and remaining 12-hour Kaggle session budget every 500 steps.
- **Post-Training**: Automatically merges the LoRA adapter into base FP16 weights and packages outputs as downloadable zip archives.
"""

    code_step1 = """# Step 1: Install required packages
!pip install -q -U "transformers>=4.48.0" accelerate peft bitsandbytes "trl>=0.12.0" datasets huggingface_hub"""

    code_step2 = """# Step 2: Verify single GPU environment
import os
import sys
import gc
import json
import time
import shutil
from datetime import datetime
from pathlib import Path

import torch

print("PyTorch Version:", torch.__version__)
assert torch.cuda.is_available(), "GPU accelerator is required! Enable GPU (e.g., T4 x2 or P100) in Kaggle notebook settings."

gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

print(f"Detected {gpu_count} GPU(s).")
print(f"Target Training Device: GPU 0 -> {gpu_name} ({total_vram_gb:.2f} GB VRAM)")
if gpu_count > 1:
    print("NOTE: Multiple GPUs detected. Explicit single-GPU allocation (`device_map={'': 0}`) will be used to avoid multi-GPU communication overhead.")

# Determine compute dtype: Turing (T4) uses fp16; Ampere/Hopper uses bf16
USE_BF16 = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"Using compute dtype: {COMPUTE_DTYPE} (bf16={USE_BF16}, fp16={not USE_BF16})")"""

    code_step3 = """# Step 3: Authenticate with Hugging Face to access google/gemma-3-1b-it
from huggingface_hub import login

hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    print("✓ Successfully retrieved HF_TOKEN from Kaggle Secrets.")
except Exception as e:
    print(f"Kaggle Secrets lookup note: {e}")
    hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    print("Please enter your Hugging Face Access Token (read access to google/gemma-3-1b-it):")
    from getpass import getpass
    hf_token = getpass("HF Token: ")

login(token=hf_token)
print("✓ Hugging Face authentication complete!")"""

    code_step4 = f"""# Step 4: Hardcoded Variant Configuration (Single Run per Notebook)
VARIANT_NAME = "{v['variant_key']}"
DATA_FILE_NAME = "{v['data_file']}"
EXPECTED_RECORDS = {v['record_count']}

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

SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 10
TIME_WARNING_INTERVAL = 500

OUTPUT_BASE_DIR = Path("/kaggle/working/models")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = OUTPUT_BASE_DIR / f"gemma3-1b-{{VARIANT_NAME}}-lora"
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR = OUTPUT_BASE_DIR / f"gemma3-1b-{{VARIANT_NAME}}-merged-fp16"

print(f"=== Active Variant: {{VARIANT_NAME.upper()}} ===")
print(f"Data Target:      {{DATA_FILE_NAME}} ({{EXPECTED_RECORDS}} records)")
print(f"Adapter Output:   {{ADAPTER_DIR}}")
print(f"Merged FP16 Out:  {{MERGED_DIR}}")"""

    code_step5 = """# Step 5: Locate and verify the variant dataset file
DATA_DIR = None

# Search mounted Kaggle inputs
for root, dirs, files in os.walk("/kaggle/input"):
    if DATA_FILE_NAME in files:
        DATA_DIR = Path(root)
        print(f"✓ Found {DATA_FILE_NAME} in Kaggle input: {DATA_DIR}")
        break

# Fallback: check working directory for zip archives or data folders
if DATA_DIR is None:
    for zip_candidate in ["train_data.zip", "kaggle_dataset.zip"]:
        for search_path in [Path(zip_candidate), Path(f"/kaggle/working/{zip_candidate}")]:
            if search_path.exists():
                print(f"Unpacking {search_path}...")
                !unzip -o -q {search_path} -d /kaggle/working/data
                DATA_DIR = Path("/kaggle/working/data")
                break
        if DATA_DIR is not None:
            break

if DATA_DIR is None:
    for dir_candidate in [Path("data"), Path("/kaggle/working/data"), Path(".")]:
        if (dir_candidate / DATA_FILE_NAME).exists():
            DATA_DIR = dir_candidate
            break

if DATA_DIR is None or not (DATA_DIR / DATA_FILE_NAME).exists():
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE_NAME}'! Please attach your dataset to this Kaggle notebook (e.g. train_data.zip)."
    )

TARGET_DATA_PATH = DATA_DIR / DATA_FILE_NAME
with TARGET_DATA_PATH.open(encoding="utf-8") as f:
    actual_records = sum(1 for line in f if line.strip())

file_size_kb = TARGET_DATA_PATH.stat().st_size / 1024
print(f"✓ Verified dataset: {TARGET_DATA_PATH.resolve()}")
print(f"  File size: {file_size_kb:.1f} KB")
print(f"  Record count: {actual_records} (Expected: {EXPECTED_RECORDS})")
if actual_records != EXPECTED_RECORDS:
    print(f"  [Notice] Record count ({actual_records}) differs from expected ({EXPECTED_RECORDS}). Proceeding with verified file.")"""

    code_step6 = """# Step 6: Dataset preparation & Time Budget Monitoring Callback
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback

def load_variant_dataset(path: Path, tokenizer, max_seq_length: int = 1024) -> tuple[Dataset, int]:
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

    print(f"Processed {len(texts)} examples from {path.name} (Exceeding {max_seq_length} tokens: {exceed_count})")
    return Dataset.from_dict({"text": texts}), len(rows)


class TimeBudgetCallback(TrainerCallback):
    \"\"\"Logs elapsed wall-clock time and remaining 12-hour session budget every N steps.\"\"\"
    def __init__(self, budget_hours: float = 12.0, warning_interval_steps: int = 500):
        super().__init__()
        self.budget_seconds = budget_hours * 3600.0
        self.warning_interval = warning_interval_steps
        self.train_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\\n[{now_str}] Training session monitor started.")
        print(f"  Kaggle 12-Hour Budget: {self.budget_seconds / 3600:.1f} hours ({self.budget_seconds:.0f} seconds)")
        print(f"  Progress & budget warnings interval: every {self.warning_interval} steps\\n")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.warning_interval == 0:
            now = time.time()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed_sec = now - self.train_start_time
            budget_left_sec = max(0.0, self.budget_seconds - elapsed_sec)

            total_steps = state.max_steps if state.max_steps and state.max_steps > 0 else 1
            steps_completed = state.global_step
            steps_remaining = max(0, total_steps - steps_completed)
            avg_sec_per_step = elapsed_sec / max(1, steps_completed)
            est_remaining_sec = steps_remaining * avg_sec_per_step

            print("\\n" + "=" * 75)
            print(f"⏱ [{now_str}] TIME BUDGET MONITOR — STEP {steps_completed} / {total_steps}")
            print("=" * 75)
            print(f"  • Elapsed Wall-Clock:      {elapsed_sec / 3600:.2f} h ({elapsed_sec / 60:.1f} min)")
            print(f"  • Remaining 12h Budget:    {budget_left_sec / 3600:.2f} h ({budget_left_sec / 60:.1f} min)")
            print(f"  • Estimated Time to End:   {est_remaining_sec / 3600:.2f} h ({est_remaining_sec / 60:.1f} min)")
            print(f"  • Average Pace:            {avg_sec_per_step:.2f} sec/step")
            if est_remaining_sec > budget_left_sec:
                print("  ⚠️  WARNING: Estimated training time exceeds remaining Kaggle session budget!")
                print("      Checkpoints are saved every 200 steps. Re-run this notebook to resume from the latest checkpoint.")
            else:
                print("  ✓ On schedule to complete comfortably within the 12-hour budget.")
            print("=" * 75 + "\\n")"""

    code_step7 = """# Step 7: QLoRA Fine-Tuning with Checkpoint Resume Support
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

print("=" * 75)
print(f"STARTING QLORA FINE-TUNING: {VARIANT_NAME.upper()}")
print("=" * 75)

# 1. Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=PINNED_REVISION,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Prepare Dataset
dataset, total_records = load_variant_dataset(TARGET_DATA_PATH, tokenizer, MAX_SEQ_LENGTH)

# 3. Load Base Model in 4-bit NF4 — Explicit Single GPU: device_map={"": 0}
print("Loading base model in 4-bit NF4 on GPU 0 explicitly...")
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
    device_map={"": 0},  # Explicit single GPU (avoids T4 x2 multi-GPU splitting overhead)
    trust_remote_code=True,
    dtype=COMPUTE_DTYPE,
)
model.config.use_cache = False

# 4. Attach LoRA Adapter
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

# 5. Training Arguments with Step Checkpointing (save_steps=200, save_total_limit=3)
training_args = SFTConfig(
    output_dir=str(ADAPTER_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
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

time_monitor_cb = TimeBudgetCallback(budget_hours=12.0, warning_interval_steps=TIME_WARNING_INTERVAL)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[time_monitor_cb],
)

# 6. Checkpoint Resume Detection
last_checkpoint = None
if ADAPTER_DIR.exists():
    last_checkpoint = get_last_checkpoint(str(ADAPTER_DIR))

if last_checkpoint is not None:
    print(f"\\n🔄 Found existing checkpoint: {last_checkpoint}")
    print(">>> Resuming training from checkpoint! <<<")
else:
    print("\\n🚀 No prior checkpoint detected in output directory.")
    print(">>> Starting fresh training run! <<<")

torch.cuda.reset_peak_memory_stats(0)
train_start = time.perf_counter()

if last_checkpoint is not None:
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    train_result = trainer.train()

total_train_sec = time.perf_counter() - train_start
peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

print("\\n" + "=" * 75)
print(f"✓ Training finished in {total_train_sec / 3600:.2f} h ({total_train_sec / 60:.1f} min)")
print(f"  Peak VRAM:   {peak_vram_gb:.2f} GB")
print(f"  Final Loss:  {train_result.training_loss:.4f}")
print("=" * 75)

# Save final adapter weights and tokenizer
print(f"Saving final adapter and tokenizer to {ADAPTER_DIR}...")
trainer.model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

# Save execution manifest
manifest = {
    "variant": VARIANT_NAME,
    "model_id": MODEL_ID,
    "revision": PINNED_REVISION,
    "data_file": DATA_FILE_NAME,
    "total_records": total_records,
    "epochs": EPOCHS,
    "batch_size": PER_DEVICE_BATCH_SIZE,
    "grad_accum": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "max_seq_length": MAX_SEQ_LENGTH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "target_modules": TARGET_MODULES,
    "wall_clock_seconds": total_train_sec,
    "final_loss": train_result.training_loss,
    "peak_vram_gb": peak_vram_gb,
    "device": gpu_name,
    "resumed_from": str(last_checkpoint) if last_checkpoint else None,
}
manifest_path = ADAPTER_DIR / "run_manifest.json"
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"✓ Saved run manifest: {manifest_path}")

# Clean up training model from GPU VRAM
del model, trainer
gc.collect()
torch.cuda.empty_cache()"""

    code_step8 = """# Step 8: Merge LoRA Adapter into Base Model (Standalone FP16 Checkpoint)
from peft import PeftModel

print("=" * 75)
print(f"MERGING LORA ADAPTER INTO FP16 BASE MODEL: {VARIANT_NAME.upper()}")
print("=" * 75)

merge_start = time.perf_counter()

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)

# 2. Load unquantized FP16 Base Model — Explicit Single GPU: device_map={"": 0}
print(f"Loading unquantized base model '{MODEL_ID}' in FP16 on GPU 0...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=PINNED_REVISION,
    dtype=torch.float16,
    device_map={"": 0},  # Explicit single GPU
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 3. Attach and Merge
print(f"Attaching adapter from {ADAPTER_DIR.name}...")
peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))

print("Merging adapter weights into base model...")
merged_model = peft_model.merge_and_unload()

# 4. Save Standalone FP16 Checkpoint
print(f"Saving merged FP16 checkpoint to {MERGED_DIR}...")
merged_model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(MERGED_DIR))

merge_sec = time.perf_counter() - merge_start
print(f"✓ Merge complete in {merge_sec:.1f} s ({merge_sec / 60:.2f} min)")
print(f"  Standalone FP16 checkpoint: {MERGED_DIR}")

# Clean memory
del base_model, peft_model, merged_model
gc.collect()
torch.cuda.empty_cache()"""

    code_step9 = """# Step 9: Package Outputs into Downloadable Zip Archives
print("=" * 75)
print(f"PACKAGING OUTPUTS FOR {VARIANT_NAME.upper()}")
print("=" * 75)

# 1. Package LoRA adapter + manifest (~30-50 MB)
adapter_zip_base = f"/kaggle/working/gemma3-1b-{VARIANT_NAME}-lora"
print(f"Creating adapter archive: {adapter_zip_base}.zip ...")
shutil.make_archive(
    base_name=adapter_zip_base,
    format="zip",
    root_dir=OUTPUT_BASE_DIR,
    base_dir=f"gemma3-1b-{VARIANT_NAME}-lora",
)
adapter_zip_file = Path(f"{adapter_zip_base}.zip")
if adapter_zip_file.exists():
    print(f"✓ LoRA Archive Created: {adapter_zip_file.name} ({adapter_zip_file.stat().st_size / (1024**2):.2f} MB)")

# 2. Package Merged FP16 Model (~2.5 GB)
merged_zip_base = f"/kaggle/working/gemma3-1b-{VARIANT_NAME}-merged-fp16"
print(f"Creating merged FP16 archive: {merged_zip_base}.zip ...")
shutil.make_archive(
    base_name=merged_zip_base,
    format="zip",
    root_dir=OUTPUT_BASE_DIR,
    base_dir=f"gemma3-1b-{VARIANT_NAME}-merged-fp16",
)
merged_zip_file = Path(f"{merged_zip_base}.zip")
if merged_zip_file.exists():
    print(f"✓ Merged FP16 Archive Created: {merged_zip_file.name} ({merged_zip_file.stat().st_size / (1024**2):.2f} MB)")

print("\\nAll files in /kaggle/working:")
for p in Path("/kaggle/working").glob("*.zip"):
    print(f"  • {p.name:45s} ({p.stat().st_size / (1024**2):.2f} MB)")
print("\\n✓ Run complete! You can download the zip files directly from Kaggle output.")"""

    def make_cell(cell_type, src):
        # Convert multiline string to list of lines with \n
        lines = [line + "\n" for line in src.split("\n")]
        # Remove trailing newline from last line if it's empty
        if lines and lines[-1] == "\n":
            lines.pop()
        elif lines:
            lines[-1] = lines[-1].rstrip("\n")

        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell

    cells = [
        make_cell("markdown", md_header),
        make_cell("code", code_step1),
        make_cell("code", code_step2),
        make_cell("code", code_step3),
        make_cell("code", code_step4),
        make_cell("code", code_step5),
        make_cell("code", code_step6),
        make_cell("code", code_step7),
        make_cell("code", code_step8),
        make_cell("code", code_step9),
    ]

    notebook_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
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

    out_path = Path(v["filename"])
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)
    print(f"Generated {out_path} ({out_path.stat().st_size} bytes)")

for v in VARIANTS:
    make_notebook(v)
