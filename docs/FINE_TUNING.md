# Fine-tuning guide (Gemma 3 1B smoke LoRA)

This project fine-tunes **`google/gemma-3-1b-it`** with **QLoRA** (4-bit) so it fits a small GPU such as an **RTX 3050 Ti (4 GiB)**.

You already prepared:

| File | Role |
|------|------|
| `data/train_bangla.jsonl` | Bangla instruction → response pairs |
| `data/train_english.jsonl` | English pairs with the **same ids** |
| `scripts/finetune_lora.py` | Smoke training script |

---

## What “tuning” does here

1. **Load** Gemma 3 1B Instruct (weights stay mostly frozen).
2. **Quantize** to 4-bit (QLoRA) to save VRAM.
3. Attach small **LoRA adapters** on attention/MLP layers (`q/k/v/o` and `gate/up/down` proj).
4. Format each JSONL row with the model’s **chat template**:
   - user = `instruction`
   - assistant = `response`
5. Run a short **supervised fine-tune** (`SFTTrainer` from TRL) for `--max-steps` steps (smoke default: 20).
6. **Save only the LoRA adapter** (+ tokenizer) under `models/…` — not a full copy of the 1B model.

Bangla and English are trained **separately** so you can later compare language-specific adapters (useful for your hallucination / quantization study).

```text
PDF / chunks → QA review → train_*.jsonl
                              ↓
                    finetune_lora.py (QLoRA)
                              ↓
              models/gemma3-1b-<lang>-lora-smoke/
                    (adapter_config.json, adapter weights)
```

---

## One-time setup

From the repo root, with the project venv active:

```powershell
cd D:\Tokenizer\quant-hallucination-study
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Needed packages for training: `peft`, `trl`, `datasets`, `bitsandbytes` (plus existing `torch` / `transformers` / `accelerate`).

Hugging Face access: you must already be able to download `google/gemma-3-1b-it` (gated model → accept license + `huggingface-cli login` if needed).

---

## Exact commands to start smoke tuning

**Bangla smoke (recommended first):**

```powershell
cd D:\Tokenizer\quant-hallucination-study
.\.venv\Scripts\Activate.ps1
python scripts/finetune_lora.py --lang bangla --max-steps 20
```

Uses **TRL 1.10 `SFTConfig`** (not older `TrainingArguments` + `dataset_text_field` on the trainer). If you see API errors about `warmup_ratio` / `dataset_text_field`, update to the current `scripts/finetune_lora.py`.

**English smoke:**

```powershell
python scripts/finetune_lora.py --lang english --max-steps 20
```

**Outputs:**

- Bangla → `models/gemma3-1b-bangla-lora-smoke/`
- English → `models/gemma3-1b-english-lora-smoke/`

During the run you should see:

- GPU name / VRAM
- “Loading model in 4-bit (QLoRA)”
- trainable parameter % (LoRA — a small fraction)
- step logs (`loss=…`) every step
- `Saved LoRA adapter + tokenizer to …`
- Peak VRAM

---

## Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--lang` | `bangla` | `bangla` or `english` |
| `--max-steps` | `20` | Smoke length; raise later for real training |
| `--max-seq-length` | `512` | Truncate long chat sequences |
| `--lr` | `2e-4` | Learning rate |
| `--lora-r` | `8` | LoRA rank (higher = more capacity / VRAM) |
| `--no-4bit` | off | Full LoRA in fp16 — **needs more VRAM** |
| `--output-dir` | auto | Custom save folder |

Example longer (still careful) run:

```powershell
python scripts/finetune_lora.py --lang bangla --max-steps 100 --lora-r 16
```

---

## After smoke training

1. Confirm adapter files exist under `models/gemma3-1b-*-lora-smoke/`.
2. Later: load base Gemma + attach adapter and generate on held-out questions.
3. Grow the dataset beyond 56 pairs before serious evaluation.
4. Study path: compare FP16 / 8-bit / 4-bit inference quality and hallucination.

---

## Hardware note (your machine)

| Setting | Why |
|---------|-----|
| 4-bit QLoRA | Fits ~4 GiB laptop GPUs |
| `batch_size=1` | Saves VRAM |
| `gradient_accumulation_steps=4` | Effective batch without more memory |
| `gradient_checkpointing=True` | Trades compute for VRAM |
| Short `--max-steps` | Smoke only — not a full train |

If you OOM: lower `--max-seq-length` (e.g. 256) or `--lora-r` (e.g. 4).

If `bitsandbytes` fails on Windows: install the matching CUDA wheel, or ask for a CPU/non-4bit fallback path (much slower / may not fit).

---

## What this smoke run is *not*

- Not a full-quality fine-tune (20 steps × 56 examples is only a pipeline check).
- Does not yet run quantization hallucination eval.
- Does not merge adapters into a standalone full model (adapters stay separate unless you merge later).
