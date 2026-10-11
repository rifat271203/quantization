# Cross-Lingual Impact of Quantization on Hallucination: A Comparative Study Between Bangla and English in Compressed 1B-Parameter Language Models

## Abstract / Introduction

The deployment of Large Language Models (LLMs) on resource-constrained mobile devices has become increasingly feasible through post-training quantization techniques (16-bit, 8-bit, and 4-bit precision), which drastically reduce model size and memory footprint. However, this compression is not free — reduced numerical precision is known to degrade model output quality, and one of the most critical failure modes is **hallucination**: the generation of factually incorrect or fabricated content. While the general relationship between quantization and hallucination has been studied for high-resource languages such as English, it remains largely unexplored for low-resource languages such as Bangla, which differ significantly in tokenizer efficiency, training data volume, and morphological complexity.

This project investigates whether hallucination behavior degrades **differently** in Bangla compared to English when a language model is compressed to the same bit-width. We fine-tune a small (~1B parameter) language model on parallel-domain data in both languages, quantize it to 16-bit, 8-bit, and 4-bit precision, and evaluate hallucination rates and error patterns at each compression level for both languages under identical conditions. The goal is to produce an empirically grounded, reproducible comparison that highlights whether low-resource languages are disproportionately vulnerable to quantization-induced quality loss — a finding directly relevant to the growing push toward on-device, offline AI tools for underserved language communities.

## Problem Statement

On-device deployment of LLMs is essential for accessibility, privacy, and offline usability, especially in regions with unreliable internet access such as Bangladesh. Quantization is the standard technique to make this feasible on mobile hardware. However:

- Most existing research on quantization-induced quality degradation and hallucination is conducted almost exclusively on **English** benchmarks.
- Bangla, as a low-resource and morphologically rich language, is underrepresented in LLM pretraining corpora and often suffers from poorer tokenizer efficiency, which may compound the negative effects of quantization.
- It is currently **unknown** whether the same bit-width reduction causes proportionally more, less, or differently-structured hallucinations in Bangla compared to English — meaning developers building Bangla-language mobile AI tools have no empirical basis to decide "how much compression is safe."
- Without this understanding, real-world deployments risk shipping models that appear acceptable in English-language testing but silently produce more unreliable output for Bangla-speaking end users.

## Objectives

1. Fine-tune a single ~1B parameter base language model on a matched-domain dataset in both Bangla and English.
2. Quantize the fine-tuned model to three precision levels: 16-bit, 8-bit, and 4-bit.
3. Construct a parallel, human-verified evaluation set of question-answer pairs in both languages to measure factual correctness.
4. Measure and compare hallucination rate at each quantization level, separately for Bangla and English.
5. Categorize hallucination types (e.g., factual substitution, fabricated entities, code-mixing breakdown, incoherent output) to identify whether the *nature* of degradation differs across languages, not just the rate.
6. Determine whether a "degradation threshold" (a bit-width at which quality drops sharply rather than gradually) exists, and whether this threshold differs between the two languages.
7. Package the final 4-bit model for on-device inference as a proof-of-concept mobile demo.

## Proposed Solution

We propose a controlled, side-by-side experimental pipeline:

1. **Data Collection** — Curate a matched-domain QA dataset available in parallel Bangla and English form (e.g., translated or naturally bilingual educational/STEM content), ensuring both language subsets are comparable in size, difficulty, and topic distribution.
2. **Fine-Tuning** — Fine-tune one shared base model (~1B parameters) on the combined bilingual dataset so that language is the primary controlled variable, not model capability.
3. **Quantization** — Produce three versions of the fine-tuned model at 16-bit, 8-bit, and 4-bit precision using a standard quantization toolchain (e.g., GGUF via llama.cpp).
4. **Evaluation** — Run a fixed, held-out evaluation set (parallel Bangla/English QA pairs) through all six model variants (2 languages × 3 bit-widths) and score outputs for factual correctness and hallucination type, using a documented human-rated rubric with inter-rater agreement checked across team members.
5. **Analysis** — Plot hallucination rate against bit-width separately for each language to directly compare degradation curves, and analyze the distribution of hallucination categories at each precision level.
6. **Deployment Demo** — Run the most aggressively compressed (4-bit) model on a mobile device using a lightweight on-device inference engine, as a practical demonstration of feasibility.

## Features

- Bilingual (Bangla–English) fine-tuned base model from a single shared training run.
- Three quantization tiers (16-bit / 8-bit / 4-bit) of the same model for direct comparison.
- Parallel, human-verified bilingual evaluation benchmark built specifically for this study.
- Structured hallucination taxonomy (not just a single aggregate score).
- Cross-lingual degradation comparison as the paper's core empirical contribution.
- Working mobile on-device inference demo using the final quantized model.

## Technology Stack

- **Base Model:** Small open-weight ~1B parameter LLM with reasonable multilingual/Bangla tokenizer support (e.g., Gemma 2 1B, Qwen2.5 1.5B/0.5B, or Llama 3.2 1B — to be finalized after a tokenizer efficiency check on Bangla text).
- **Fine-Tuning:** Hugging Face Transformers + PEFT/LoRA for efficient fine-tuning on limited compute; training tracked via Weights & Biases.
- **Quantization:** llama.cpp / GGUF conversion pipeline for 16-bit, 8-bit, and 4-bit variants; bitsandbytes as a secondary tool for early experimentation.
- **Evaluation Tooling:** Python-based scoring scripts, embedding-similarity metrics (e.g., cosine similarity via multilingual sentence embeddings) alongside manual human rubric scoring.
- **Mobile Deployment:** llama.cpp Android build or MLC-LLM for running the final 4-bit model on-device.
- **Data/Experiment Management:** Google Colab / local GPU for fine-tuning; Google Sheets or a lightweight spreadsheet for tracking annotation and scoring across the 3-person team.

## System Architecture

```
                        ┌───────────────────────────┐
                        │   Bilingual Dataset        │
                        │ (Bangla + English, matched │
                        │   domain QA pairs)          │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌───────────────────────────┐
                        │   Base Model (~1B params)  │
                        │      Fine-Tuning (LoRA)    │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌───────────────────────────┐
                        │   Fine-Tuned Model (FP16)  │
                        └────────────┬───────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌────────────────┐   ┌────────────────┐    ┌────────────────┐
     │  16-bit variant │   │  8-bit variant  │    │  4-bit variant  │
     └────────┬────────┘   └────────┬────────┘    └────────┬────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Parallel Bangla/English          │
                    │  Evaluation Benchmark             │
                    │  (Human-rated + Automatic scoring)│
                    └────────────────┬───────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Hallucination Rate & Taxonomy    │
                    │  Analysis (per language, per      │
                    │  bit-width)                       │
                    └────────────────┬───────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │  On-Device Mobile Demo            │
                    │  (4-bit model, llama.cpp/MLC-LLM) │
                    └─────────────────────────────────┘
```

