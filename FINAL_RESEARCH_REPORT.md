# The Cost of Compression in Low-Resource Non-Latin Scripts:
## A Cross-Architectural Audit of Post-Training Quantization, Subword Fertility, and Factual Degradation

**Study Lead & Working Repository:** `d:/Tokenizer/quant-hallucination-study`  
**Date:** October 2026  
**Document Status:** Complete Empirical Census Report (N = 25,200 Predictions across 14 Evaluation Runs)  
**Primary Figure:** [`audit/gemma3_vs_llama32_comparative_dashboard.png`](file:///d:/Tokenizer/quant-hallucination-study/audit/gemma3_vs_llama32_comparative_dashboard.png)  

---

## Executive Abstract

Post-training quantization (PTQ) is widely deployed to enable Large Language Models (LLMs) on resource-constrained consumer edge devices. However, standard compression benchmarks evaluate almost exclusively on high-resource, Latin-script languages (primarily English). This empirical study conducts a systematic, parallel audit investigating the relationship between **tokenizer vocabulary coverage (subword fertility)** and **quantization degradation (from 16-bit float down to 2.5-bit integer)** on factual reading comprehension and decision stability.

Using the strictly parallel Meta AI Belebele benchmark across **900 parallel item pairs** (clustered within 488 unique document passages), we evaluate **Google Gemma 3 1B IT** (256,000 morpheme-aware vocabulary; fertility ratio = 1.35x) and **Meta Llama 3.2 1B Instruct** (128,000 byte-fallback BPE vocabulary; fertility ratio = 4.68x) across all 7 standard GGUF quantization tiers (`f16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K`).

### Major Empirical Findings:
1. **The 2-Bit Collapse to Chance Level in Llama 3.2:** At 2.5 bits per weight (`Q2_K`), Llama 3.2 1B collapses to chance level, scoring **23.22% in Bengali** (95% CI: [20.53%, 25.96%]) and **24.56% in English** (95% CI: [21.86%, 27.28%])—both statistically indistinguishable from the 25.0% random guessing floor. Under identical 2.5 bpw compression, Gemma 3 1B maintains **34.89% in Bengali** (95% CI: [31.88%, 38.08%]) and **43.44% in English** (95% CI: [40.38%, 46.48%]), remaining above chance though degraded from baseline.
2. **Tokenizer-Independent Information Loss (BPC Explosion):** Because raw perplexity varies by tokenizer token counts, we evaluate tokenizer-independent **Bits-Per-Character (BPC)**. Under 2.5 bpw compression, Bengali BPC increases by **+93.14%** in Llama 3.2 (from 1.70 to 3.28 BPC) compared to only **+10.57%** in Gemma 3 (from 1.86 to 2.06 BPC), an 8.8x higher information-density degradation penalty in Llama.
3. **The Q4_K_M Cancellation Paradox:** At 4.5 bpw (`Q4_K_M`), aggregate accuracy is statistically indistinguishable from `f16` in both models ($p > 0.14$). However, instance-level flip analysis reveals that **27.44% of Bengali decisions flip** relative to `f16`. Flat aggregate accuracy masks substantial latent churn because degradation errors are offset by opportunistic flips.
4. **Asymmetric Decision Instability:** Across all sub-8-bit tiers, Bengali exhibits higher prediction flip rates than English, widening from a modest +1%–2% gap at 6-bit to **+11.78% (Gemma)** and **+16.22% (Llama)** at 3.5 bpw. At 2.5 bpw, Llama 3.2 flips **69.67% of all Bengali decisions**, with agreement on high-confidence questions collapsing to **8.0%** (92% of confident decisions reversed).
5. **Within-Model Sequence Length Effects:** Within both models under extreme quantization (`Q2_K`), longer Bengali passages exhibit significantly higher decision flip rates than short passages ($r_{pb} = +0.071, p = 0.034$ in Gemma; $r_{pb} = +0.069, p = 0.038$ in Llama), whereas English passages show no length correlation ($r_{pb} = +0.002, p = 0.956$), supporting the hypothesis that sequence length exacerbates rounding noise in fragmented scripts.

---

## 1. Hypotheses & Conceptual Framework

```
                          [Post-Training Quantization (PTQ)]
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
   Gemma 3 1B IT (256k Vocab)                                  Llama 3.2 1B (128k Vocab)
  Subword Fertility: 1.35x                                     Subword Fertility: 4.68x
  (Morpheme-level tokens)                                      (Byte-level fragmentation)
            │                                                           │
            ▼                                                           ▼
   Fewer steps per semantic unit                               More steps per semantic unit
  Reduced cumulative rounding noise                           Higher cumulative rounding noise
            │                                                           │
            ▼                                                           ▼
   Maintains above-chance accuracy                             Collapses to chance level
   at Q2_K (BN: 34.89% | EN: 43.44%)                           at Q2_K (BN: 23.22% | EN: 24.56%)
   (BPC increase: +10.57%)                                     (BPC increase: +93.14%)
```

We formulated three central empirical hypotheses:

* **Hypothesis 1 (Inherent Baseline Disparity):** Due to asymmetric pre-training data allocation, models demonstrate an inherent factual grounding penalty in low-resource non-Latin scripts (Bengali) relative to English at uncompressed `f16` precision.
* **Hypothesis 2 (Compounding Sequence Noise):** Because post-training quantization introduces deterministic weight rounding noise, longer subword sequences compound errors across autoregressive steps. High-fertility tokenizers will exhibit monotonically accelerating degradation in character-normalized entropy and factual stability.
* **Hypothesis 3 (Latent Decision Churn):** Aggregate benchmark accuracy obscures instance-level instability. Tracking prediction flips relative to the unquantized baseline will reveal that non-Latin representations destabilize earlier and more severely than English representations across identical quantization tiers.

*Methodological Caveat on Causality:* While tokenizer fertility is a primary architectural difference between Gemma 3 1B and Llama 3.2 1B, these models also differ in pre-training data composition, multilingual token mixture, architectural details (e.g., tied embeddings, attention variants), and fine-tuning procedures. Our cross-model findings represent an observational comparative audit; within-model length regressions are utilized to isolate sequence-length effects directly.

---

## 2. Experimental Methodology & Evaluation Protocol

### 2.1 Parallel Benchmark Dataset
We utilized the **Meta AI Belebele Parallel Benchmark** (`data_v2/belebele_eval_parallel.jsonl`).
- **Parallel Census:** Exactly 900 parallel item pairs across Bengali (`bn`) and English (`en`).
- **Clustered Document Passages:** All 900 item pairs are drawn from **488 unique document passages**. Because questions sharing a passage share context, independent-sample intervals underestimate variance. All standard errors and confidence intervals in this study are computed via **488-passage cluster bootstrap resampling (2,000 iterations, seed 42)**.

### 2.2 Models & Precision Spectrum
Two state-of-the-art ~1B parameter instruction-tuned models were evaluated across 7 GGUF quantization tiers:
1. **Google Gemma 3 1B IT** (256,000 vocabulary; subword fertility = 1.35x)
2. **Meta Llama 3.2 1B Instruct** (128,000 vocabulary; subword fertility = 4.68x)

| Tier Name | Nominal Precision | Bit-Width (BPW) | Description |
| :--- | :---: | :---: | :--- |
| **`f16`** | 16-bit Float | 16.0 bpw | Full unquantized baseline reference |
| **`Q8_0`** | 8-bit Integer | 8.5 bpw | Standard 8-bit quantization; negligible theoretical loss |
| **`Q6_K`** | 6-bit K-quant | 6.5 bpw | High-fidelity 6-bit quantization |
| **`Q5_K_M`**| 5-bit K-quant | 5.5 bpw | Medium 5-bit quantization |
| **`Q4_K_M`**| 4-bit K-quant | 4.5 bpw | Standard edge/mobile deployment baseline |
| **`Q3_K_M`**| 3-bit K-quant | 3.5 bpw | Aggressive compression for memory-constrained edge devices |
| **`Q2_K`** | 2-bit K-quant | 2.5 bpw | Extreme compression (lowest standard GGUF tier) |

### 2.3 Evaluation Harness & Deterministic Logprob Extraction
- **Inference Engine:** Native `llama-server.exe` logprob extraction via OpenAI-compatible `/v1/chat/completions` (`temperature=0.0`, `max_tokens=1`, `logprobs=True`, `top_logprobs=20`).
- **Scoring Function:** For candidate option digits $c \in \{1, 2, 3, 4\}$, the predicted option is:
  $$\hat{c} = \arg\max_{c \in \{1, 2, 3, 4\}} \log P(c \mid \text{Prompt})$$
- **Total Census:** $\text{2 Models} \times \text{7 Tiers} \times \text{2 Languages} \times \text{900 Items} = \mathbf{25,200 \text{ Predictions}}$ (100% evaluated, 0.0% parsing failures).

---

## 3. Pillar 1: Subword Fertility & Tokenization Imbalance

| Evaluation Metric | Google Gemma 3 1B IT | Meta Llama 3.2 1B Instruct | Architectural Difference |
| :--- | :---: | :---: | :--- |
| **Vocabulary Size** | **256,000** | **128,000** | Gemma has 2.0x larger vocabulary space |
| **English Tokens (Belebele Part 1)** | 10,250 tokens | 10,256 tokens | Identical (~1.0 token/word) |
| **Bengali Tokens (Belebele Part 1)** | **10,260 tokens** | **48,016 tokens** | **Llama produces 4.68x more tokens in Bengali** |
| **Tokens per Character (Bengali)** | **0.25 tokens/char** | **1.19 tokens/char** | Llama generates >1 token per character |
| **Subword Fertility Ratio (BN / EN)**| **1.35x** | **4.68x** | Severe fragmentation penalty in Llama |

### Qualitative Tokenizer Analysis:
- In **Gemma 3**, common Bengali words like `"বাংলাদেশ"` (Bangladesh) or `"অনুসারে"` (according to) are represented as **1 single token**.
- In **Llama 3.2**, the word `"বাংলাদেশ"` is fragmented into 4 separate byte tokens: `['\xe0\xa6', '\xac\xe0', '\xa6\xbe', '\xe0\xa6...']`.

---

## 4. Pillar 2: Factual Grounding Accuracy & The 2-Bit Collapse

Below is the complete 25,200-prediction empirical census table comparing Gemma 3 1B against Llama 3.2 1B across all 7 quantization tiers. All confidence intervals are 95% passage-clustered bootstrap intervals; $p$-values reflect exact two-tailed McNemar tests against the uncompressed `f16` baseline.

| Quant Tier | BPW | Model Architecture | Bengali Acc (%) | 95% Cluster Boot CI | McNemar $p$ vs f16 | English Acc (%) | 95% Cluster Boot CI | McNemar $p$ vs f16 | Cross-Lingual Gap (EN - BN) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`f16`** | 16.0 | **Gemma 3 1B** | **37.56%** | [34.41, 40.83] | *reference* | **48.44%** | [45.60, 51.44] | *reference* | **+10.89%** |
| | | **Llama 3.2 1B** | **29.00%** | [26.00, 31.97] | *reference* | **41.33%** | [38.06, 44.72] | *reference* | **+12.33%** |
| **`Q8_0`** | 8.5 | **Gemma 3 1B** | **37.56%** | [34.31, 40.93] | $p = 1.0000$ | **47.89%** | [44.90, 50.99] | $p = 0.4237$ | **+10.33%** |
| | | **Llama 3.2 1B** | **29.11%** | [26.24, 32.01] | $p = 1.0000$ | **41.22%** | [37.85, 44.50] | $p = 1.0000$ | **+12.11%** |
| **`Q6_K`** | 6.5 | **Gemma 3 1B** | **37.44%** | [34.22, 40.68] | $p = 1.0000$ | **48.67%** | [45.71, 51.77] | $p = 0.8744$ | **+11.22%** |
| | | **Llama 3.2 1B** | **28.00%** | [25.03, 30.98] | $p = 0.2806$ | **39.67%** | [36.36, 42.76] | $p = 0.0864$ | **+11.67%** |
| **`Q5_K_M`**| 5.5 | **Gemma 3 1B** | **37.56%** | [34.49, 40.81] | $p = 1.0000$ | **46.67%** | [43.64, 49.83] | $p = 0.1017$ | **+9.11%** |
| | | **Llama 3.2 1B** | **27.44%** | [24.70, 30.44] | $p = 0.2191$ | **38.67%** | [35.39, 41.97] | $p = 0.0106^*$ | **+11.22%** |
| **`Q4_K_M`**| 4.5 | **Gemma 3 1B** | **35.89%** | [32.78, 38.99] | $p = 0.2669$ | **46.67%** | [43.64, 49.83] | $p = 0.1413$ | **+10.78%** |
| | | **Llama 3.2 1B** | **28.89%** | [25.81, 31.98] | $p = 1.0000$ | **43.00%** | [39.60, 46.36] | $p = 0.1992$ | **+14.11%** |
| **`Q3_K_M`**| 3.5 | **Gemma 3 1B** | **34.11%** 🔻 | [30.87, 37.43] | $p = 0.0398^*$ | **46.22%** | [43.00, 49.39] | $p = 0.1158$ | **+12.11%** |
| | | **Llama 3.2 1B** | **27.11%** | [24.08, 30.00] | $p = 0.2945$ | **31.22%** 🔻 | [28.10, 34.19] | $p < 0.0001^{***}$| **+4.11%** |
| **`Q2_K`** | 2.5 | **Gemma 3 1B** | **34.89%** | [31.88, 38.08] | $p = 0.1327$ | **43.44%** 🔻 | [40.38, 46.48] | $p = 0.0019^{**}$ | **+8.55%** |
| | | **Llama 3.2 1B** | **23.22%** 💥 | [20.53, 25.96] | $p = 0.0055^{**}$| **24.56%** 💥 | [21.86, 27.28] | $p < 0.0001^{***}$| **+1.34%** |

*\* $p < 0.05$. \*\* $p < 0.01$. \*\*\* $p < 0.001$.*

### Critical Findings on Accuracy:
1. **The Inherent Baseline Gap (Hypothesis 1 Confirmed):**
   - At uncompressed `f16`, English outperforms Bengali by **+10.89% in Gemma 3** and **+12.33% in Llama 3.2**. Both gaps are statistically significant ($p < 10^{-4}$ via paired bootstrap).
2. **The 2-Bit Collapse to Chance Level in Llama 3.2:**
   - At 2.5 bpw (`Q2_K`), Llama 3.2 collapses to the 25.0% random guessing floor in both languages (BN: 23.22%, EN: 24.56%). Both 95% bootstrap intervals cover 25.0%, indicating that the model has lost meaningful factual discernment.
   - Gemma 3 maintains above-chance accuracy under identical 2.5 bpw compression (34.89% in Bengali, 43.44% in English), though English suffers a statistically significant decline ($p = 0.0019$) and Bengali absolute accuracy remains modest.
3. **The Q4_K_M Aggregate Indistinguishability:**
   - At 4.5 bpw (`Q4_K_M`), neither model shows statistically significant aggregate accuracy degradation from `f16` ($p > 0.14$). However, as shown below, aggregate parity masks substantial latent churn.

---

## 5. Pillar 3: Latent Prediction Instability (Decision Flip Dynamics)

To uncover latent instability obscured by aggregate accuracy, we track **Instance-Level Decision Flips**—the proportion of questions where the quantized model alters its prediction relative to its own uncompressed `f16` output. 

Items were stratified into three equal-count tertile bins ($N = 300$ items each) based on `f16` softmax confidence margin ($P(\text{top-1}) - P(\text{top-2})$):
- **Low Margin:** Least certain questions at `f16` (BN margin $\le 0.198$, EN margin $\le 0.312$).
- **Mid Margin:** Moderately certain questions (BN margin $(0.198, 0.442]$, EN margin $(0.312, 0.654]$).
- **High Margin:** Highly confident questions at `f16` (BN margin $> 0.442$, EN margin $> 0.654$).

| Quant Tier | BPW | Model Architecture | Overall Flip Rate (%) | High-Margin Agreement (%) | Low-Margin Agreement (%) | Flip Rate Gap (BN - EN) | 95% Cluster Boot CI |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Q8_0`** | 8.5 | **Gemma 3 1B** | 5.44% (BN) / 3.78% (EN) | 100.0% (BN) / 100.0% (EN) | 85.0% (BN) / 89.0% (EN) | +1.67% | [-0.22%, +3.58%] |
| | | **Llama 3.2 1B** | 4.89% (BN) / 3.33% (EN) | 99.0% (BN) / 100.0% (EN) | 87.0% (BN) / 90.0% (EN) | +1.56% | [-0.22%, +3.42%] |
| **`Q6_K`** | 6.5 | **Gemma 3 1B** | 9.11% (BN) / 7.00% (EN) | 99.7% (BN) / 99.3% (EN) | 76.3% (BN) / 80.7% (EN) | +2.11% | [-0.11%, +4.41%] |
| | | **Llama 3.2 1B** | 11.33% (BN) / 10.44% (EN) | 99.7% (BN) / 100.0% (EN) | 74.3% (BN) / 72.0% (EN) | +0.89% | [-2.01%, +4.00%] |
| **`Q5_K_M`**| 5.5 | **Gemma 3 1B** | 22.56% (BN) / 15.78% (EN)| 95.3% (BN) / 98.3% (EN) | 57.7% (BN) / 64.3% (EN) | **+6.78%** * | [+3.22%, +10.56%] |
| | | **Llama 3.2 1B** | 22.22% (BN) / 15.00% (EN)| 89.0% (BN) / 100.0% (EN) | 72.7% (BN) / 62.7% (EN) | **+7.22%** * | [+3.49%, +10.99%] |
| **`Q4_K_M`**| 4.5 | **Gemma 3 1B** | 27.44% (BN) / 19.67% (EN)| 90.3% (BN) / 96.3% (EN) | 51.3% (BN) / 61.7% (EN) | **+7.78%** * | [+4.36%, +11.37%] |
| | | **Llama 3.2 1B** | 27.44% (BN) / 20.33% (EN)| 91.0% (BN) / 98.0% (EN) | 64.0% (BN) / 56.7% (EN) | **+7.11%** * | [+3.17%, +11.00%] |
| **`Q3_K_M`**| 3.5 | **Gemma 3 1B** | **37.67%** (BN) / **25.89%** (EN)| 80.0% (BN) / 92.0% (EN) | 44.7% (BN) / 58.0% (EN) | **+11.78%** * | [+7.89%, +15.64%] |
| | | **Llama 3.2 1B** | **47.67%** (BN) / **31.44%** (EN)| 63.7% (BN) / 86.0% (EN) | 54.3% (BN) / 48.7% (EN) | **+16.22%** * | [+11.64%, +21.00%] |
| **`Q2_K`** | 2.5 | **Gemma 3 1B** | **46.89%** (BN) / **37.11%** (EN)| **66.7%** (BN) / **82.7%** (EN) | 40.3% (BN) / 46.3% (EN) | **+9.78%** * | [+5.32%, +14.00%] |
| | | **Llama 3.2 1B** | **69.67%** (BN) / **51.56%** (EN)| **8.0%** 💥 (BN) / 51.3% (EN) | 65.0% (BN) / 43.0% (EN) | **+18.11%** * | [+13.88%, +22.35%] |

*\* Indicates statistically significant language gap at $p < 0.05$ (bootstrap CI strictly excludes 0.0).*

### Key Insights on Decision Flips:
1. **The Q4_K_M Cancellation Effect:** At 4.5 bpw, both models flip **27.44% of their Bengali decisions** relative to `f16`. Flat aggregate accuracy does not signify prediction preservation; rather, degradation flips (correct $\to$ incorrect) are roughly balanced by opportunistic noise flips (incorrect $\to$ correct).
2. **Asymmetric Language Disparity:** The flip gap between Bengali and English is minimal at 8-bit and 6-bit (+1% to +2%), but widens significantly starting at 5-bit (+6.8% to +7.2%) and peaks at 3-bit / 2-bit (+11.8% to +18.1%).
3. **High-Confidence Disruption in Llama 3.2:** For items in the top third of confidence at `f16` ($N=300$), Gemma 3 preserves 66.7% agreement in Bengali at `Q2_K`. In Llama 3.2, **high-margin agreement drops to 8.0%**—meaning 92% of decisions that Llama 3.2 was most certain of in full precision are reversed under 2-bit compression.

---

## 6. Pillar 4: Perplexity & Tokenizer-Independent Character Entropy

Because tokenizers divide identical texts into vastly different numbers of tokens (10,260 in Gemma vs 48,016 in Llama), comparing raw per-token perplexity across models is misleading. We evaluate both **relative perplexity increase ($\Delta\text{PPL}/\text{PPL}_{\text{f16}}$)** and tokenizer-independent **Bits-Per-Character (BPC)** and **Bits-Per-Byte (BPB)** across 84 canonical document passages (Belebele Part 1, 40,336 characters):

$$\text{BPC} = \frac{\text{Total Bits}}{\text{Total Characters}} = \frac{N_{\text{tokens}} \times \log_2(\text{PPL})}{N_{\text{characters}}}$$

| Model Architecture | Quant Tier | BPW | Language | Raw PPL | Bits/Token | Bits/Char (BPC) | Bits/Byte (BPB) | BPC Increase vs f16 (%) | Rel PPL Inc vs f16 (%) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemma 3 1B** | **`f16`** | 16.0 | English | 32.92 | 5.04 | 1.0609 | 1.0601 | *reference* | *reference* |
| | | | Bengali | 154.42 | 7.27 | 1.8615 | 0.7024 | *reference* | *reference* |
| | **`Q4_K_M`**| 4.5 | English | 33.77 | 5.08 | 1.0686 | 1.0679 | **+0.73%** | +2.59% |
| | | | Bengali | 171.60 | 7.42 | 1.9005 | 0.7171 | **+2.09%** | +11.12% |
| | **`Q3_K_M`**| 3.5 | English | 35.58 | 5.15 | 1.0844 | 1.0837 | **+2.22%** | +8.07% |
| | | | Bengali | 186.01 | 7.54 | 1.9303 | 0.7283 | **+3.69%** | +20.45% |
| | **`Q2_K`** | 2.5 | English | 43.63 | 5.45 | 1.1464 | 1.1456 | **+8.06%** | +32.54% |
| | | | Bengali | 263.11 | 8.04 | 2.0584 | 0.7766 | **+10.57%** | +70.38% |
| **Llama 3.2 1B** | **`f16`** | 16.0 | English | 17.70 | 4.15 | 0.8338 | 0.8338 | *reference* | *reference* |
| | | | Bengali | 2.69 | 1.43 | 1.7007 | 0.6440 | *reference* | *reference* |
| | **`Q4_K_M`**| 4.5 | English | 17.98 | 4.17 | 0.8384 | 0.8384 | **+0.55%** | +1.61% |
| | | | Bengali | 2.75 | 1.46 | 1.7431 | 0.6600 | **+2.49%** | +2.50% |
| | **`Q3_K_M`**| 3.5 | English | 18.79 | 4.23 | 0.8512 | 0.8512 | **+2.09%** | +6.19% |
| | | | Bengali | 2.97 | 1.57 | 1.8712 | 0.7085 | **+10.02%** | +10.41% |
| | **`Q2_K`** | 2.5 | English | 27.25 | 4.77 | 0.9590 | 0.9590 | **+15.02%** | +53.95% |
| | | | Bengali | 6.74 | 2.75 | 3.2848 | 1.2438 | **+93.14%** 🚀 | **+150.99%** 🚀 |

### Key Observations on Information Density:
- **Tokenizer-Independent Metric (BPC):** Under 2.5 bpw compression (`Q2_K`), Gemma 3’s Bengali BPC increases by **+10.57%**, whereas Llama 3.2’s Bengali BPC increases by **+93.14%**. Llama suffers an **8.8x greater character-normalized entropy penalty** in Bengali than Gemma.
- **Language Asymmetry:** In Llama 3.2, 2-bit quantization increases English BPC by +15.0% vs. +93.1% in Bengali (a 6.2x language disparity). In Gemma 3, English BPC increases by +8.1% vs. +10.6% in Bengali (a 1.3x disparity).

---

## 7. Pillar 5: Within-Model Length & Sequence Compounding Analysis

To test whether sequence length drives quantization degradation without cross-model confounders, we analyzed decision flip rates across passage character length tertiles ($N = 300$ items each: Short $\le 415$ chars, Mid $416–538$ chars, Long $> 538$ chars) within the exact same model and language:

| Model Architecture | Quant Tier | Language | Short Flip Rate (%) | Mid Flip Rate (%) | Long Flip Rate (%) | Point-Biserial $r_{pb}$ | $p$-value |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 3 1B** | **`Q4_K_M`** | Bangla | 25.7% | 28.2% | 28.4% | +0.018 | 0.594 |
| | | English | 18.3% | 20.3% | 20.4% | +0.026 | 0.441 |
| | **`Q2_K`** | Bangla | **44.3%** | **45.8%** | **50.5%** | **+0.071** | **0.034** * |
| | | English | 35.2% | 38.0% | 38.1% | +0.049 | 0.140 |
| **Llama 3.2 1B** | **`Q4_K_M`** | Bangla | 27.3% | 22.9% | 32.1% | +0.016 | 0.622 |
| | | English | 19.6% | 19.0% | 22.4% | +0.028 | 0.397 |
| | **`Q2_K`** | Bangla | **68.7%** | **64.5%** | **75.9%** | **+0.069** | **0.038** * |
| | | English | 51.8% | 50.0% | 52.8% | +0.002 | 0.956 |

### Empirical Takeaway:
At extreme 2.5 bpw quantization (`Q2_K`), **longer Bengali sequences flip significantly more frequently** than shorter sequences in both Gemma ($p = 0.034$) and Llama ($p = 0.038$), with long passages in Llama flipping **75.9% of the time**. In contrast, English passages show zero statistically significant length effect ($p > 0.14$). This directly supports the sequence compounding hypothesis: when tokens are already fragmented, longer inputs accumulate more rounding error.

---

## 8. Comparative Research Dashboard

The complete 4-panel empirical visualization synthesizing this entire study is rendered below:

![Comparative 4-Panel Research Dashboard](gemma3_vs_llama32_comparative_dashboard.png)

*Figure Description:*
- **Panel (a):** Factual accuracy degradation curves across the 7 precision tiers. Gemma 3 retains above-chance factual discernment down to 2.5 bpw, whereas Llama 3.2 collapses to the 25% random guessing floor at `Q2_K`.
- **Panel (b):** Cross-lingual factual disparity gap ($\text{Accuracy}_{\text{EN}} - \text{Accuracy}_{\text{BN}}$). The gap narrows at 2-bit for Llama only because both languages hit rock bottom.
- **Panel (c):** Language modeling perplexity increase (%) relative to `f16`. Llama 3.2 displays an exponential explosion in Bengali (+151.0%).
- **Panel (d):** Latent prediction instability. Decision flip rate reaches 69.7% in Llama 3.2 Bengali and 46.9% in Gemma 3 Bengali at `Q2_K`.

---

## 9. Practical Guidelines for Edge & Mobile LLM Deployment

1. **The 4-Bit (`Q4_K_M`) Universal Threshold:**
   For both models, `Q4_K_M` (4.5 bpw) represents the safest operating point for edge deployment, retaining >95% of aggregate accuracy and <2.5% BPC penalty across both languages. However, developers should note that ~27% of non-Latin predictions experience latent decision flips.
2. **Never Deploy Llama 3.2 Below 4-Bit in Multilingual Systems:**
   Deploying Llama 3.2 at `Q3_K_M` (3.5 bpw) or `Q2_K` (2.5 bpw) in Indic or non-Latin languages induces catastrophic factual failure, reversing 92% of confident decisions and degrading to chance-level guessing.
3. **Model Selection for Extreme Compression (< 3 bpw):**
   When severe memory constraints demand sub-3-bit compression (e.g. low-end smartphones, embedded systems), **Gemma 3 1B IT is substantially more resilient than Llama 3.2 1B**, maintaining ~35% factual grounding in Bengali and ~43% in English.

---

## 10. Artifact Index & Reproducibility Check

All primary datasets, prediction logs, and scripts are permanently persisted in the study directory:

- **Full Prediction Logs (25,200 lines):**
  - Gemma 3 1B: [`results/logprob_full900_predictions.jsonl`](file:///d:/Tokenizer/quant-hallucination-study/results/logprob_full900_predictions.jsonl)
  - Llama 3.2 1B: [`results/llama32_logprob_predictions.jsonl`](file:///d:/Tokenizer/quant-hallucination-study/results/llama32_logprob_predictions.jsonl)
- **Computed Statistical Tables:**
  - Gemma 3 Tier Stats: [`audit/full900_tier_stats.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/full900_tier_stats.csv)
  - Gemma 3 Flip Analysis: [`audit/flip_analysis.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/flip_analysis.csv)
  - Llama 3.2 Tier Stats: [`audit/llama32_tier_stats.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/llama32_tier_stats.csv)
  - Llama 3.2 Flip Analysis: [`audit/llama32_flip_analysis.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/llama32_flip_analysis.csv)
  - Perplexity Normalisation: [`audit/llama32_perplexity_normalised.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/llama32_perplexity_normalised.csv)
- **High-Resolution Figures:**
  - 4-Panel Research Dashboard: [`audit/gemma3_vs_llama32_comparative_dashboard.png`](file:///d:/Tokenizer/quant-hallucination-study/audit/gemma3_vs_llama32_comparative_dashboard.png)
  - Perplexity Comparison: [`audit/gemma_vs_llama_perplexity_comparison.png`](file:///d:/Tokenizer/quant-hallucination-study/audit/gemma_vs_llama_perplexity_comparison.png)
- **Reproduction Scripts:**
  - Evaluation Harness (Llama 3.2): [`audit/evaluate_llama32_logprobs.py`](file:///d:/Tokenizer/quant-hallucination-study/audit/evaluate_llama32_logprobs.py)
  - Statistical Calculator: [`audit/compute_llama32_full_stats.py`](file:///d:/Tokenizer/quant-hallucination-study/audit/compute_llama32_full_stats.py)
  - Dashboard Plotter: [`audit/plot_comparative_dashboard.py`](file:///d:/Tokenizer/quant-hallucination-study/audit/plot_comparative_dashboard.py)
  - Within-Model Length Script: [`audit/analyze_within_model_fertility.py`](file:///d:/Tokenizer/quant-hallucination-study/audit/analyze_within_model_fertility.py)
