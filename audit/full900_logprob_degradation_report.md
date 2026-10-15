# Gemma 3 1B IT — Full 900-Item Multilingual Quantization Degradation Audit

**Document Version:** 4.0 (Full 900-Item Census, N = 12,600 Total Predictions)  
**Target Model:** `google/gemma-3-1b-it` (256k Vocabulary, Subword Fertility Ratio = 1.35x)  
**Evaluation Engine:** Native Token Log-Likelihood ($\arg\max_{c \in \{1,2,3,4\}} \log P(c)$ via `llama-server` `logprobs=True`, `max_tokens=1`, assistant prefilled with `(`).  
**Benchmark:** Meta AI Belebele Parallel Benchmark (`data_v2/belebele_eval_parallel.jsonl`, all 900 parallel item pairs; 488 unique document passages).  
**Parsing Failure Rate:** **0.0% (Zero unparsed items across all 12,600 queries)**.  
**Primary Artifacts:**
* Prediction Logs: [`results/logprob_full900_predictions.jsonl`](file:///d:/Tokenizer/quant-hallucination-study/results/logprob_full900_predictions.jsonl) (12,600 lines)
* Statistical Table: [`audit/full900_tier_stats.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/full900_tier_stats.csv)
* Flip Analysis: [`audit/flip_analysis.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/flip_analysis.csv)
* Entropy & Distribution: [`audit/entropy_stats.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/entropy_stats.csv)
* Perplexity Normalisation: [`audit/perplexity_normalised.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/perplexity_normalised.csv)
* Multi-Panel Research Figure: [`audit/gemma3_full900_empirical_dashboard.png`](file:///d:/Tokenizer/quant-hallucination-study/audit/gemma3_full900_empirical_dashboard.png)

---

## 1. Executive Summary & Core Research Findings

This empirical audit investigates the cross-lingual impact of post-training quantization (PTQ) across all 7 standard GGUF tiers (`f16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K`) using Gemma 3 1B IT on 900 strictly parallel items in Bengali (BN) and English (EN).

### The Four Major Empirical Discoveries:
1. **The Factual Grounding Disparity (Baseline Gap):**
   * At uncompressed 16-bit precision (`f16`), English achieves **48.44%** accuracy while Bengali achieves **37.56%**, representing an inherent baseline cross-lingual gap of **+10.89%** (488-passage cluster bootstrap 95% CI: `[+7.38%, +14.59%]`).
2. **The Masking Effect of Surface Accuracy:**
   * Evaluating purely on overall accuracy creates the misleading impression that quantization degradation is mild and symmetric across languages: English drops from 48.44% to 43.44% at 2-bit, while Bengali drops from 37.56% to 34.89% at 2-bit.
   * However, statistical testing reveals that **Bengali experiences statistically significant accuracy drop earlier than English**: Bengali degrades significantly at `Q3_K_M` (McNemar $p = 0.0398$), whereas English remains statistically indistinguishable from `f16` until `Q2_K` (McNemar $p = 0.0019$).
3. **Latent Prediction Instability (Decision Flips from f16):**
   * Tracking instance-level prediction shifts against the `f16` baseline demonstrates severe cross-lingual disparity.
   * Under 4-bit compression (`Q4_K_M`), **27.44%** of Bengali items flip decisions vs. **19.67%** in English (Language Gap: **+7.78%**, 95% CI: `[+4.36%, +11.37%]`).
   * Under 3-bit compression (`Q3_K_M`), **37.67%** of Bengali items flip decisions vs. **25.89%** in English (Language Gap: **+11.78%**, 95% CI: `[+7.89%, +15.64%]`).
   * Even among high-confidence questions (top margin tertile), Bengali decisions flip in 20.0% of items at 3-bit, compared to only 8.0% in English.
4. **Disproportionate Language Modeling Degradation (Perplexity):**
   * Unconstrained passage perplexity reveals that quantization penalty falls disproportionately on Bengali text:
     * At `Q4_K_M`: Bengali PPL increases by **+11.12%** vs. only **+2.59%** in English (**4.3x worse** in Bengali).
     * At `Q3_K_M`: Bengali PPL increases by **+20.45%** vs. only **+8.07%** in English (**2.5x worse** in Bengali).
     * At `Q2_K`: Bengali PPL increases by **+70.38%** vs. **+32.54%** in English (**2.2x worse** in Bengali).
   * This degradation pattern holds robustly across 3 independent passage splits without exception.

---

## 2. Full 900-Item Factual Accuracy & Statistical Tests

All 900 items cluster around 488 unique document passages. Standard independent-sample intervals underestimate uncertainty; therefore, all confidence intervals and tests incorporate passage-clustered resampling (2,000 resamples, seed 42).

| Tier | Nominal BPW | Bengali Acc (%) | Wilson 95% CI | Passage Boot 95% CI | McNemar $p$ vs f16 | English Acc (%) | Wilson 95% CI | Passage Boot 95% CI | McNemar $p$ vs f16 | Cross-Lingual Gap (EN - BN) | Gap Boot 95% CI | $\Delta$ Gap vs f16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`f16`** | 16.0 | **37.56%** | [34.45, 40.77] | [34.41, 40.83] | *reference* | **48.44%** | [45.19, 51.71] | [45.60, 51.44] | *reference* | **+10.89%** | [+7.38, +14.59] | *reference* |
| **`Q8_0`** | 8.5 | **37.56%** | [34.45, 40.77] | [34.31, 40.93] | 1.0000 | **47.89%** | [44.64, 51.15] | [44.90, 50.99] | 0.4237 | **+10.33%** | [+6.89, +13.87] | -0.56% |
| **`Q6_K`** | 6.5 | **37.44%** | [34.34, 40.65] | [34.22, 40.68] | 1.0000 | **48.67%** | [45.41, 51.93] | [45.71, 51.77] | 0.8744 | **+11.22%** | [+7.67, +15.16] | +0.33% |
| **`Q5_K_M`** | 5.5 | **37.56%** | [34.45, 40.77] | [34.49, 40.81] | 1.0000 | **46.67%** | [43.43, 49.93] | [43.64, 49.83] | 0.1017 | **+9.11%** | [+5.56, +12.54] | -1.78% |
| **`Q4_K_M`** | 4.5 | **35.89%** | [32.82, 39.08] | [32.78, 38.99] | 0.2669 | **46.67%** | [43.43, 49.93] | [43.64, 49.83] | 0.1413 | **+10.78%** | [+7.04, +14.59] | -0.11% |
| **`Q3_K_M`** | 3.5 | **34.11%** 🔻 | [31.09, 37.27] | [30.87, 37.43] | **0.0398** * | **46.22%** | [42.99, 49.49] | [43.00, 49.39] | 0.1158 | **+12.11%** | [+8.55, +15.89] | +1.22% |
| **`Q2_K`** | 2.5 | **34.89%** | [31.85, 38.06] | [31.88, 38.08] | 0.1327 | **43.44%** 🔻 | [40.24, 46.70] | [40.38, 46.48] | **0.0019** ** | **+8.56%** | [+4.74, +12.28] | -2.33% |

*\* Statistically significant at $p < 0.05$. \*\* Statistically significant at $p < 0.01$.*

---

## 3. Decision Flips & Confidence Margin Stratification

We stratified the 900 items into 3 equal-count bins based on `f16` confidence margin ($P(\text{top-1}) - P(\text{top-2})$):
* **Low Margin:** Margins $\le 0.9996$ (items where the model is least certain).
* **Mid Margin:** Margins between $0.9996$ and $0.99999$.
* **High Margin:** Margins $> 0.99999$ (items where the model is extremely certain).

| Tier | Language | Overall Flip Rate (%) | Low Margin Agreement (%) | Mid Margin Agreement (%) | High Margin Agreement (%) | Accuracy Bottom 3rd (%) | Accuracy Top 3rd (%) | Flip Rate Gap (BN - EN) | Flip Gap Boot 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| **`f16`** | Bengali | 0.00% | 100.0% | 100.0% | 100.0% | 30.33% | 47.00% | 0.00% | [0.00, 0.00] |
| | English | 0.00% | 100.0% | 100.0% | 100.0% | 34.00% | 71.33% | | |
| **`Q8_0`** | Bengali | 5.44% | 85.00% | 98.67% | 100.0% | 30.00% | 47.00% | **+1.67%** | [-0.22, +3.58] |
| | English | 3.78% | 89.00% | 99.67% | 100.0% | 32.67% | 71.33% | | |
| **`Q6_K`** | Bengali | 9.11% | 76.33% | 96.67% | 99.67% | 30.33% | 47.00% | **+2.11%** | [-0.11, +4.41] |
| | English | 7.00% | 80.67% | 99.00% | 99.33% | 34.67% | 71.67% | | |
| **`Q5_K_M`** | Bengali | 22.56% 🔺 | 57.67% | 79.33% | 95.33% | 30.00% | 46.33% | **+6.78%** * | [+3.22, +10.56] |
| | English | 15.78% | 64.33% | 90.00% | 98.33% | 30.67% | 70.67% | | |
| **`Q4_K_M`** | Bengali | 27.44% 🔺 | 51.33% | 76.00% | 90.33% | 29.00% | 45.67% | **+7.78%** * | [+4.36, +11.37] |
| | English | 19.67% | 61.67% | 83.00% | 96.33% | 33.00% | 69.67% | | |
| **`Q3_K_M`** | Bengali | **37.67%** 🚀 | 44.67% | 62.33% | 80.00% | 27.67% | 43.00% | **+11.78%** * | [+7.89, +15.64] |
| | English | 25.89% | 58.00% | 72.33% | 92.00% | 33.00% | 70.33% | | |
| **`Q2_K`** | Bengali | **46.89%** 🚀 | 40.33% | 52.33% | 66.67% | 28.00% | 41.67% | **+9.78%** * | [+5.32, +14.00] |
| | English | 37.11% | 46.33% | 59.67% | 82.67% | 31.33% | 63.67% | | |

*\* Significant positive gap: Bengali flips significantly more often than English across all sub-6-bit tiers ($p < 0.001$).*

---

## 4. Perplexity Normalisation & Subword Efficiency

### Corpus Measurements:
* **Bengali:** 120,060 characters | 318,202 UTF-8 bytes | 30,739 Gemma tokens (Tokens/Byte: 0.0966, Bytes/Token: 10.35)
* **English:** 121,997 characters | 122,084 UTF-8 bytes | 25,675 Gemma tokens (Tokens/Byte: 0.2103, Bytes/Token: 4.75)

### Full Corpus Normalised Information Density:
| Tier | BPW | Language | Perplexity (PPL) | Bits per Token | Bits per Char (BPC) | Bits per Byte (BPB) | BPC Increase vs f16 (%) | Relative PPL Increase vs f16 (%) |
|---|---|---|---|---|---|---|---|---|
| **`f16`** | 16.0 | Bengali | 154.42 | 7.271 | 1.8615 | 0.7024 | *reference* | *reference* |
| | | English | 32.92 | 5.041 | 1.0609 | 1.0601 | *reference* | *reference* |
| **`Q8_0`** | 8.5 | Bengali | 153.72 | 7.264 | 1.8599 | 0.7017 | -0.09% | -0.45% |
| | | English | 32.75 | 5.033 | 1.0593 | 1.0585 | -0.15% | -0.52% |
| **`Q6_K`** | 6.5 | Bengali | 153.02 | 7.258 | 1.8582 | 0.7011 | -0.18% | -0.91% |
| | | English | 32.73 | 5.032 | 1.0591 | 1.0583 | -0.17% | -0.59% |
| **`Q5_K_M`** | 5.5 | Bengali | 154.15 | 7.268 | 1.8609 | 0.7021 | -0.03% | -0.18% |
| | | English | 32.33 | 5.015 | 1.0554 | 1.0546 | -0.52% | -1.79% |
| **`Q4_K_M`** | 4.5 | Bengali | **171.60** | 7.423 | 1.9005 | 0.7171 | **+2.09%** | **+11.12%** (4.3x of EN) |
| | | English | **33.77** | 5.078 | 1.0686 | 1.0679 | **+0.73%** | **+2.59%** |
| **`Q3_K_M`** | 3.5 | Bengali | **186.01** | 7.539 | 1.9303 | 0.7283 | **+3.69%** | **+20.45%** (2.5x of EN) |
| | | English | **35.58** | 5.153 | 1.0844 | 1.0837 | **+2.22%** | **+8.07%** |
| **`Q2_K`** | 2.5 | Bengali | **263.11** | 8.040 | 2.0584 | 0.7766 | **+10.57%** | **+70.38%** (2.2x of EN) |
| | | English | **43.63** | 5.447 | 1.1464 | 1.1456 | **+8.06%** | **+32.54%** |

### 3-Part Passage Split Robustness Check (Spread):
| Tier | Language | Part 1 PPL Inc vs f16 (%) | Part 2 PPL Inc vs f16 (%) | Part 3 PPL Inc vs f16 (%) | Spread Mean (%) |
|---|---|---|---|---|---|
| **`Q4_K_M`** | Bengali | +10.88% | +11.54% | +12.13% | **+11.52%** |
| | English | +3.23% | +1.00% | +1.86% | **+2.03%** |
| **`Q3_K_M`** | Bengali | +22.93% | +20.58% | +20.48% | **+21.33%** |
| | English | +6.82% | +9.49% | +4.63% | **+6.98%** |
| **`Q2_K`** | Bengali | +82.61% | +75.41% | +58.82% | **+72.28%** |
| | English | +31.00% | +35.28% | +27.90% | **+31.39%** |

---

## 5. Distribution Entropy & Predictive Uncertainty

| Tier | Language | Mean Entropy (nats) | Mean Entropy (bits) | Mean Max Softmax Probability | Mean $\ln P(\text{gold})$ (Clipped at $10^{-4}$) |
|---|---|---|---|---|---|
| **`f16`** | Bengali | 0.0498 | 0.0718 | 0.9825 | -4.9372 |
| | English | 0.0537 | 0.0775 | 0.9830 | -3.8163 |
| **`Q8_0`** | Bengali | 0.0586 | 0.0846 | 0.9801 | -4.9301 |
| | English | 0.0559 | 0.0806 | 0.9815 | -3.8250 |
| **`Q6_K`** | Bengali | 0.0464 | 0.0669 | 0.9851 | -4.9577 |
| | English | 0.0637 | 0.0919 | 0.9778 | -3.8087 |
| **`Q5_K_M`** | Bengali | 0.0553 | 0.0798 | 0.9801 | -4.9933 |
| | English | 0.0462 | 0.0666 | 0.9833 | -4.0907 |
| **`Q4_K_M`** | Bengali | 0.0702 🔺 | 0.1013 🔺 | 0.9740 | -4.8487 |
| | English | 0.0565 | 0.0815 | 0.9802 | -3.9661 |
| **`Q3_K_M`** | Bengali | 0.0885 🔺 | 0.1277 🔺 | 0.9695 | -4.6441 |
| | English | 0.0686 | 0.0990 | 0.9779 | -3.8622 |
| **`Q2_K`** | Bengali | **0.1814** 🚀 | **0.2617** 🚀 | 0.9348 | -4.0187 |
| | English | **0.1540** | **0.2222** | 0.9424 | -3.5553 |

---

## 6. Llama-3.2-1B Comparative Study Readiness

### Why Llama 3.2 1B is Crucial for Testing Hypothesis H2:
* **`google/gemma-3-1b-it`:** 256k Vocabulary, Subword Fertility Ratio = **1.35x** (Bengali 1.81 tokens/word, English 1.34 tokens/word).  
  *Finding:* Shields Bengali grammar from catastrophic accuracy collapse even at 2-bit (accuracy remains ~35%).
* **`meta-llama/Llama-3.2-1B-Instruct`:** 128k Vocabulary, Subword Fertility Ratio = **6.06x** (Bengali 8.02 tokens/word, English 1.32 tokens/word).  
  *Hypothesis:* Under identical bit-width compression, severe token fragmentation is expected to induce catastrophic breakdown (collapsing to random guess $\le 25\%$).

### Local Storage Check:
* GGUF models for Llama-3.2-1B are **currently not present** in `models/` or `models/gguf/`.
* Exact Hugging Face GGUF files needed for comparative execution:
  - `Llama-3.2-1B-Instruct-F16.gguf`
  - `Llama-3.2-1B-Instruct-Q8_0.gguf`
  - `Llama-3.2-1B-Instruct-Q6_K.gguf`
  - `Llama-3.2-1B-Instruct-Q5_K_M.gguf`
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
  - `Llama-3.2-1B-Instruct-Q3_K_M.gguf`
  - `Llama-3.2-1B-Instruct-Q2_K.gguf`
