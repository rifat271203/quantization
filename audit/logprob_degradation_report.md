# Gemma 3 1B IT — 7-Tier Quantization Degradation Audit (Logprob Scoring)

**Evaluation Method:** Native Token Log-Likelihood ($rg\max_{c \in \{1,2,3,4\}} \log P(c)$ via `llama-server` `logprobs=True`, `max_tokens=1`, assistant prefilled with `(`).  
**Benchmark:** Meta AI Belebele Parallel Benchmark (300 paired items in Bangla and English).  
**Parsing Failure Rate:** **0.0% (Zero unparsed items across all 4,200 queries)**.  
**Predictions Log:** [`results/logprob_7tier_predictions.jsonl`](file:///d:/Tokenizer/quant-hallucination-study/results/logprob_7tier_predictions.jsonl)  
**Summary JSON:** [`results/logprob_7tier_summary.json`](file:///d:/Tokenizer/quant-hallucination-study/results/logprob_7tier_summary.json)  
**Statistical CSV:** [`audit/logprob_tier_stats.csv`](file:///d:/Tokenizer/quant-hallucination-study/audit/logprob_tier_stats.csv)  

---

### Comparative Degradation Audit Table

| Tier | Nominal BPW | Bangla Accuracy (Wilson 95% CI) | McNemar $p$ vs f16 | English Accuracy (Wilson 95% CI) | McNemar $p$ vs f16 | Cross-Lingual Gap (EN - BN, Boot 95% CI) |
|---|---|---|---|---|---|---|
| **f16** | 16.0 | 32.3% ([27.3, 37.8]) | ref | 47.3% ([41.8, 53.0]) | ref | +15.0% ([8.3, 21.7]) |
| **Q8_0** | 8.5 | 33.0% ([27.9, 38.5]) | 0.7237 | 45.7% ([40.1, 51.3]) | 0.1824 | +12.7% ([6.3, 19.0]) |
| **Q6_K** | 6.5 | 32.0% ([27.0, 37.5]) | 1.0000 | 46.7% ([41.1, 52.3]) | 0.7518 | +14.7% ([8.3, 20.7]) |
| **Q5_K_M** | 5.5 | 34.3% ([29.2, 39.9]) | 0.4047 | 47.3% ([41.8, 53.0]) | 1.0000 | +13.0% ([6.7, 19.7]) |
| **Q4_K_M** | 4.5 | 32.7% ([27.6, 38.2]) | 1.0000 | 44.7% ([39.1, 50.3]) | 0.2433 | +12.0% ([5.7, 18.3]) |
| **Q3_K_M** | 3.5 | 30.3% ([25.4, 35.8]) | 0.5040 | 48.0% ([42.4, 53.6]) | 0.8918 | +17.7% ([11.3, 24.0]) |
| **Q2_K** | 2.5 | 33.7% ([28.6, 39.2]) | 0.7160 | 45.3% ([39.8, 51.0]) | 0.5443 | +11.7% ([5.0, 18.3]) |

