# Cross-Lingual Impact of Quantization on Hallucination: A Comparison of Bangla and English in Compressed ~1B-Parameter Language Models (Revised Proposal, Draft v2)

> **Status:** This is a revised version of the original `project_proposal.md`. Decisions you need to confirm are marked `[Confirm]`.

---

## 1. Abstract

Post-training quantization (16-, 8-, and 4-bit) is now the standard way to run LLMs on mobile and offline devices. But lower numerical precision degrades output quality, and one of the most serious failure modes is **hallucination**. This relationship has been studied more for English, but to our knowledge it has not been studied at scale for low-resource, morphologically rich languages like Bangla (to be verified by a literature search before submission).

This study asks: **when compressed to the same bit-width, do the rate and type of hallucination change differently in Bangla and English, and does tokenizer efficiency mediate the difference?**

## 2. Problem Statement

- Research on quantization-induced quality degradation is dominated by English benchmarks.
- Bangla has less pretraining data and often weaker tokenizer efficiency, which is hypothesized, but not yet shown, to amplify quantization damage.
- Developers of Bangla mobile AI tools do not know "how much compression is safe."
- A model that looks fine in English may silently be less reliable for Bangla-speaking users.

## 3. Operational Definition of Hallucination (new)

The primary evaluation is **context-grounded**: the model receives a source passage and a question, and an answer that is unsupported by, or contradicts, the passage counts as a hallucination. This is because a ~1B model gets most closed-book questions wrong anyway, which would bury the effect of precision in noise.

A **secondary closed-book set** (same questions, no passage) will be kept only to check whether results point the same direction under both setups.

`[Confirm]` Your existing QA pairs were built from PDF sources, so the context-grounded setup appears directly applicable.

## 4. Objectives (revised)

1. Fine-tune a base model with bilingual LoRA on parallel-domain data (disaster, agriculture, education), and keep the un-tuned base model as a comparison baseline.
2. Quantize the fine-tuned model to seven GGUF levels (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K) so the shape of the degradation curve can actually be measured.
3. Build a held-out evaluation set of at least 300 parallel QA pairs, split at the document level and verified by native speakers.
4. Measure and compare the hallucination rate at each quantization level, separately for each language.
5. Classify hallucination types (factual substitution, fabricated entity, code-mixing/tokenizer breakdown, incoherence, ungrounded refusal/contradiction) to see whether the *nature*, not just the rate, differs across languages.
6. Test the language × precision interaction, and whether a "degradation threshold" exists and differs by language.
7. **Test the role of tokenizer efficiency:** run a second model with a less efficient Bangla tokenizer (e.g., Llama 3.2 1B) through the same pipeline to see whether the language gap relates to tokenizer fertility.
8. (Optional, final stage) An on-device mobile demo of the 4-bit model.

## 5. Hypotheses (fixed before seeing data)

- **H0:** The increase in hallucination rate caused by quantization is the same in Bangla and English (interaction = 0).
- **H1:** Going from FP16 to Q4_K_M raises the hallucination rate more in Bangla than in English (interaction > 0).
- **H2:** The language gap is larger in the model with the less efficient tokenizer (three-way interaction: language × precision × model).
- **H3:** As precision drops, the distribution of hallucination types shifts toward code-mixing and incoherence in Bangla, and toward factual substitution in English.

`[Confirm]` Whether to label H3 as exploratory, since it is the most speculative.

## 6. Statistical Plan (new)

- **Primary model:** mixed-effects logistic regression on the binary per-item label (hallucinated / not): `hallucinated ~ language * precision + (1 | question_id)`. With question ID as a random effect, the Bangla-English pairing and repeated measurement of each question across precisions are handled correctly.
- **Key statistic:** the coefficient and confidence interval of the language × precision interaction.
- **Supporting tests:** McNemar or paired bootstrap on paired questions; bootstrap CIs on degradation curves.
- **Threshold analysis:** piecewise or monotone regression over the seven points to distinguish a "sharp drop" from a "gradual drop."
- **Sizing:** a pilot and power analysis will be run before fixing the number of items.
- **Multiple comparisons:** keep pre-specified hypotheses few and label exploratory analyses clearly.
- **Decoding:** greedy for primary results; a few sampled seeds separately for variance.

## 7. Methodology

### 7.1 Data
- Domains: disaster, agriculture, education. Bangla and English versions of each domain must match in size and quality.
- The **source of the Bangla portion** (original Bangla, machine translation, or LLM-generated) must be stated explicitly, since it is a direct confound. A native speaker will verify the Bangla version of every eval item.
- **Split:** document-level train/dev/eval split, stratified by domain. Questions from one document never appear in two splits, so fine-tuned facts cannot leak into eval.
- Parallel question IDs are identical across both languages and all splits.

### 7.2 Models
- **Primary model:** `google/gemma-3-1b-it` (Bangla/English fertility ratio 1.41, i.e., relatively favorable for Bangla).
- **Second model:** Llama 3.2 1B (fertility 7.17), used only to test the tokenizer role. `[Confirm]` Given compute limits, the second model could initially run fewer quantization levels (e.g., F16, Q8_0, Q4_K_M, Q2_K).
- Results are reported separately by model and tokenizer.

### 7.3 Fine-tuning
- Bilingual QLoRA (one shared adapter, interleaved data from both languages), plus per-language ablations.
- **Baseline:** the un-tuned base model, since fine-tuning on little data mostly teaches format, not knowledge.
- The LoRA adapter is merged into FP16 to form a single checkpoint, and all quantizations are made from that checkpoint.

### 7.4 Quantization
- **Primary path:** GGUF (llama.cpp). Record the exact llama.cpp commit and conversion flags.
- **Secondary path:** bitsandbytes (INT8/NF4), for supplementary comparison only. It is a different scheme from k-quants, so it is shown as a separate axis.
- Verifying that the Gemma-3 chat template is identical in the HF and GGUF paths is mandatory.
- Q8_0 vs F16 is treated as a near-null control.

### 7.5 Evaluation and Scoring
- **Primary ground truth:** human labels. Three annotators, blinded to precision and language ordering, with randomized item order.
- **Pilot:** test the rubric on about 30 items to fix the boundary between categories 3 and 4, then run full annotation.
- **Agreement:** Fleiss' kappa (three raters), with a conflict-resolution rule.
- **Automatic metrics** (ROUGE, embedding similarity, faithfulness/NLI): used only for triage and supplementary analysis, because their noise is unequal across languages. How "faithfulness" is computed (which NLI model, which threshold) must be defined in advance, and that model's Bangla performance checked separately.
- Latency and memory are recorded with every output.

## 8. Risks and Limitations

| Risk | Mitigation |
|---|---|
| Eval set too small | 300+ items; power analysis; report with CIs |
| Translation errors in Bangla data | Native speaker verification; disclose sourcing method |
| Language bias in automatic metrics | Human labels primary, metrics supplementary |
| Low annotator agreement | Pilot, explicit rubric, annotator training |
| Overfitting or leakage in fine-tuning | Document-level split; base-model baseline |
| Compute limits (4 GiB GPU) | 1B QLoRA; fewer levels on second model; Colab backup |
| Only two models and one domain set | Keep generalization claims modest |

## 9. Expected Contributions

1. A native-verified parallel Bangla-English hallucination benchmark (data and rubric).
2. Seven-level quantization degradation curves on the same document set, by language and model.
3. A comparative taxonomy of hallucination types in Bangla and English.
4. Preliminary evidence on whether tokenizer efficiency mediates language-specific quantization damage.
5. (Optional) On-device implementation guidance.

> **Wording of claims:** Instead of "the first Bangla quantization study," write the contribution conservatively after verifying against the literature, for example: "a Bangla-English comparison of quantization and hallucination with a native-verified benchmark."

## 10. Suggested Timeline (example)

| Stage | Work |
|---|---|
| 1 | Literature search; finalize hypotheses and statistical plan |
| 2 | Expand data, document-level split, native verification |
| 3 | Pilot: annotate 30 items, finalize rubric, power analysis |
| 4 | Bilingual fine-tune, merge, seven-level quantization |
| 5 | Run evaluation (primary model), automatic scoring |
| 6 | Human annotation, agreement, analysis |
| 7 | Second model, tokenizer analysis |
| 8 | Writing; optional mobile demo |

## 11. Required Related Changes to `implementation_plan.md`

- `compile_approved.py`: **document-level** split instead of question-level; domain stratification; identical IDs.
- `quantize_models.py`: seven GGUF levels instead of three.
- `evaluate_models.py`: context-grounded and closed-book prompt modes; base-model baseline; model parameter.
- `score_hallucinations.py`: precise definition of the faithfulness method.
- `plot_degradation_analysis.py`: mixed-effects model and interaction CIs; seven-point curves.
- `data/eval_rubric.md`: clarify the boundary between categories 3 and 4.