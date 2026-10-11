# Evaluation Rubric & Hallucination Taxonomy

This rubric establishes the standardized criteria for evaluating factual correctness and categorizing hallucinations across quantized language model outputs (16-bit down to 2-bit GGUF tiers) in Bangla and English.

---

## 1. Overall Correctness Scale (0 - 2)

| Score | Rating | Operational Definition |
|:-----:|:-------|:-----------------------|
| **2** | **Fully Correct / Grounded** | The answer directly and accurately conveys the ground truth fact with no fabricated or unsupported details. In context-grounded mode, all assertions are directly verifiable from the provided passage. Minor stylistic variation is permitted. |
| **1** | **Partially Correct** | The answer contains correct core elements but includes minor inaccuracies, omission of key constraints, or tangential non-factual commentary. |
| **0** | **Hallucinated / Incorrect** | The answer makes factual claims contradicted by the ground truth or passage, invents entities/figures, suffers linguistic collapse, or is completely unsupported. |

---

## 2. Hallucination Taxonomy (Categories)

When an output receives a correctness score of `0` or `1`, it is assigned to one or more of the following 5 distinct error categories:

### Category 1: Factual Substitution (`FACT_SUB`)
- **Definition**: The model replaces a specific entity, numeric value, date, role, or relationship with another plausible but incorrect entity.
- **Example (EN)**: Ground truth says "Cabinet Secretary is the member-secretary", but model outputs "The Home Minister is the member-secretary".
- **Example (BN)**: প্রশ্নে "প্রধান নির্বাহী কে?", মডেলে "সচিব" বলে যেখানে সঠিক উত্তর "মহাপরিচালক"।

### Category 2: Fabricated Entities (`FAB_ENT`)
- **Definition**: The model invents completely non-existent legal clauses, committees, departments, scientific laws, or citations not present in reality or the source text.
- **Example (EN)**: Inventing an Act titled "National Cyclone Regulation Protocol of 2005".
- **Example (BN)**: সম্পূর্ণ কাল্পনিক কোনো পদবী বা আইন ধারা তৈরি করা (যেমন: "ধারা ৪৯(ক) অনুসারে দুর্যোগ পুলিশ গঠন")।

### Category 3: Code-Mixing & Tokenizer Breakdown (`CODE_MIX_TOK`)
- **Definition**: Inability to maintain language purity or tokenizer decoding failure. Symptoms include sudden mid-sentence switching to English in a Bangla prompt, repetitive subword token loops, broken Unicode grapheme clusters (যুক্তবর্ণ বিকৃতি), or byte fallbacks / garbled glyphs.
- **Example (BN)**: "কাউন্সিলের quorum হয় two-third members attend করলে..." বা "দুর্যোগgg-য়োগgg..." বা Broken conjuncts (`ক` `্` `ষ` বিচ্ছিন্ন হওয়া).

### Category 4: Incoherent Output / Semantic Degeneration (`INCOHERENT`)
- **Definition**: Structural syntax collapse, phrase-level infinite repetition loops, word salad, or self-contradictory logic where text ceases to form sensible communication, even if individual words are spelled correctly.
- **Example**: Generating repeated sequences ("এবং এটি কার্যকর হয় এবং এটি কার্যকর হয়...") or circular nonsensical syntax.

### Category 5: Ungrounded Refusal / False Contradiction (`UNGROUNDED_REFUSAL`)
- **Definition**: The model incorrectly claims the information is not present in the passage, claims the question cannot be answered, or outputs an unfounded ethical refusal despite the query being factual and benign.

---

## 3. Boundary & Disambiguation Guide: Category 3 vs. Category 4

Because quantization degradation often manifests near the border between tokenizer subword degradation and semantic incoherence, annotators must adhere to the following boundary rules:

| Criterion | Category 3: `CODE_MIX_TOK` | Category 4: `INCOHERENT` |
|:---|:---|:---|
| **Locus of Failure** | **Subword / Tokenizer / Lexicon**: Individual tokens are corrupt, split, garbled, or borrowed from English. | **Syntax / Discourse / Semantics**: High-level grammatical structure collapses; words make no semantic sense together. |
| **Token Validity** | Contains malformed tokens, broken conjuncts, subword fragments (`##`, ` `), or unexpected cross-lingual code-switches. | Individual words are valid dictionary words, but their syntactic combination is gibberish or repetitive loop. |
| **Repetition Type** | **Morpheme/Subword Loop**: e.g., "পরি-রি-রি-চালনা" or "cyclone-one-one". | **Phrase/Clause Loop**: e.g., "সরকার এই আদেশ জারি করেছে। সরকার এই আদেশ জারি করেছে।". |
| **Language Mixing** | Mid-sentence language drift where English words replace standard Bangla equivalents. | Grammatically nonsensical sequence even within a single language. |

### Disambiguation Decision Tree:
1. **Did the model switch languages mid-sentence (e.g. Bangla to English)?**  
   $\rightarrow$ Mark **`CODE_MIX_TOK`**.
2. **Are there broken Bengali conjuncts (যুক্তবর্ণ বিকৃতি) or garbage characters?**  
   $\rightarrow$ Mark **`CODE_MIX_TOK`**.
3. **Are words well-formed Bengali/English words, but strung together as meaningless word salad?**  
   $\rightarrow$ Mark **`INCOHERENT`**.
4. **Is the output trapped in an infinite loop of complete words or sentences?**  
   $\rightarrow$ Mark **`INCOHERENT`**.
5. **Both subword breakdown and syntax collapse present?**  
   $\rightarrow$ Assign primary label based on the root trigger (usually **`CODE_MIX_TOK`** if subwords fractured first), or assign both: `[CODE_MIX_TOK, INCOHERENT]`.

---

## 4. Human Annotation Protocol

1. **Blinded Evaluation**: Human raters are provided with `id`, `instruction`, `passage` (if grounded), `ground_truth_response`, and `model_response`. The model name, language adapter, and bit-width precision are masked (`model_A`, `model_B`, etc.) to prevent bias.
2. **Independent Scoring**: Each evaluator independently assigns:
   - `correctness` (0, 1, or 2)
   - `error_category` (None, FACT_SUB, FAB_ENT, CODE_MIX_TOK, INCOHERENT, UNGROUNDED_REFUSAL)
   - `notes` (brief justification)
3. **Inter-Rater Reliability**: Computed using Cohen's Kappa ($\kappa$) for pairs of raters and Fleiss' Kappa for team-wide consensus. A threshold of $\kappa \ge 0.70$ signifies substantial agreement.
