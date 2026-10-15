# -*- coding: utf-8 -*-
"""Language x Length Interaction Analysis with Passage-Clustered Statistics.

Strict evaluation:
- Model: Y (flip indicator: 1 if pred_tier != pred_f16, else 0)
- Predictors: is_bangla (1 for BN, 0 for EN), passage length, and is_bangla * length interaction
- Metrics: Token count (under respective tokenizer) and Character count
- Passage clustering: 488 unique document passages (link URL)
- Cluster-robust standard errors via statsmodels
- 2000 cluster bootstrap resamples (seed 42) for interaction CI
- Holm-Bonferroni correction across the 6 quantized tiers
- Direct comparison between Gemma 3 (1.35x fertility) and Llama 3.2 (4.68x fertility)
"""

import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from transformers import AutoTokenizer

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
RESULTS_DIR = ROOT / "results"
AUDIT_DIR = ROOT / "audit"
PARALLEL_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"
GEMMA_FILE = RESULTS_DIR / "logprob_full900_predictions.jsonl"
LLAMA_FILE = RESULTS_DIR / "llama32_logprob_predictions.jsonl"

TIERS_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
BPW_MAP = {
    "Q8_0": 8.5,
    "Q6_K": 6.5,
    "Q5_K_M": 5.5,
    "Q4_K_M": 4.5,
    "Q3_K_M": 3.5,
    "Q2_K": 2.5
}

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = (m - rank) * p
        running_max = max(running_max, val)
        adj[orig_idx] = min(running_max, 1.0)
    return adj

def fast_logistic_fit(X: np.ndarray, y: np.ndarray, beta_init: np.ndarray = None, max_iter: int = 15) -> np.ndarray:
    """Fast Newton-Raphson logistic regression solver in numpy."""
    n, k = X.shape
    if beta_init is None:
        beta = np.zeros(k)
    else:
        beta = beta_init.copy()

    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30.0, 30.0)))
        w = p * (1.0 - p)
        w = np.clip(w, 1e-6, 0.25)
        # X^T W X
        XtW = X.T * w
        H = XtW @ X + 1e-5 * np.eye(k) # small ridge for numerical stability in bootstrap
        grad = X.T @ (y - p)
        try:
            delta = np.linalg.solve(H, grad)
            beta += delta
            if np.max(np.abs(delta)) < 1e-4:
                break
        except np.linalg.LinAlgError:
            break
    return beta

def load_data():
    print("Loading Belebele items and tokenizers...")
    tok_gemma = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    tok_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    items = []
    with open(PARALLEL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Precompute lengths
    item_meta = {}
    for it in items:
        pid = it["pair_id"]
        bn_p = it["bn"]["passage"]
        en_p = it["en"]["passage"]
        link = it.get("link", pid)
        
        g_bn_tok = len(tok_gemma.encode(bn_p, add_special_tokens=False))
        g_en_tok = len(tok_gemma.encode(en_p, add_special_tokens=False))
        l_bn_tok = len(tok_llama.encode(bn_p, add_special_tokens=False))
        l_en_tok = len(tok_llama.encode(en_p, add_special_tokens=False))
        
        bn_chars = len(bn_p)
        en_chars = len(en_p)

        item_meta[pid] = {
            "link": link,
            "char_len": {"bangla": bn_chars, "english": en_chars},
            "gemma_tok": {"bangla": g_bn_tok, "english": g_en_tok},
            "llama_tok": {"bangla": l_bn_tok, "english": l_en_tok},
        }

    # Load predictions
    def load_preds(path):
        preds = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pid = r.get("id") or r.get("pair_id")
                    opt = str(r.get("pred_option") or r.get("predicted_option")).strip()
                    preds[(r["tier"], r["language"], pid)] = opt
        return preds

    gemma_preds = load_preds(GEMMA_FILE)
    llama_preds = load_preds(LLAMA_FILE)

    return items, item_meta, gemma_preds, llama_preds

def run_interaction_analysis(model_name: str, preds_dict: dict, item_meta: dict, pair_ids: list[str], length_type: str = "tokens"):
    """Runs clustered logistic regression and cluster bootstrap for all 6 tiers."""
    rows = []
    
    # Identify unique clusters (488 passages)
    cluster_ids = [item_meta[pid]["link"] for pid in pair_ids]
    unique_clusters = np.array(sorted(list(set(cluster_ids))))
    cluster_to_idx = {c: [] for c in unique_clusters}

    tok_key = "gemma_tok" if "Gemma" in model_name else "llama_tok"

    # Pre-map cluster indices for N = 1800 rows (each pid has 2 rows: BN and EN)
    for i, pid in enumerate(pair_ids):
        link = item_meta[pid]["link"]
        # In a 1800 array ordered [all BN (900), all EN (900)]:
        # BN row index: i
        # EN row index: i + 900
        cluster_to_idx[link].extend([i, i + len(pair_ids)])

    for c in unique_clusters:
        cluster_to_idx[c] = np.array(cluster_to_idx[c], dtype=int)

    # 2000 cluster bootstrap resamples
    rng = np.random.RandomState(42)
    n_clusters = len(unique_clusters)
    n_resamples = 2000
    boot_cluster_choices = [rng.choice(unique_clusters, size=n_clusters, replace=True) for _ in range(n_resamples)]
    
    # Precompute sample row indices for all 2000 resamples
    boot_indices_list = [np.concatenate([cluster_to_idx[c] for c in chosen_clusters]) for chosen_clusters in boot_cluster_choices]

    p_values_to_correct = []
    model_tier_results = []

    scale_factor = 100.0

    for tier in TIERS_ORDER:
        bpw = BPW_MAP[tier]
        
        # Build arrays of length 1800: BN (0..899) and EN (900..1799)
        bn_lens = np.array([float(item_meta[pid][tok_key]["bangla"] if length_type == "tokens" else item_meta[pid]["char_len"]["bangla"]) for pid in pair_ids])
        en_lens = np.array([float(item_meta[pid][tok_key]["english"] if length_type == "tokens" else item_meta[pid]["char_len"]["english"]) for pid in pair_ids])
        
        bn_flips = np.array([1.0 if preds_dict[("f16", "bangla", pid)] != preds_dict[(tier, "bangla", pid)] else 0.0 for pid in pair_ids])
        en_flips = np.array([1.0 if preds_dict[("f16", "english", pid)] != preds_dict[(tier, "english", pid)] else 0.0 for pid in pair_ids])

        raw_length = np.concatenate([bn_lens, en_lens])
        flip = np.concatenate([bn_flips, en_flips])
        is_bangla = np.concatenate([np.ones(len(pair_ids)), np.zeros(len(pair_ids))])
        clusters_arr = np.concatenate([[item_meta[pid]["link"] for pid in pair_ids], [item_meta[pid]["link"] for pid in pair_ids]])

        len_scaled = raw_length / scale_factor
        interaction = is_bangla * len_scaled
        const = np.ones_like(len_scaled)

        X = np.column_stack([const, is_bangla, len_scaled, interaction])
        y = flip

        df_tier = pd.DataFrame({
            "const": const,
            "is_bangla": is_bangla,
            "len_scaled": len_scaled,
            "interaction": interaction,
            "flip": flip,
            "cluster": clusters_arr
        })

        # Fit cluster-robust logistic regression via statsmodels
        try:
            logit_mod = sm.Logit(df_tier["flip"], df_tier[["const", "is_bangla", "len_scaled", "interaction"]])
            res = logit_mod.fit(cov_type="cluster", cov_kwds={"groups": df_tier["cluster"]}, disp=False, maxiter=100)
            
            beta_const = res.params["const"]
            beta_lang = res.params["is_bangla"]
            beta_len = res.params["len_scaled"]
            beta_int = res.params["interaction"]

            se_int = res.bse["interaction"]
            pval_int = res.pvalues["interaction"]
            ci_int_lo = beta_int - 1.96 * se_int
            ci_int_hi = beta_int + 1.96 * se_int
            
            or_int = math.exp(beta_int) if abs(beta_int) < 50 else float("inf")
            or_ci_lo = math.exp(ci_int_lo) if abs(ci_int_lo) < 50 else 0.0
            or_ci_hi = math.exp(ci_int_hi) if abs(ci_int_hi) < 50 else float("inf")

            # Simple slope for Bangla: beta_len + beta_int
            cov_matrix = res.cov_params()
            var_bn_slope = cov_matrix.loc["len_scaled", "len_scaled"] + cov_matrix.loc["interaction", "interaction"] + 2 * cov_matrix.loc["len_scaled", "interaction"]
            se_bn_slope = math.sqrt(max(var_bn_slope, 1e-12))
            beta_bn_slope = beta_len + beta_int
            pval_bn_slope = 2.0 * (1.0 - stats.norm.cdf(abs(beta_bn_slope) / se_bn_slope))

            full_beta = np.array([beta_const, beta_lang, beta_len, beta_int])

        except Exception as e:
            print(f"[{model_name} {tier} {length_type}] Logit fit error: {e}")
            beta_const = beta_lang = beta_len = beta_int = se_int = pval_int = ci_int_lo = ci_int_hi = or_int = or_ci_lo = or_ci_hi = float("nan")
            beta_bn_slope = se_bn_slope = pval_bn_slope = float("nan")
            full_beta = np.zeros(4)

        # Fast vectorised cluster bootstrap for interaction coefficient
        boot_betas = []
        for b_idx in boot_indices_list:
            b_X = X[b_idx]
            b_y = y[b_idx]
            # Fast fit initialized at full sample beta
            b_beta = fast_logistic_fit(b_X, b_y, beta_init=full_beta, max_iter=8)
            boot_betas.append(b_beta[3]) # index 3 is interaction

        boot_betas = np.array(boot_betas)
        boot_int_lo = float(np.percentile(boot_betas, 2.5))
        boot_int_hi = float(np.percentile(boot_betas, 97.5))

        p_values_to_correct.append(pval_int)

        model_tier_results.append({
            "model": model_name,
            "tier": tier,
            "bpw": bpw,
            "length_metric": length_type,
            "scale_units": f"per {int(scale_factor)} {length_type}",
            "beta_interaction": round(beta_int, 4),
            "se_interaction_clustered": round(se_int, 4),
            "cluster_robust_ci_95": f"[{ci_int_lo:.4f}, {ci_int_hi:.4f}]",
            "cluster_boot_ci_95": f"[{boot_int_lo:.4f}, {boot_int_hi:.4f}]",
            "odds_ratio_interaction": round(or_int, 4),
            "odds_ratio_95_ci": f"[{or_ci_lo:.4f}, {or_ci_hi:.4f}]",
            "p_value_raw": pval_int,
            "beta_english_length": round(beta_len, 4),
            "se_english_length": round(res.bse["len_scaled"], 4) if 'res' in locals() else float("nan"),
            "p_english_length": round(res.pvalues["len_scaled"], 4) if 'res' in locals() else float("nan"),
            "beta_bangla_length": round(beta_bn_slope, 4),
            "se_bangla_length": round(se_bn_slope, 4),
            "p_bangla_length": round(pval_bn_slope, 4),
        })

    # Apply Holm-Bonferroni correction
    adj_p = holm_bonferroni(p_values_to_correct)
    for idx, r in enumerate(model_tier_results):
        r["p_value_holm"] = round(adj_p[idx], 4)
        r["p_value_raw"] = round(r["p_value_raw"], 4)

    return model_tier_results

def main():
    items, item_meta, gemma_preds, llama_preds = load_data()
    pair_ids = sorted([it["pair_id"] for it in items])
    total_n = len(pair_ids)
    print(f"Total items: {total_n} parallel pairs across {len(set(item_meta[p]['link'] for p in pair_ids))} passage clusters.\n")

    all_token_results = []
    all_char_results = []

    # Run for Gemma and Llama under both length metrics
    for model_name, p_dict in [("Gemma-3-1B-IT", gemma_preds), ("Llama-3.2-1B-Instruct", llama_preds)]:
        t0 = time.time()
        print(f"--- Running Token Length Interactions for {model_name} (6 tiers, 2000 cluster bootstraps) ---", flush=True)
        res_tok = run_interaction_analysis(model_name, p_dict, item_meta, pair_ids, length_type="tokens")
        all_token_results.extend(res_tok)
        print(f"    Completed in {time.time() - t0:.1f}s")

        t0 = time.time()
        print(f"--- Running Character Length Interactions for {model_name} (6 tiers, 2000 cluster bootstraps) ---", flush=True)
        res_char = run_interaction_analysis(model_name, p_dict, item_meta, pair_ids, length_type="chars")
        all_char_results.extend(res_char)
        print(f"    Completed in {time.time() - t0:.1f}s")

    df_tok = pd.DataFrame(all_token_results)
    df_char = pd.DataFrame(all_char_results)

    tok_csv = AUDIT_DIR / "length_interaction_tokens.csv"
    char_csv = AUDIT_DIR / "length_interaction_chars.csv"
    df_tok.to_csv(tok_csv, index=False)
    df_char.to_csv(char_csv, index=False)

    print(f"\nSaved Token Interaction CSV: {tok_csv}")
    print(f"Saved Character Interaction CSV: {char_csv}\n")

    # Generate comprehensive summary report
    summary_path = AUDIT_DIR / "length_interaction_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("TASK 3: LANGUAGE X LENGTH INTERACTION ANALYSIS WITH PASSAGE-CLUSTERED STATISTICS\n")
        f.write("=" * 110 + "\n")
        f.write(f"Total hypothesis tests run: 24 regressions (2 models x 2 metrics x 6 tiers)\n")
        f.write(f"Clusters: 488 unique document passages\n")
        f.write(f"Sample size per regression: 1,800 observations (900 Bangla + 900 English paired items)\n")
        f.write(f"Cluster bootstrap iterations: 2,000 resamples of 488 passages (seed 42)\n")
        f.write(f"Multiple testing adjustment: Holm-Bonferroni across the 6 tiers per model x metric family\n\n")

        f.write("-" * 110 + "\n")
        f.write("1. TOKEN LENGTH INTERACTION (per 100 model-specific tokens)\n")
        f.write("-" * 110 + "\n")
        header = f"{'Model':<22} | {'Tier':<6} | {'BPW':<4} | {'Beta_Int':<9} | {'SE(Clust)':<9} | {'Boot 95% CI':<19} | {'OR_Int':<8} | {'p(raw)':<8} | {'p(Holm)':<8} | {'BN Slope p':<10}"
        f.write(header + "\n")
        f.write("-" * 110 + "\n")
        for r in all_token_results:
            line = (f"{r['model']:<22} | {r['tier']:<6} | {r['bpw']:<4.1f} | {r['beta_interaction']:<+9.4f} | "
                    f"{r['se_interaction_clustered']:<9.4f} | {r['cluster_boot_ci_95']:<19} | {r['odds_ratio_interaction']:<8.4f} | "
                    f"{r['p_value_raw']:<8.4f} | {r['p_value_holm']:<8.4f} | {r['p_bangla_length']:<10.4f}")
            f.write(line + "\n")

        f.write("\n" + "-" * 110 + "\n")
        f.write("2. CHARACTER LENGTH INTERACTION (per 100 passage characters - constant physical text length)\n")
        f.write("-" * 110 + "\n")
        f.write(header + "\n")
        f.write("-" * 110 + "\n")
        for r in all_char_results:
            line = (f"{r['model']:<22} | {r['tier']:<6} | {r['bpw']:<4.1f} | {r['beta_interaction']:<+9.4f} | "
                    f"{r['se_interaction_clustered']:<9.4f} | {r['cluster_boot_ci_95']:<19} | {r['odds_ratio_interaction']:<8.4f} | "
                    f"{r['p_value_raw']:<8.4f} | {r['p_value_holm']:<8.4f} | {r['p_bangla_length']:<10.4f}")
            f.write(line + "\n")

        # Gemma vs Llama Comparison
        f.write("\n" + "=" * 110 + "\n")
        f.write("3. COMPARISON: GEMMA 3 (1.35x FERTILITY) VS LLAMA 3.2 (4.68x FERTILITY)\n")
        f.write("=" * 110 + "\n")
        f.write("Question: Is the Bengali length effect larger in Llama (where BN passage token count is ~4.7x larger)?\n\n")
        for tier in ["Q4_K_M", "Q3_K_M", "Q2_K"]:
            g_tok = next(x for x in all_token_results if x["model"] == "Gemma-3-1B-IT" and x["tier"] == tier)
            l_tok = next(x for x in all_token_results if x["model"] == "Llama-3.2-1B-Instruct" and x["tier"] == tier)
            g_char = next(x for x in all_char_results if x["model"] == "Gemma-3-1B-IT" and x["tier"] == tier)
            l_char = next(x for x in all_char_results if x["model"] == "Llama-3.2-1B-Instruct" and x["tier"] == tier)

            f.write(f"--- TIER: {tier} ({BPW_MAP[tier]} bpw) ---\n")
            f.write(f"  Token Metric (per 100 model-specific tokens):\n")
            f.write(f"    Gemma 3:   Beta_BN_slope = {g_tok['beta_bangla_length']:+.4f} (p = {g_tok['p_bangla_length']:.4f}) | Beta_Int = {g_tok['beta_interaction']:+.4f} (p_Holm = {g_tok['p_value_holm']:.4f})\n")
            f.write(f"    Llama 3.2: Beta_BN_slope = {l_tok['beta_bangla_length']:+.4f} (p = {l_tok['p_bangla_length']:.4f}) | Beta_Int = {l_tok['beta_interaction']:+.4f} (p_Holm = {l_tok['p_value_holm']:.4f})\n")
            f.write(f"  Character Metric (per 100 chars, identical physical length):\n")
            f.write(f"    Gemma 3:   Beta_BN_slope = {g_char['beta_bangla_length']:+.4f} (p = {g_char['p_bangla_length']:.4f}) | Beta_Int = {g_char['beta_interaction']:+.4f} (p_Holm = {g_char['p_value_holm']:.4f})\n")
            f.write(f"    Llama 3.2: Beta_BN_slope = {l_char['beta_bangla_length']:+.4f} (p = {l_char['p_bangla_length']:.4f}) | Beta_Int = {l_char['beta_interaction']:+.4f} (p_Holm = {l_char['p_value_holm']:.4f})\n\n")

    print(f"Generated comprehensive report at: {summary_path}")

    # Also print summary to terminal
    with open(summary_path, "r", encoding="utf-8") as f:
        print(f.read())

if __name__ == "__main__":
    main()
