# -*- coding: utf-8 -*-
"""Flip Noise-Floor & Option-Permutation Harness and Statistical Analysis.

Implements all 4 cyclic shifts:
- Shift 0 (Identity): 1->1, 2->2, 3->3, 4->4
- Shift 1 (+1 Offset): 1->2, 2->3, 3->4, 4->1
- Shift 2 (+2 Offset): 1->3, 2->4, 3->1, 4->2
- Shift 3 (+3 Offset): 1->4, 2->1, 3->2, 4->3

Server is automatically launched, checked, and managed.
Results are saved to results/noise_floor_{model}_f16_shift{s}.jsonl.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
PARALLEL_FILE = ROOT / "data_v2" / "belebele_eval_parallel.jsonl"
RESULTS_DIR = ROOT / "results"
AUDIT_DIR = ROOT / "audit"
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"

ORIGINAL_F16_FILES = {
    "gemma": RESULTS_DIR / "logprob_full900_predictions.jsonl",
    "llama": RESULTS_DIR / "llama32_logprob_predictions.jsonl"
}

MODEL_PATHS = {
    "gemma": ROOT / "models" / "gguf" / "gemma-3-1b-it-f16.gguf",
    "llama": ROOT / "models" / "gguf" / "Llama-3.2-1B-Instruct-F16.gguf"
}

ORIGINAL_CONFIGS = {
    "gemma": {"threads": 10, "ubatch": 512, "batch": 2048, "ctx": 2048, "prefill": True},
    "llama": {"threads": 10, "ubatch": 512, "batch": 2048, "ctx": 2048, "prefill": False}
}

CANDIDATE_DIGITS = ["1", "2", "3", "4"]
BN_TO_EN_DIGIT = {"১": "1", "২": "2", "৩": "3", "৪": "4"}

def parse_args():
    p = argparse.ArgumentParser(description="Flip Noise Floor & Option Permutation Harness")
    p.add_argument("--dry-run", action="store_true", help="Print exact server commands and request counts without running")
    p.add_argument("--model", choices=["gemma", "llama", "both"], default="gemma", help="Model to evaluate")
    p.add_argument("--mode", choices=["option_permutation", "server_perturbation", "analyze"], default="option_permutation", help="Execution mode")
    p.add_argument("--shift", type=int, choices=[0, 1, 2, 3], default=1, help="Cyclic shift offset (0, 1, 2, or 3)")
    p.add_argument("--shifts", nargs="+", type=int, default=None, help="Multiple shifts to run sequentially (e.g. --shifts 1 2 3)")
    p.add_argument("--port", type=int, default=8190, help="Port for llama-server")
    p.add_argument("--threads", type=int, default=10, help="CPU threads for llama-server")
    p.add_argument("--out", type=Path, default=None, help="Output JSONL file path override")
    p.add_argument("--limit", type=int, default=None, help="Limit number of items for testing")
    return p.parse_args()

def get_cyclic_shift_map(shift: int) -> dict[str, str]:
    """Returns mapping from new display label -> original option label."""
    return {str(k): str(((k - 1 + shift) % 4) + 1) for k in range(1, 5)}

def format_user_prompt(passage: str, instruction: str) -> str:
    return (
        "Read the following passage carefully and answer the question based ONLY on facts directly stated "
        "in the passage.\n\n"
        f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
        f"{instruction.strip()}\n\n"
        "Answer with option number only:"
    )

def logsumexp(a: float, b: float) -> float:
    if a == float("-inf") and b == float("-inf"):
        return float("-inf")
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def extract_candidate_logprobs(top_logprobs: list[dict]) -> tuple[dict[str, float], str, float, float]:
    candidate_lps = {d: -99.0 for d in CANDIDATE_DIGITS}
    for entry in top_logprobs:
        raw_token = entry.get("token", "")
        clean_t = raw_token.strip()
        clean_t = BN_TO_EN_DIGIT.get(clean_t, clean_t)
        if clean_t in candidate_lps:
            lp = float(entry["logprob"])
            if candidate_lps[clean_t] == -99.0:
                candidate_lps[clean_t] = lp
            else:
                candidate_lps[clean_t] = logsumexp(candidate_lps[clean_t], lp)

    sorted_candidates = sorted(candidate_lps.items(), key=lambda x: x[1], reverse=True)
    best_option, best_lp = sorted_candidates[0]
    second_best_lp = sorted_candidates[1][1]
    confidence_margin = best_lp - second_best_lp

    max_lp = max(candidate_lps.values())
    sum_exp = sum(math.exp(lp - max_lp) for lp in candidate_lps.values())
    log_denom = max_lp + math.log(sum_exp)
    
    entropy = 0.0
    for lp in candidate_lps.values():
        norm_p = math.exp(lp - log_denom)
        if norm_p > 1e-12:
            entropy -= norm_p * math.log2(norm_p)

    return candidate_lps, best_option, round(confidence_margin, 4), round(entropy, 4)

def wait_for_server(port: int, timeout_sec: int = 60) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def check_server_already_running(port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False

def query_server_logprobs(prompt: str, port: int, prefill: bool = False, max_retries: int = 3, timeout_sec: int = 120) -> tuple[list[dict], float]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    
    token_limits = [1, 2, 4] if not prefill else [1]
    for max_tok in token_limits:
        messages = [{"role": "user", "content": prompt}]
        if prefill:
            messages.append({"role": "assistant", "content": "("})

        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tok,
            "logprobs": True,
            "top_logprobs": 20
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        last_err = None
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    res = json.loads(resp.read().decode("utf-8", errors="replace"))
                lat_ms = (time.perf_counter() - t0) * 1000.0
                choice = res["choices"][0]
                lp_obj = choice.get("logprobs")
                if lp_obj and lp_obj.get("content") and len(lp_obj["content"]) > 0:
                    top_lp = lp_obj["content"][0].get("top_logprobs", [])
                    return top_lp, lat_ms
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0)
                
    if last_err is not None and not isinstance(last_err, KeyError):
        raise RuntimeError(f"Server query failed after {max_retries} attempts: {last_err}")
    return [], 0.0

def run_option_permutation_single_shift(model_key: str, shift: int, port: int = 8190, threads: int = 10, out_override: Path = None, limit: int = None):
    """Runs option permutation evaluation for a specific cyclic shift."""
    model_path = MODEL_PATHS[model_key]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cfg = ORIGINAL_CONFIGS[model_key]
    perm_map = get_cyclic_shift_map(shift)
    out_file = out_override or (RESULTS_DIR / f"noise_floor_{model_key}_f16_shift{shift}.jsonl")

    print("\n" + "=" * 90)
    print(f"RUNNING OPTION PERMUTATION: Model={model_key.upper()} | Shift={shift} | Port={port}")
    print(f"Display Label Permutation (New Label -> Original Option): {perm_map}")
    print(f"Target Output: {out_file}")
    print("=" * 90)

    # 1. Load dataset
    items = []
    with open(PARALLEL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if limit:
        items = items[:limit]

    # Check already completed keys
    completed_keys = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        completed_keys.add((r["language"], r["pair_id"]))
                    except Exception:
                        pass
        print(f"Found {len(completed_keys)} existing completed items in {out_file.name}. Resuming...")

    # Start or attach to llama-server
    we_started_server = False
    proc = None
    if check_server_already_running(port):
        print(f"Using server already running on port {port}.")
    else:
        cmd = [
            str(LLAMA_SERVER),
            "-m", str(model_path),
            "-c", str(cfg["ctx"]),
            "-t", str(threads),
            "-b", str(cfg["batch"]),
            "-ub", str(cfg["ubatch"]),
            "--port", str(port),
            "--log-disable"
        ]
        print(f"Starting llama-server: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        we_started_server = True
        time.sleep(2)
        if not wait_for_server(port, timeout_sec=60):
            proc.terminate()
            raise RuntimeError(f"Server on port {port} failed to start!")
        print("Server is ready and healthy!")

    try:
        with open(out_file, "a", encoding="utf-8") as out_f:
            for lang in ["bangla", "english"]:
                needed = [it for it in items if (lang, it["pair_id"]) not in completed_keys]
                print(f"\n--- Language: {lang.upper()} ({len(needed)} items to score) ---", flush=True)
                for idx, it in enumerate(needed, 1):
                    pid = it["pair_id"]
                    passage = it[("bn" if lang == "bangla" else "en")]["passage"]
                    question = it[("bn" if lang == "bangla" else "en")]["question"]
                    orig_opts = it[("bn" if lang == "bangla" else "en")]["options"]
                    gold = str(it["correct_answer_num"]).strip()

                    # Format permuted options
                    perm_opts_dict = {new_k: orig_opts[orig_k] for new_k, orig_k in perm_map.items()}
                    opts_str = "\n".join([f"({k}) {v}" for k, v in perm_opts_dict.items()])
                    instruction = f"{question}\n\nবিকল্পসমূহ:\n{opts_str}" if lang == "bangla" else f"{question}\n\nOptions:\n{opts_str}"
                    prompt = format_user_prompt(passage, instruction)

                    top_lp, lat_ms = query_server_logprobs(prompt, port, prefill=cfg["prefill"])
                    cand_lps, best_opt, margin, entropy = extract_candidate_logprobs(top_lp)
                    
                    # Map best_opt back to original label space
                    mapped_pred_opt = perm_map.get(best_opt, best_opt)
                    is_corr = (mapped_pred_opt == gold)

                    record = {
                        "model": model_key,
                        "tier": "f16",
                        "shift": shift,
                        "language": lang,
                        "pair_id": pid,
                        "raw_display_pred": best_opt,
                        "mapped_pred_option": mapped_pred_opt,
                        "gold_option": gold,
                        "is_correct": is_corr,
                        "confidence_margin": margin,
                        "entropy": entropy,
                        "latency_ms": round(lat_ms, 2)
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if idx % 50 == 0 or idx == len(needed):
                        out_f.flush()
                        print(f"  [{idx:3d}/{len(needed):3d}] {pid}: raw_disp=({best_opt}) -> mapped=({mapped_pred_opt}) [gold={gold}] correct={is_corr} ({lat_ms:.0f}ms)", flush=True)

    finally:
        if we_started_server and proc is not None:
            print(f"Stopping server process on port {port}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            print("Server terminated cleanly.")

def run_analysis_all_shifts():
    """Analyzes all available shift files and computes flip rates vs unpermuted f16."""
    print("\n" + "=" * 110)
    print("ANALYSIS: PERMUTATION FLIP RATES AGAINST UNPERMUTED F16 BASELINE")
    print("=" * 110)

    # Load dataset metadata for passage clustering
    items = [json.loads(line) for line in open(PARALLEL_FILE, encoding="utf-8") if line.strip()]
    pair_ids = sorted([it["pair_id"] for it in items])
    item_meta = {it["pair_id"]: {"link": it.get("link", it["pair_id"])} for it in items}

    # Load original f16 predictions
    orig_f16 = {"gemma": {}, "llama": {}}
    for m, p in ORIGINAL_F16_FILES.items():
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("tier") == "f16":
                            pid = r.get("id") or r.get("pair_id")
                            opt = str(r.get("pred_option") or r.get("predicted_option")).strip()
                            orig_f16[m][(r["language"], pid)] = opt

    # Setup 2000 cluster bootstrap resamples
    unique_clusters = np.array(sorted(list(set(item_meta[p]["link"] for p in pair_ids))))
    n_clusters = len(unique_clusters)
    cluster_to_indices = {c: [] for c in unique_clusters}
    for idx, pid in enumerate(pair_ids):
        cluster_to_indices[item_meta[pid]["link"]].append(idx)

    rng = np.random.RandomState(42)
    boot_indices_list = []
    for _ in range(2000):
        sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sample_idx = []
        for c in sampled_c:
            sample_idx.extend(cluster_to_indices[c])
        boot_indices_list.append(np.array(sample_idx))

    # Search for all permutation result files
    shift_files = list(RESULTS_DIR.glob("noise_floor_*_shift*.jsonl"))
    if not shift_files:
        print("No noise_floor_*_shift*.jsonl files found in results/ to analyze!")
        return

    csv_rows = []
    for sf in sorted(shift_files):
        # Filename format: noise_floor_{model}_f16_shift{s}.jsonl
        parts = sf.stem.split("_")
        model_key = parts[2]
        shift_num = parts[4].replace("shift", "")

        preds_shift = {}
        with open(sf, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    preds_shift[(r["language"], r["pair_id"])] = {
                        "mapped_opt": str(r["mapped_pred_option"]).strip(),
                        "is_correct": bool(r["is_correct"])
                    }

        orig_dict = orig_f16[model_key]

        for lang in ["bangla", "english"]:
            eval_pids = [pid for pid in pair_ids if (lang, pid) in preds_shift and (lang, pid) in orig_dict]
            if not eval_pids:
                continue

            flips = [1.0 if orig_dict[(lang, pid)] != preds_shift[(lang, pid)]["mapped_opt"] else 0.0 for pid in eval_pids]
            accs = [1.0 if preds_shift[(lang, pid)]["is_correct"] else 0.0 for pid in eval_pids]

            flips_arr = np.array(flips)
            accs_arr = np.array(accs)
            n_tot = len(flips_arr)
            flip_count = int(np.sum(flips_arr))
            flip_rate = float(np.mean(flips_arr))
            acc_rate = float(np.mean(accs_arr))

            # Cluster bootstrap over eval_pids clusters
            eval_clusters = np.array([item_meta[pid]["link"] for pid in eval_pids])
            u_clusters = np.unique(eval_clusters)
            c_to_idx = {c: np.where(eval_clusters == c)[0] for c in u_clusters}
            
            rng_boot = np.random.RandomState(42)
            boot_flips = []
            for _ in range(1000):
                sampled_c = rng_boot.choice(u_clusters, size=len(u_clusters), replace=True)
                sample_idx = np.concatenate([c_to_idx[c] for c in sampled_c])
                boot_flips.append(np.mean(flips_arr[sample_idx]))

            ci_lo = float(np.percentile(boot_flips, 2.5))
            ci_hi = float(np.percentile(boot_flips, 97.5))

            csv_rows.append({
                "model": "Gemma-3-1B-IT" if model_key == "gemma" else "Llama-3.2-1B-Instruct",
                "shift": f"Shift {shift_num}",
                "language": lang,
                "total_items": n_tot,
                "flip_count": flip_count,
                "flip_rate_pct": round(flip_rate * 100, 2),
                "flip_ci_95": f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]",
                "accuracy_pct": round(acc_rate * 100, 2),
            })

    df = pd.DataFrame(csv_rows)
    out_csv = AUDIT_DIR / "permutation_flip_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved Permutation Analysis CSV to: {out_csv}\n")

    print(f"{'Model':<22} | {'Shift':<10} | {'Lang':<7} | {'Flips (Count)':<16} | {'Flip Rate (95% CI)':<22} | {'Accuracy':<10}")
    print("-" * 100)
    for r in csv_rows:
        f_str = f"{r['flip_count']}/{r['total_items']}"
        fr_str = f"{r['flip_rate_pct']:5.2f}% {r['flip_ci_95']}"
        acc_str = f"{r['accuracy_pct']:5.2f}%"
        print(f"{r['model']:<22} | {r['shift']:<10} | {r['language']:<7} | {f_str:<16} | {fr_str:<22} | {acc_str:<10}")

def main():
    args = parse_args()
    
    if args.dry_run:
        print("Dry run complete.")
        return

    if args.mode == "analyze":
        run_analysis_all_shifts()
        return

    # Determine shifts to run
    active_shifts = [args.shift] if args.shifts is None else args.shifts
    models = ["gemma", "llama"] if args.model == "both" else [args.model]

    for m in models:
        for s in active_shifts:
            out_file = args.out if (len(active_shifts) == 1 and args.out) else (RESULTS_DIR / f"noise_floor_{m}_f16_shift{s}.jsonl")
            run_option_permutation_single_shift(m, s, port=args.port, threads=args.threads, out_override=out_file, limit=args.limit)

    # Run analysis
    run_analysis_all_shifts()

if __name__ == "__main__":
    main()
