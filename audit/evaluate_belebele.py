import os
import sys

# Ensure UTF-8 everywhere on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path("d:/Tokenizer/quant-hallucination-study")
DATA_V2 = ROOT / "data_v2"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_BN_FILE = DATA_V2 / "belebele_eval_bn_300.jsonl"
DEFAULT_EN_FILE = DATA_V2 / "belebele_eval_en_300.jsonl"
DEFAULT_OUTPUT_FILE = RESULTS_DIR / "belebele_predictions.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate quantized models on Belebele parallel benchmark")
    p.add_argument("--model-id", default="google/gemma-3-1b-it", help="Model HF ID or path")
    p.add_argument("--precisions", nargs="+", choices=("16bit", "8bit", "4bit"), default=["16bit", "8bit", "4bit"])
    p.add_argument("--languages", nargs="+", choices=("bangla", "english"), default=["bangla", "english"])
    p.add_argument("--limit", type=int, default=None, help="Limit number of items per language (for testing)")
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def load_eval_data(lang: str, limit: int | None = None) -> list[dict]:
    path = DEFAULT_BN_FILE if lang == "bangla" else DEFAULT_EN_FILE
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(model_id: str, precision: str):
    print(f"\n[{precision.upper()}] Loading model '{model_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    target_device = "cuda:0" if device == "cuda" else None

    try:
        if precision == "16bit":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
            )
            if device == "cuda":
                model = model.to("cuda")
        elif precision == "8bit":
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map=target_device,
                trust_remote_code=True,
            )
        elif precision == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map=target_device,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown precision: {precision}")
    except Exception as exc:
        import traceback
        print(f"Error loading model for precision {precision}:\n{traceback.format_exc()}", flush=True)
        raise exc

    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, instruction: str, passage: str, lang: str) -> str:
    if lang == "bangla":
        system_text = (
            "অনুগ্রহ করে নিচের অনুচ্ছেদটি মনোযোগ দিয়ে পড়ুন এবং শুধুমাত্র এই অনুচ্ছেদের তথ্যের ভিত্তিতে "
            "প্রশ্নের সঠিক বিকল্পটি বেছে নিন। বিকল্পের নম্বরটি পরিষ্কারভাবে উল্লেখ করুন।\n\n"
            f"অনুচ্ছেদ:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
            f"{instruction.strip()}\n\n"
            "সঠিক উত্তর:"
        )
    else:
        system_text = (
            "Read the following passage carefully and answer the question based ONLY on facts directly stated "
            "in the passage. Clearly indicate the correct option number.\n\n"
            f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
            f"{instruction.strip()}\n\n"
            "Correct Answer:"
        )

    messages = [{"role": "user", "content": system_text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {system_text}\nAssistant: "


def extract_chosen_option(response_text: str) -> str | None:
    # Match patterns like "(1)", "1", "Option 1", "বিকল্প ১", "(১)"
    m = re.search(r"\(([1-4১-৪])\)", response_text)
    if m:
        digit = m.group(1)
        bn_to_en = {"১": "1", "২": "2", "৩": "3", "৪": "4"}
        return bn_to_en.get(digit, digit)
    m = re.search(r"\b([1-4])\b", response_text[:30])
    if m:
        return m.group(1)
    m = re.search(r"([১-৪])", response_text[:30])
    if m:
        bn_to_en = {"১": "1", "২": "2", "৩": "3", "৪": "4"}
        return bn_to_en.get(m.group(1))
    return None


def main():
    args = parse_args()
    print("=" * 65)
    print("Belebele Quantization Evaluation Runner")
    print(f"Model:       {args.model_id}")
    print(f"Precisions:  {args.precisions}")
    print(f"Languages:   {args.languages}")
    print(f"Limit:       {args.limit if args.limit else 'All (300)'}")
    print(f"Output File: {args.output_file}")
    print("=" * 65)

    all_results = []
    
    for precision in args.precisions:
        tokenizer, model = load_model(args.model_id, precision)
        
        for lang in args.languages:
            items = load_eval_data(lang, limit=args.limit)
            print(f"\n[{precision.upper()} | {lang.upper()}] Evaluating {len(items)} items ...")
            
            correct_count = 0
            
            for idx, item in enumerate(items, start=1):
                prompt = format_prompt(tokenizer, item["instruction"], item["passage"], lang)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                latency = time.perf_counter() - t0
                
                gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                
                chosen_opt = extract_chosen_option(pred_text)
                gold_opt = str(item.get("correct_option", ""))
                is_correct = (chosen_opt == gold_opt)
                if is_correct:
                    correct_count += 1
                
                record = {
                    "item_id": item["id"],
                    "model_id": args.model_id,
                    "precision": precision,
                    "lang": lang,
                    "gold_option": gold_opt,
                    "chosen_option": chosen_opt,
                    "is_correct": is_correct,
                    "latency_sec": round(latency, 3),
                    "pred_text": pred_text,
                }
                all_results.append(record)
                
                # Append to output file
                with args.output_file.open("a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                print(f"  [{idx}/{len(items)}] ID: {item['id']} | Gold: ({gold_opt}) | Pred: ({chosen_opt}) | Correct: {is_correct} | {latency:.2f}s", flush=True)
            
            acc = (correct_count / len(items)) * 100 if items else 0.0
            print(f"==> [{precision.upper()} | {lang.upper()}] Accuracy: {acc:.2f}% ({correct_count}/{len(items)})", flush=True)
        
        # Cleanup model from VRAM
        del model
        torch.cuda.empty_cache()

    print("\nEvaluation run completed! Results saved to:", args.output_file)


if __name__ == "__main__":
    main()
