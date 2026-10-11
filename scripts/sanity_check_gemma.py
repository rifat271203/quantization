"""
Smoke test: load google/gemma-3-1b-it and run three short generations.

Confirms the instruct model fits on a single GPU (bf16, device_map=auto)
before committing to a fine-tuning pipeline. max_new_tokens is kept at 100.
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"
MAX_NEW_TOKENS = 100

PROMPTS = (
    ("Bangla", "বাংলাদেশের রাজধানীর নাম কী?"),
    ("English", "What is the capital of Bangladesh?"),
    ("Bangla STEM", "সালোকসংশ্লেষণ প্রক্রিয়া সংক্ষেপে ব্যাখ্যা কর।"),
)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.3f} GiB"


def cuda_allocated(device: int = 0) -> int:
    return int(torch.cuda.memory_allocated(device))


def cuda_peak(device: int = 0) -> int:
    return int(torch.cuda.max_memory_allocated(device))


def format_user_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_model():
    print(f"Loading {MODEL_ID} (bfloat16, device_map=auto) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_once(tokenizer, model, prompt: str) -> tuple[str, float, int]:
    """Return (decoded continuation, elapsed seconds, peak allocated bytes)."""
    device = model.device
    text = format_user_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = cuda_peak()

    prompt_len = inputs["input_ids"].shape[-1]
    continuation = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    return continuation.strip(), elapsed, peak


def main() -> int:
    _configure_stdout()

    if not torch.cuda.is_available():
        print("CUDA is not available. This smoke test requires a single GPU.")
        return 1

    n_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU 0: {gpu_name} ({gb(total_vram)} total)")
    if n_gpus > 1:
        print(f"Note: {n_gpus} GPUs visible; expecting a single-GPU fit via device_map=auto.")
    print()

    try:
        tokenizer, model = load_model()
        torch.cuda.synchronize()
        idle_vram = cuda_allocated()
        print(f"Loaded without OOM. Idle VRAM: {gb(idle_vram)}")
        print(f"Device map: {getattr(model, 'hf_device_map', model.device)}")
        print()
    except torch.cuda.OutOfMemoryError as exc:
        print("FAIL: OOM while loading the model on this GPU.")
        print(exc)
        return 1

    peak_generation = idle_vram
    oom_during_generate = False

    for i, (label, prompt) in enumerate(PROMPTS, start=1):
        print("=" * 72)
        print(f"[{i}/{len(PROMPTS)}] {label}")
        print(f"Prompt: {prompt}")
        print("-" * 72)
        try:
            output, elapsed, peak = generate_once(tokenizer, model, prompt)
        except torch.cuda.OutOfMemoryError as exc:
            oom_during_generate = True
            print("FAIL: OOM during generation.")
            print(exc)
            continue

        peak_generation = max(peak_generation, peak)
        print(f"Output: {output}")
        print(f"Generation time: {elapsed:.2f}s")
        print(f"Peak VRAM (max_memory_allocated): {gb(peak)}")
        print()

    print("=" * 72)
    print("VRAM summary")
    print("=" * 72)
    print(f"Idle (after load):          {gb(idle_vram)}")
    print(f"Peak during generation:     {gb(peak_generation)}")
    print(f"Delta (generation - idle):  {gb(max(peak_generation - idle_vram, 0))}")
    print(f"GPU capacity:               {gb(total_vram)}")

    if oom_during_generate:
        print("\nFAIL: model loaded, but generation hit OOM.")
        return 1

    print("\nPASS: google/gemma-3-1b-it loaded and generated without OOM on a single GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
