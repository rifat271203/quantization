import sys
import traceback

print("Starting test_run_direct.py ...", flush=True)

try:
    import torch
    print("PyTorch version:", torch.__version__, "CUDA:", torch.cuda.is_available(), flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Transformers imported. Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    print("Tokenizer loaded. Loading model in FP16 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully on:", model.device, flush=True)
    print("VRAM used (GB):", torch.cuda.memory_allocated() / (1024**3), flush=True)
except Exception as e:
    print("Exception caught:")
    traceback.print_exc()
