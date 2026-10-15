import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Testing HuggingFace access for Belebele and TyDiQA...")

try:
    from datasets import load_dataset
    print("Loading facebook/belebele metadata or sample...")
    # Belebele is split by language subsets
    ds_bn = load_dataset("facebook/belebele", split="test", subset="ben_Beng")
    ds_en = load_dataset("facebook/belebele", split="test", subset="eng_Latn")
    print(f"Belebele loaded successfully! BN rows: {len(ds_bn)}, EN rows: {len(ds_en)}")
    print("Sample BN keys:", list(ds_bn[0].keys()))
    print("Sample BN question:", ds_bn[0]['question'][:60])
    print("Sample EN question:", ds_en[0]['question'][:60])
except Exception as e:
    print("Belebele check failed/skipped:", e)

try:
    print("\nLoading tydiqa goldp subset...")
    ds_tydi = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    bn_items = [x for x in ds_tydi if x['id'].startswith('bengali')]
    en_items = [x for x in ds_tydi if x['id'].startswith('english')]
    print(f"TyDi QA secondary_task loaded! Total validation rows: {len(ds_tydi)}, BN: {len(bn_items)}, EN: {len(en_items)}")
except Exception as e:
    print("TyDiQA check failed/skipped:", e)
