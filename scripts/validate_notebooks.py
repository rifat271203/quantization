import json
import ast
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


notebook_names = [
    "gemma3-qlora-bilingual.ipynb",
    "gemma3-qlora-bn-only.ipynb",
    "gemma3-qlora-en-only.ipynb",
]

for name in notebook_names:
    print("=" * 60)
    print(f"Validating {name}...")
    p = Path(name)
    assert p.exists(), f"File {name} does not exist!"

    with p.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"  Total cells: {len(cells)}")

    # Verify device_map
    all_text = "".join("".join(c["source"]) for c in cells)
    assert 'device_map={"": 0}' in all_text, f"Missing device_map={{\"\": 0}} in {name}!"
    assert 'device_map="auto"' not in all_text, f"Found lingering device_map='auto' in {name}!"

    # Verify checkpoint resume
    assert 'resume_from_checkpoint=last_checkpoint' in all_text, f"Missing resume_from_checkpoint in {name}!"
    assert 'save_strategy="steps"' in all_text, f"Missing save_strategy steps in {name}!"
    assert 'save_steps=SAVE_STEPS' in all_text or 'save_steps=200' in all_text, f"Missing save_steps in {name}!"
    assert 'save_total_limit=3' in all_text or 'save_total_limit=SAVE_TOTAL_LIMIT' in all_text, f"Missing save_total_limit in {name}!"

    # Verify periodic warning
    assert "TIME BUDGET MONITOR" in all_text, f"Missing TIME BUDGET MONITOR in {name}!"
    assert "warning_interval_steps=500" in all_text or "warning_interval_steps=TIME_WARNING_INTERVAL" in all_text, f"Missing warning interval in {name}!"

    # Verify zip output
    assert "make_archive" in all_text, f"Missing zip archive creation in {name}!"

    # Validate Python syntax for code cells
    for idx, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Remove jupyter bash commands like !pip, !unzip
            py_lines = [line for line in src.split("\n") if not line.strip().startswith("!")]
            py_src = "\n".join(py_lines)
            try:
                ast.parse(py_src)
            except SyntaxError as e:
                print(f"  [ERROR] SyntaxError in cell {idx}: {e}")
                raise

    print("  ✓ All structural, semantic, and syntax checks passed!")
print("=" * 60)
print("ALL THREE NOTEBOOKS VERIFIED SUCCESSFULLY!")
