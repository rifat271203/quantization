import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    print("Importing evaluate_belebele...")
    import audit.evaluate_belebele as eb
    print("Calling eb.main()...")
    eb.main()
except Exception as e:
    print("CAUGHT EXCEPTION:")
    traceback.print_exc(file=sys.stdout)
except SystemExit as e:
    print(f"SYSTEM EXIT: {e}")
