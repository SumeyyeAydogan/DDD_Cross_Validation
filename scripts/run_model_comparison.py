"""Backward-compatible entry — prefer ``python scripts/run/model_comparison.py``."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "run" / "model_comparison.py"), run_name="__main__")
