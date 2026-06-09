"""Backward-compatible entry — prefer ``python scripts/run/overlap_comparison.py``."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "run" / "overlap_comparison.py"), run_name="__main__")
