"""Backward-compatible entry — prefer ``python scripts/run/compute_weights.py``."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "run" / "compute_weights.py"), run_name="__main__")
