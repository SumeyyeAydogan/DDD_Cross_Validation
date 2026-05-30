"""Backward-compatible entry — prefer ``python scripts/run/cv_train.py``."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "run" / "cv_train.py"), run_name="__main__")
