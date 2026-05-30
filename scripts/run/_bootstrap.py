"""Shared ``sys.path`` setup for CLI entry points under ``scripts/run/``."""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> tuple[Path, Path]:
    """
    Return ``(project_root, scripts_dir)`` and ensure both are importable.

    - ``project_root`` → ``src.*``, ``scripts.*`` package imports
    - ``scripts_dir`` → flat imports like ``overlap_accuracy_comparison``
    """
    scripts_dir = Path(__file__).resolve().parent.parent
    project_root = scripts_dir.parent
    for p in (str(project_root), str(scripts_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return project_root, scripts_dir
