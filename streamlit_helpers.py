"""Shared helpers for the Streamlit experiment viewer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def discover_run_names(runs_root: Path) -> List[str]:
    if not runs_root.is_dir():
        return []
    names: List[str] = []
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if any(child.is_dir() and child.name.startswith("fold_") for child in p.iterdir()):
            names.append(p.name)
    return names


def list_folds_for_run(project_root: Path, run_name: str) -> List[int]:
    run_path = project_root / "runs" / run_name
    if not run_path.is_dir():
        return []
    folds: List[int] = []
    for p in run_path.iterdir():
        if p.is_dir() and p.name.startswith("fold_"):
            try:
                folds.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(folds)


def fold_run_dir(project_root: Path, run_name: str, fold_id: int) -> Path:
    return project_root / "runs" / run_name / f"fold_{fold_id}"


def model_h5_path(project_root: Path, run_name: str, fold_id: int) -> Path:
    return fold_run_dir(project_root, run_name, fold_id) / "models" / f"fold_{fold_id}.h5"


def parse_run_registry(text: str) -> List[Tuple[str, str]]:
    """Lines: label=run_folder_name"""
    rows: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        label, run = line.split("=", 1)
        label, run = label.strip(), run.strip()
        if label and run:
            rows.append((label, run))
    return rows


def merge_registry(
    parsed: List[Tuple[str, str]],
    discovered: List[str],
) -> List[Tuple[str, str]]:
    """Parsed aliases first; then discovered runs not already listed."""
    out = list(parsed)
    seen = {run for _, run in out}
    for run in discovered:
        if run not in seen:
            out.append((run, run))
            seen.add(run)
    return out


def load_metrics_csv(csv_path: Path) -> Dict[str, List[float]]:
    if not csv_path.is_file():
        return {}
    hist: Dict[str, List[float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key in ("epoch", "timestamp") or value in (None, ""):
                    continue
                try:
                    hist.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return hist


def plot_training_history_fig(hist: Dict[str, List[float]], title: str = "") -> Optional[plt.Figure]:
    if "accuracy" not in hist or "loss" not in hist:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(hist["accuracy"], label="Train")
    if "val_accuracy" in hist:
        axes[0].plot(hist["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["loss"], label="Train")
    if "val_loss" in hist:
        axes[1].plot(hist["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_extra_metrics_fig(hist: Dict[str, List[float]], title: str = "") -> Optional[plt.Figure]:
    keys = [k for k in ("precision", "recall", "auc") if k in hist]
    if not keys:
        return None
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, keys):
        ax.plot(hist[metric], label=f"Train {metric}")
        val_key = f"val_{metric}"
        if val_key in hist:
            ax.plot(hist[val_key], label=f"Val {metric}")
        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def saved_plot_path(fold_dir: Path, plot_kind: str) -> Optional[Path]:
    mapping = {
        "Training history (accuracy & loss)": "training_history.png",
        "Training metrics (precision, recall, AUC)": "training_metrics.png",
    }
    name = mapping.get(plot_kind)
    if not name:
        return None
    p = fold_dir / "plots" / name
    return p if p.is_file() else None
