"""Shared helpers for the Streamlit experiment viewer."""
from __future__ import annotations

import csv
import inspect
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

_IMAGE_SUPPORTS_CONTAINER_WIDTH: Optional[bool] = None
_DATAFRAME_SUPPORTS_CONTAINER_WIDTH: Optional[bool] = None


def _image_supports_use_container_width() -> bool:
    global _IMAGE_SUPPORTS_CONTAINER_WIDTH
    if _IMAGE_SUPPORTS_CONTAINER_WIDTH is None:
        _IMAGE_SUPPORTS_CONTAINER_WIDTH = "use_container_width" in inspect.signature(
            st.image
        ).parameters
    return _IMAGE_SUPPORTS_CONTAINER_WIDTH


def _dataframe_supports_use_container_width() -> bool:
    global _DATAFRAME_SUPPORTS_CONTAINER_WIDTH
    if _DATAFRAME_SUPPORTS_CONTAINER_WIDTH is None:
        _DATAFRAME_SUPPORTS_CONTAINER_WIDTH = "use_container_width" in inspect.signature(
            st.dataframe
        ).parameters
    return _DATAFRAME_SUPPORTS_CONTAINER_WIDTH


def st_image(
    image: Any,
    *,
    caption: Optional[str] = None,
    width: Optional[int] = None,
    use_container_width: bool = False,
) -> None:
    """``st.image`` compatible with Streamlit < 1.14 (no ``use_container_width``)."""
    base: Dict[str, Any] = {}
    if caption is not None:
        base["caption"] = caption
    if width is not None:
        base["width"] = width

    attempts: List[Dict[str, Any]] = []
    if use_container_width and _image_supports_use_container_width():
        attempts.append({**base, "use_container_width": True})
    if use_container_width:
        attempts.append({**base, "use_column_width": True})
    attempts.append(dict(base))

    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            st.image(image, **kwargs)
            return
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err


def st_dataframe(data: Any, *, use_container_width: bool = True, **kwargs: Any) -> None:
    """``st.dataframe`` with optional ``use_container_width`` when supported."""
    if use_container_width and _dataframe_supports_use_container_width():
        st.dataframe(data, use_container_width=True, **kwargs)
    else:
        st.dataframe(data, **kwargs)


def st_rerun() -> None:
    """Refresh the app (Streamlit 1.27+ ``rerun`` or legacy ``experimental_rerun``)."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


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


def plot_overlay_metric_fig(
    series: Dict[str, List[float]],
    metric_key: str,
    *,
    title: str = "",
    ylabel: str = "",
) -> Optional[plt.Figure]:
    """Overlay one metric (e.g. val_accuracy) for multiple runs on the same axes."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    any_line = False
    for label, hist in series.items():
        if metric_key not in hist or not hist[metric_key]:
            continue
        ax.plot(hist[metric_key], label=label, linewidth=1.8)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title or metric_key)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weight_histogram_fig(values: List[float], *, title: str = "", bins: int = 50) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title(title or "Weight distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scatter_fig(
    x: List[float],
    y: List[float],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    alpha: float = 0.35,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x, y, s=12, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def short_image_label(rel_path: str) -> str:
    """Compact label: ``ClassName / filename.png`` (no full disk path)."""
    rel = str(rel_path).replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]} / {parts[-1]}"
    return parts[-1] if parts else rel


def load_image_uint8(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(img_size)
    return np.asarray(img).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    m = np.clip(mask, 0.0, 1.0)
    h, w = m.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 1] = m
    return (rgb * 255.0).astype(np.uint8)


def overlay_mask_on_image(image_uint8: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = image_uint8.astype(np.float32) / 255.0
    m = np.clip(mask[..., None], 0.0, 1.0)
    tint = np.zeros_like(img)
    tint[..., 1] = m[..., 0]
    out = (1.0 - alpha) * img + alpha * tint
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def show_figure_in_streamlit(fig: plt.Figure, width: int = 700) -> None:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st_image(buf, width=width)


def plot_pie_counts_fig(
    counts: Dict[str, int],
    *,
    title: str = "",
) -> Optional[plt.Figure]:
    labels = [k for k, v in counts.items() if v > 0]
    sizes = [counts[k] for k in labels]
    if not sizes:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_class_weight_boxplot_fig(
    df,
    *,
    weight_col: str,
    class_col: str = "class_name",
    set_col: str = "weights_set",
    title: str = "",
) -> Optional[plt.Figure]:
    """Grouped boxplot: weight by class, coloured by weight set (needs pandas)."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    sets = sorted(df[set_col].unique())
    classes = list(df[class_col].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(sets), 1)
    for i, cname in enumerate(classes):
        for j, tag in enumerate(sets):
            sub = df[(df[class_col] == cname) & (df[set_col] == tag)][weight_col]
            if sub.empty:
                continue
            pos = i + (j - (len(sets) - 1) / 2) * width
            ax.boxplot(
                sub.values,
                positions=[pos],
                widths=width * 0.9,
                patch_artist=True,
                labels=[""],
            )
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel("Sample weight")
    ax.set_title(title or "Weight by class")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def build_weights_comparison_table(
    per_set_records: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    One row per image; one column per weight set + optional delta between first two sets.
    """
    if not per_set_records:
        return []

    tags = list(per_set_records.keys())
    by_rel: Dict[str, Dict[str, Any]] = {}

    for tag, recs in per_set_records.items():
        col = f"weight ({tag})"
        for r in recs:
            rel = r["rel_path"]
            if rel not in by_rel:
                by_rel[rel] = {
                    "sample": short_image_label(rel),
                    "rel_path": rel,
                    "class": r.get("class_name", ""),
                    "abs_path": r.get("abs_path", ""),
                }
            by_rel[rel][col] = float(r["weight"])
            by_rel[rel]["class"] = r.get("class_name", by_rel[rel].get("class", ""))

    rows = list(by_rel.values())
    if len(tags) >= 2:
        a, b = tags[0], tags[1]
        ca, cb = f"weight ({a})", f"weight ({b})"
        for row in rows:
            wa, wb = row.get(ca), row.get(cb)
            if wa is not None and wb is not None:
                row[f"Δ ({b}−{a})"] = float(wb) - float(wa)

    display_cols = ["sample", "class"] + [f"weight ({t})" for t in tags]
    if len(tags) >= 2:
        display_cols.append(f"Δ ({tags[1]}−{tags[0]})")
    for row in rows:
        for k in display_cols:
            row.setdefault(k, None)
    return [{k: row.get(k) for k in display_cols + ["rel_path", "abs_path"]} for row in rows]


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