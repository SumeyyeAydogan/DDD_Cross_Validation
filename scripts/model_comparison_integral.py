import os, sys, json, csv, gc
import math
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add root path (project structure assumed same as before)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM
from src.ds_with_paths_pipeline import get_dataset_with_paths
from src.fold_functions import load_fold_manifest
from src.cv_dataloader import make_tf_dataset_from_paths
from src.mask_helpers import create_landmark_mask, image_to_float01_rgb, image_to_uint8_rgb, ROI_IDX
from src.focus_metrics import (
    FOCUS_METRIC_LABELS,
    compute_focus_score,
    histogram_right_tail_area,
)

# (plot label, run directory name under project_root/runs/) — cv_main layout:
#   runs/<name>/fold_k/models/fold_k.h5
_FOLD_ID_FOR_PATHS = 1
_MODEL_RUNS: List[Tuple[str, str]] = [
    ("original", "baseline"),
    ("reward", "reward"),
    ("log-reward", "log"),
    ("exp-reward", "exp"),
]


def model_configs_for_fold(fold_id: int) -> List[Dict[str, Any]]:
    """
    One fold's model list: ``runs/<run_name>/fold_<id>/models/fold_<id>.h5``.

    ``fold_id`` is **1-based** (same as ``fold_{fold_id}.json`` / ``cv_main``).
    """
    if fold_id < 1:
        raise ValueError(f"fold_id must be >= 1, got {fold_id}")
    rows: List[Dict[str, Any]] = []
    for label, run_name in _MODEL_RUNS:
        p = project_root / "runs" / run_name / f"fold_{fold_id}" / "models" / f"fold_{fold_id}.h5"
        rows.append({"label": label, "model_path": str(p)})
    return rows


def _default_model_configs() -> List[Dict[str, Any]]:
    return model_configs_for_fold(_FOLD_ID_FOR_PATHS)


# ================== CONFIG (ddd_cv standalone) ======================
CONFIG: Dict[str, Any] = {
    "data_dir": str(project_root / "dataset"),
    "img_size": (224, 224),
    "dataset_name": "dataset",

    "class_names": ["NotDrowsy", "Drowsy"],

    "background_mask_value": 0.0,
    "landmark_box_half_size": 12,

    "threshold_source": "baseline_median",

    "hist_bins": 50,
    "plot_fixed_x_range": True,
    "plot_x_min": 0.0,
    "plot_x_max": 1.0,

    #Gradcam Source

    "gradcam_class_source": "model_prediction",

    # focus_ratio | density_gap | density_gap_shifted | inside_density
    "focus_metric": "focus_ratio",
    "roi_threshold": 0.5,
    "eps": 1e-8,
}

MODEL_CONFIGS = _default_model_configs()

# ================== CORE ======================
def _collect_focus_ratios_core(model, image_batches, total_items, img_size, cfg: Optional[Dict[str, Any]] = None):
    """
    Shared core for per-image focus scores regardless of dataset source.
    `image_batches` should yield tensors shaped like (1, H, W, C).
    """
    cfg = dict(CONFIG if cfg is None else {**CONFIG, **cfg})
    focus_metric = str(cfg.get("focus_metric", "focus_ratio"))
    roi_threshold = float(cfg.get("roi_threshold", 0.5))
    eps = float(cfg.get("eps", 1e-8))

    gradcam = CustomGradCAM(model)
    ratios = []
    face_ok = 0
    total = 0

    for idx, images in enumerate(image_batches):
        total += 1
        image = images[0].numpy()

        image_rgb_uint8 = image_to_uint8_rgb(image)
        image_float01 = image_to_float01_rgb(image)

        # prediction -> class_idx for gradcam
        prob = float(model.predict(image_float01[None, ...], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0

        heatmap = gradcam.compute_heatmap(image_float01, class_idx=pred)
        heatmap = tf.image.resize(
            heatmap[..., None],
            img_size,
            method="bilinear",
            antialias=True,
        ).numpy()[..., 0]
        heatmap = np.clip(heatmap, 0.0, None).astype(np.float32)

        mask = create_landmark_mask(
            image_rgb_uint8,
            img_size,
            background_mask_value=float(cfg.get("background_mask_value", 0.0)),
            landmark_box_half_size=int(cfg.get("landmark_box_half_size", 12)),
        )
        if mask is not None:
            face_ok += 1
            score, _ = compute_focus_score(
                heatmap,
                mask,
                metric=focus_metric,
                roi_threshold=roi_threshold,
                eps=eps,
            )
            ratios.append(score)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{total_items} | face_ok={face_ok}")

    ratios = np.array(ratios, dtype=np.float32)
    stats = {
        "N_total": int(total),
        "N_face": int(face_ok),
        "face_rate": float(face_ok / max(total, 1)),
    }
    return ratios, stats


def collect_pred_class_focus_ratios_for_model(
    model,
    data_dir,
    img_size,
    class_names,
    cfg: Optional[Dict[str, Any]] = None,
):
    """
    Returns:
      focus_ratios: np.array (only face_ok==1)
      stats: dict with N_total, N_face, face_rate
    """
    ds, file_paths = get_dataset_with_paths(data_dir, img_size, class_names)

    def _image_batches():
        for data_batch, _path_batch in ds:
            images, _labels = data_batch
            yield images

    return _collect_focus_ratios_core(
        model=model,
        image_batches=_image_batches(),
        total_items=len(file_paths),
        img_size=img_size,
        cfg=cfg,
    )


def collect_focus_ratios_from_fold_val(
    model,
    fold_id: int,
    img_size: Tuple[int, int],
    fold_datasets_dir: str,
    cfg: Optional[Dict[str, Any]] = None,
):
    """
    Val split from ``fold_datasets/fold_{fold_id}.json`` (same layout as ``cv_main`` / ``fold_functions``).

    ``fold_id`` is **1-based** (``fold_1.json`` …).
    """
    if fold_id < 1:
        raise ValueError(f"fold_id must be >= 1, got {fold_id}")
    fold_idx = fold_id - 1
    data = load_fold_manifest(fold_idx, fold_datasets_dir)
    val_files = data["val"]["files"]
    val_labels = np.asarray(data["val"]["labels"], dtype=np.float32)

    ds = make_tf_dataset_from_paths(
        val_files,
        val_labels,
        img_size,
        batch_size=1,
        augment=False,
        seed=None,
    )

    def _image_batches():
        for images, _labels in ds:
            yield images

    return _collect_focus_ratios_core(
        model=model,
        image_batches=_image_batches(),
        total_items=len(val_files),
        img_size=img_size,
        cfg=cfg,
    )


# ================== PLOT ======================
def _resolve_plot_range(results_dict: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Tuple[float, float]:
    if bool(cfg.get("plot_fixed_x_range", True)):
        return float(cfg.get("plot_x_min", 0.0)), float(cfg.get("plot_x_max", 1.0))
    nonempty = [v for v in results_dict.values() if v is not None and len(v) > 0]
    all_vals = np.concatenate(nonempty) if len(nonempty) > 0 else np.array([0.0, 1.0])
    return float(np.min(all_vals)), float(np.max(all_vals))


def plot_focus_ratio_by_model(
    results_dict,
    dataset_name,
    output_path,
    cfg,
    threshold_T=None,
    metadata_text="",
    summary_by_label: Optional[Dict[str, Dict[str, Any]]] = None,
    model_order: Optional[List[str]] = None,
):
    labels = list(model_order) if model_order else list(results_dict.keys())
    labels = [lbl for lbl in labels if lbl in results_dict]
    if not labels:
        return

    n = len(labels)
    n_cols = 2 if n > 1 else 1
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    fig.suptitle(f'Focus Ratio Distribution by Model - {dataset_name}', fontsize=16, fontweight='bold')
    axes_arr = np.array(axes).reshape(-1)

    # ---------- 1) GLOBAL X RANGE ----------
    x_min, x_max = _resolve_plot_range(results_dict, cfg)

    # ---------- 2) SAME BINS ----------
    n_bins = int(cfg.get("hist_bins", 50))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # ---------- 3) GLOBAL Y RANGE ----------
    global_ymax = 0.0
    for label in labels:
        vals = results_dict.get(label, np.array([]))
        if vals is None or len(vals) == 0:
            continue
        hist, _ = np.histogram(vals, bins=bin_edges, density=True)
        global_ymax = max(global_ymax, float(hist.max()))
    global_ymax *= 1.10 if global_ymax > 0 else 1.0

    base_palette = ["blue", "green", "orange", "red", "purple", "brown", "teal", "magenta", "gray"]
    colors = {label: base_palette[i % len(base_palette)] for i, label in enumerate(labels)}

    for idx, label in enumerate(labels):
        ax = axes_arr[idx]
        ratios = results_dict.get(label, np.array([]))
        if ratios is None:
            ratios = np.array([])

        if len(ratios) > 0:
            ratios = np.clip(ratios, x_min, x_max)
            ax.hist(
                ratios,
                bins=bin_edges,
                edgecolor='black',
                alpha=0.7,
                color=colors.get(label, "gray"),
                density=True
            )

            median_val = float(np.median(ratios))
            mean_val = float(np.mean(ratios))

            ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_val:.3f}')
            ax.axvline(mean_val, color='black', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.3f}')
            if threshold_T is not None:
                ax.axvline(float(threshold_T), color='purple', linestyle=':', linewidth=2,
                           label=f'T: {float(threshold_T):.3f}')
                # Shade right-tail area to visualize integral metric.
                ax.axvspan(float(threshold_T), x_max, alpha=0.08, color="purple")

            ax.set_title(f'{label} (Count: {len(ratios)})', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.3)

            detail_text = [f"Mean: {mean_val:.3f}", f"Std: {np.std(ratios):.3f}"]
            if summary_by_label and label in summary_by_label:
                summary_row = summary_by_label[label]
                val_acc = summary_row.get("val_accuracy")
                val_auc = summary_row.get("val_auc")
                val_acc_txt = f"{float(val_acc):.4f}" if val_acc is not None and np.isfinite(float(val_acc)) else "N/A"
                val_auc_txt = f"{float(val_auc):.4f}" if val_auc is not None and np.isfinite(float(val_auc)) else "N/A"
                detail_text.extend(
                    [
                        f"ΔP: {float(summary_row.get('delta_P_vs_baseline', np.nan)):+.3f}",
                        f"ΔArea: {float(summary_row.get('delta_Area_hist_vs_baseline', np.nan)):+.3f}",
                        f"val_acc: {val_acc_txt}",
                        f"val_auc: {val_auc_txt}",
                    ]
                )
            ax.text(
                0.02, 0.98, "\n".join(detail_text), transform=ax.transAxes,
                fontsize=8.5, verticalalignment='top', horizontalalignment='left',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.75)
            )
        else:
            ax.text(0.5, 0.5, f'No data for {label}',
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(label, fontsize=13, fontweight='bold')

        # ---------- 4) FORCE SAME AXES ----------
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, global_ymax)
        ax.set_xlabel('Focus Ratio', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        if metadata_text:
            ax.text(
                0.98, 0.02, metadata_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
            )

    for j in range(len(labels), len(axes_arr)):
        axes_arr[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Model Comparison] Histogram saved: {output_path}")


def plot_fold_boxplot(rows: List[Dict[str, Any]], output_path: str, model_order: Optional[List[str]] = None) -> None:
    grouped: Dict[str, List[float]] = {}
    for r in rows:
        grouped.setdefault(str(r["model_label"]), []).append(float(r["mean_focus"]))
    if model_order:
        labels = [k for k in model_order if k in grouped]
    else:
        labels = sorted(grouped.keys())
    if not labels:
        return
    data = [grouped[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanprops=dict(marker="^", markerfacecolor="black", markeredgecolor="black", markersize=7),
        medianprops=dict(color="#FFD54F", linewidth=2),
        boxprops=dict(linewidth=1.4),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    ax.set_title("Fold Mean Focus by Model")
    ax.set_ylabel("Mean Focus")
    ax.grid(True, alpha=0.3)
    legend_elements = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Triangle (^): fold mean-focus average"),
        Line2D([0], [0], color="#FFD54F", lw=2, label="Yellow line: median"),
        Line2D([0], [0], color="black", lw=1.4, label="Box: Q1-Q3 (middle 50%)"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Model Comparison] Fold boxplot saved: {output_path}")


def plot_aggregate_metrics(aggregates: List[Dict[str, Any]], output_path: str) -> None:
    if len(aggregates) == 0:
        return
    labels = [a["weight_type"] for a in aggregates]
    x = np.arange(len(labels))
    p_mean = [a["mean_delta_P_vs_baseline"] for a in aggregates]
    p_std = [a["std_delta_P_vs_baseline"] for a in aggregates]
    a_mean = [a["mean_delta_Area_hist_vs_baseline"] for a in aggregates]
    a_std = [a["std_delta_Area_hist_vs_baseline"] for a in aggregates]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, p_mean, yerr=p_std, capsize=4, alpha=0.8)
    axes[0].set_title("Mean ΔP vs baseline")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].axhline(0.0, color="#FFD54F", linestyle="--", linewidth=2, label="Yellow dashed line: no change vs baseline (0)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].bar(x, a_mean, yerr=a_std, capsize=4, alpha=0.8, color="orange")
    axes[1].set_title("Mean ΔArea_hist vs baseline")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].axhline(0.0, color="#FFD54F", linestyle="--", linewidth=2, label="Yellow dashed line: no change vs baseline (0)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")

    fig.text(
        0.5,
        0.01,
        "Bars = fold mean improvement per weight type | Error bars = std across folds",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Model Comparison] Aggregate plot saved: {output_path}")


def _build_recommendation_text(
    *,
    aggregates: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    if len(aggregates) == 0:
        return "No aggregate data available. Run multi-fold comparison first.\n"

    best_by_delta_p = max(aggregates, key=lambda x: float(x["mean_delta_P_vs_baseline"]))
    best_by_delta_area = max(aggregates, key=lambda x: float(x["mean_delta_Area_hist_vs_baseline"]))
    most_stable = min(aggregates, key=lambda x: float(x["std_delta_P_vs_baseline"]))

    lines: List[str] = []
    lines.append("Model Comparison Recommendation")
    lines.append("=" * 34)
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Output dir: {output_dir}")
    lines.append("")
    lines.append("Delta-P Summary (vs baseline)")
    lines.append("-" * 30)
    for a in sorted(aggregates, key=lambda x: x["weight_type"]):
        lines.append(
            f"- {a['weight_type']}: mean ΔP={float(a['mean_delta_P_vs_baseline']):+.4f}, "
            f"std={float(a['std_delta_P_vs_baseline']):.4f}, folds={a.get('fold_ids', [])}"
        )
    lines.append("")
    lines.append("Delta-Area Summary (vs baseline)")
    lines.append("-" * 34)
    for a in sorted(aggregates, key=lambda x: x["weight_type"]):
        lines.append(
            f"- {a['weight_type']}: mean ΔArea={float(a['mean_delta_Area_hist_vs_baseline']):+.4f}, "
            f"std={float(a['std_delta_Area_hist_vs_baseline']):.4f}"
        )
    lines.append("")
    lines.append("Quick Recommendation")
    lines.append("-" * 20)
    lines.append(
        f"- Best mean ΔP: {best_by_delta_p['weight_type']} "
        f"({float(best_by_delta_p['mean_delta_P_vs_baseline']):+.4f})"
    )
    lines.append(
        f"- Best mean ΔArea: {best_by_delta_area['weight_type']} "
        f"({float(best_by_delta_area['mean_delta_Area_hist_vs_baseline']):+.4f})"
    )
    lines.append(
        f"- Most stable (lowest ΔP std): {most_stable['weight_type']} "
        f"({float(most_stable['std_delta_P_vs_baseline']):.4f})"
    )
    lines.append("")
    lines.append("Plot Legend Notes")
    lines.append("-" * 18)
    lines.append("- aggregate_weight_types.png:")
    lines.append("  Bars = fold-mean improvement by weight type.")
    lines.append("  Error bars = standard deviation across folds.")
    lines.append("  Yellow dashed horizontal line = delta 0 (no change vs baseline).")
    lines.append("- fold_model_boxplot.png:")
    lines.append("  Triangle (^) marker = mean value.")
    lines.append("  Yellow line inside each box = median.")
    lines.append("  Box = Q1-Q3 (middle 50%); whiskers = spread outside box.")
    lines.append("")
    lines.append(f"Total rows analyzed: {len(all_rows)}")
    return "\n".join(lines) + "\n"


def _save_recommendation_text(
    *,
    output_dir: str,
    aggregates: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
) -> str:
    recommendation_path = os.path.join(output_dir, "recommendation.txt")
    os.makedirs(output_dir, exist_ok=True)
    text = _build_recommendation_text(aggregates=aggregates, all_rows=all_rows, output_dir=output_dir)
    with open(recommendation_path, "w", encoding="utf-8") as f:
        f.write(text)
    return recommendation_path


# ================== SUMMARY SAVE ======================
def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
def print_summary_table(rows, dataset_name):
    """
    rows: list of dicts with keys:
      model_label, N_face, N_total, threshold_T, P_focus_above_T, delta_P_vs_baseline,
      mean_focus, median_focus, std_focus
    """
    print("\n================= SUMMARY TABLE =================")
    print(f"Dataset: {dataset_name}")
    headers = ["Model", "N_face", "N_total", "T", "P(focus>T)", "ΔP", "Mean", "Median", "Std"]
    colw = [10, 7, 7, 7, 11, 7, 7, 7, 7]

    def fmt_row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, colw))

    print(fmt_row(headers))
    print(fmt_row(["-"*len(h) for h in headers]))

    for r in rows:
        vals = [
            r["model_label"],
            r["N_face"],
            r["N_total"],
            f'{r["threshold_T"]:.4f}',
            f'{r["P_focus_above_T"]:.4f}',
            f'{r["delta_P_vs_baseline"]:+.4f}',
            f'{r["mean_focus"]:.4f}',
            f'{r["median_focus"]:.4f}',
            f'{r["std_focus"]:.4f}',
        ]
        print(fmt_row(vals))

def print_summary_table_markdown(rows, dataset_name):
    """
    rows: list of summary dicts
    Prints a GitHub/Markdown-compatible table.
    """
    print("\n================= SUMMARY TABLE (MARKDOWN) =================")
    print(f"**Dataset:** `{dataset_name}`\n")

    headers = [
        "Model", "N_face", "N_total", "T",
        "P(focus > T)", "ΔP vs baseline",
        "Mean", "Median", "Std"
    ]

    # header
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        print(
            "| "
            f"{r['model_label']} | "
            f"{r['N_face']} | "
            f"{r['N_total']} | "
            f"{r['threshold_T']:.4f} | "
            f"{r['P_focus_above_T']:.4f} | "
            f"{r['delta_P_vs_baseline']:+.4f} | "
            f"{r['mean_focus']:.4f} | "
            f"{r['median_focus']:.4f} | "
            f"{r['std_focus']:.4f} |"
        )

def _compute_fold_summary_rows(
    *,
    fold_id: int,
    experiment_id: str,
    weight_type: str,
    dataset_name: str,
    data_dir: str,
    img_size: Tuple[int, int],
    ratios_by_model: Dict[str, np.ndarray],
    stats_by_model: Dict[str, Dict[str, Any]],
    model_paths: Dict[str, str],
    cfg: Dict[str, Any],
    baseline_label: str,
) -> List[Dict[str, Any]]:
    if baseline_label not in ratios_by_model or len(ratios_by_model[baseline_label]) == 0:
        raise RuntimeError(f"Fold {fold_id}: baseline '{baseline_label}' ratios missing/empty.")

    T = float(np.median(ratios_by_model[baseline_label]))
    baseline_ratios = ratios_by_model[baseline_label]
    P_baseline = float(np.mean(baseline_ratios > T))

    x_min, x_max = _resolve_plot_range(ratios_by_model, cfg)
    n_bins = int(cfg.get("hist_bins", 50))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    A_hist_baseline = histogram_right_tail_area(baseline_ratios, T, bin_edges)

    summary_rows: List[Dict[str, Any]] = []
    for label, ratios in ratios_by_model.items():
        if len(ratios) == 0:
            continue
        p_above = float(np.mean(ratios > T))
        delta_p = p_above - P_baseline
        area_hist = histogram_right_tail_area(ratios, T, bin_edges)
        delta_area_hist = area_hist - A_hist_baseline
        mean_v = float(np.mean(ratios))
        med_v = float(np.median(ratios))
        std_v = float(np.std(ratios))

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": experiment_id,
            "fold_id": int(fold_id),
            "weight_type": weight_type,
            "dataset": dataset_name,
            "data_dir": data_dir,
            "img_size": list(img_size),
            "model_label": label,
            "baseline_label": baseline_label,
            "model_path": model_paths.get(label, ""),
            "focus_metric": str(cfg.get("focus_metric", "focus_ratio")),
            "focus_metric_description": FOCUS_METRIC_LABELS.get(
                str(cfg.get("focus_metric", "focus_ratio")), ""
            ),
            "threshold_source": cfg.get("threshold_source", "baseline_median"),
            "threshold_T": T,
            "P_focus_above_T": p_above,
            "delta_P_vs_baseline": delta_p,
            "Area_hist_T_to_max": area_hist,
            "delta_Area_hist_vs_baseline": delta_area_hist,
            "hist_bins": int(cfg.get("hist_bins", 50)),
            "hist_x_min": x_min,
            "hist_x_max": x_max,
            "mean_focus": mean_v,
            "median_focus": med_v,
            "std_focus": std_v,
            "N_total": stats_by_model[label]["N_total"],
            "N_face": stats_by_model[label]["N_face"],
            "face_rate": stats_by_model[label]["face_rate"],
            "mask": {
                "roi": "eye_mouth",
                "landmark_box_half_size": cfg["landmark_box_half_size"],
                "background_mask_value": cfg["background_mask_value"],
                "roi_landmark_count": len(ROI_IDX),
            },
        }
        summary_rows.append(summary)
    return summary_rows


def _best_val_metrics_from_fold_training_csv(fold_run_dir: Path) -> Tuple[float, float]:
    """Max val_accuracy and val_auc over epochs from ``training_metrics.csv`` (ddd_cv ``cv_main`` layout)."""
    csv_path = fold_run_dir / "training_metrics.csv"
    if not csv_path.is_file():
        return float("nan"), float("nan")
    best_acc = float("-inf")
    best_auc = float("-inf")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                va = float(row.get("val_accuracy", "nan"))
                if np.isfinite(va):
                    best_acc = max(best_acc, va)
            except (TypeError, ValueError):
                pass
            try:
                vu = float(row.get("val_auc", "nan"))
                if np.isfinite(vu):
                    best_auc = max(best_auc, vu)
            except (TypeError, ValueError):
                pass
    acc_out = float(best_acc) if np.isfinite(best_acc) else float("nan")
    auc_out = float(best_auc) if np.isfinite(best_auc) else float("nan")
    return acc_out, auc_out


def _load_val_metrics_for_fold(model_paths: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Best val_accuracy / val_auc from ``runs/<run>/fold_k/training_metrics.csv`` next to each model."""
    metrics_by_label: Dict[str, Dict[str, float]] = {}
    for label, mpath in model_paths.items():
        if not mpath or not os.path.isfile(mpath):
            continue
        fold_dir = Path(mpath).resolve().parent.parent
        acc, auc = _best_val_metrics_from_fold_training_csv(fold_dir)
        if np.isfinite(acc) or np.isfinite(auc):
            metrics_by_label[label] = {"val_accuracy": acc, "val_auc": auc}
    return metrics_by_label


def _aggregate_by_weight_type(rows: List[Dict[str, Any]], baseline_label: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = row.get("model_label")
        if label == baseline_label:
            continue
        key = str(row.get("weight_type", label))
        grouped.setdefault(key, []).append(row)

    aggregates: List[Dict[str, Any]] = []
    for weight_type, group in grouped.items():
        delta_p_values = np.array([float(r["delta_P_vs_baseline"]) for r in group], dtype=np.float32)
        delta_area_values = np.array([float(r["delta_Area_hist_vs_baseline"]) for r in group], dtype=np.float32)
        fold_ids = sorted(list({int(r["fold_id"]) for r in group}))

        aggregates.append(
            {
                "weight_type": weight_type,
                "n_rows": int(len(group)),
                "fold_ids": fold_ids,
                "mean_delta_P_vs_baseline": float(np.mean(delta_p_values)),
                "std_delta_P_vs_baseline": float(np.std(delta_p_values)),
                "mean_delta_Area_hist_vs_baseline": float(np.mean(delta_area_values)),
                "std_delta_Area_hist_vs_baseline": float(np.std(delta_area_values)),
            }
        )
    return sorted(aggregates, key=lambda x: x["weight_type"])


def _save_aggregates(
    aggregates: List[Dict[str, Any]],
    aggregate_json_path: str,
    aggregate_csv_path: str,
) -> None:
    os.makedirs(os.path.dirname(aggregate_json_path), exist_ok=True)
    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2, ensure_ascii=False)

    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "weight_type",
                "n_rows",
                "fold_ids",
                "mean_delta_P_vs_baseline",
                "std_delta_P_vs_baseline",
                "mean_delta_Area_hist_vs_baseline",
                "std_delta_Area_hist_vs_baseline",
            ],
        )
        writer.writeheader()
        for row in aggregates:
            flat_row = dict(row)
            flat_row["fold_ids"] = ",".join(str(x) for x in row.get("fold_ids", []))
            writer.writerow(flat_row)


def run_single_fold_comparison(
    *,
    fold_id: int,
    fold_datasets_dir: str,
    model_configs: List[Dict[str, str]],
    output_dir: str,
    experiment_id: str = "default",
    weight_type: Optional[str] = None,
    baseline_label: str = "original",
    config_override: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    cfg = dict(CONFIG)
    if config_override:
        cfg.update(config_override)

    img_size = tuple(cfg["img_size"])
    dataset_name = cfg.get("dataset_name", f"fold_{fold_id}_val")
    resolved_weight_type = weight_type if weight_type is not None else "mixed"
    data_dir = f"fold_datasets:fold_{fold_id}"

    ratios_by_model: Dict[str, np.ndarray] = {}
    stats_by_model: Dict[str, Dict[str, Any]] = {}
    model_paths: Dict[str, str] = {}

    print(f"\n================= FOLD {fold_id} =================")
    print(f"Val split from: {os.path.join(fold_datasets_dir, f'fold_{fold_id}.json')}")
    for m_cfg in model_configs:
        label = m_cfg["label"]
        model_path = m_cfg["model_path"]
        model_paths[label] = model_path
        print(f"\n[LOAD] fold={fold_id} label={label}: {model_path}")
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model path, skip: {model_path}")
            continue
        model = None
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"[RUN] Collecting focus ratios for fold={fold_id}, label={label} ...")
            ratios, stats = collect_focus_ratios_from_fold_val(
                model=model,
                fold_id=fold_id,
                img_size=img_size,
                fold_datasets_dir=fold_datasets_dir,
                cfg=cfg,
            )
            ratios_by_model[label] = ratios
            stats_by_model[label] = stats
            if len(ratios) > 0:
                print(
                    f"[DONE] {label}: N_face={stats['N_face']} / N_total={stats['N_total']} "
                    f"| mean={ratios.mean():.4f} | median={np.median(ratios):.4f}"
                )
            else:
                print(f"[DONE] {label}: No valid ratios.")
        finally:
            if model is not None:
                del model
            tf.keras.backend.clear_session()
            gc.collect()

    summary_rows = _compute_fold_summary_rows(
        fold_id=fold_id,
        experiment_id=experiment_id,
        weight_type=resolved_weight_type,
        dataset_name=dataset_name,
        data_dir=data_dir,
        img_size=img_size,
        ratios_by_model=ratios_by_model,
        stats_by_model=stats_by_model,
        model_paths=model_paths,
        cfg=cfg,
        baseline_label=baseline_label,
    )
    val_metrics_by_label = _load_val_metrics_for_fold(model_paths)
    if val_metrics_by_label:
        print(f"[VAL METRICS] fold={fold_id} loaded metrics:")
        for label_key in model_paths.keys():
            if label_key not in val_metrics_by_label:
                continue
            m = val_metrics_by_label[label_key]
            print(
                f"  - {label_key}: "
                f"val_acc={m.get('val_accuracy', float('nan'))} | "
                f"val_auc={m.get('val_auc', float('nan'))}"
            )
    else:
        print(f"[VAL METRICS] fold={fold_id} no val_acc/val_auc found (will show N/A on plots).")
    for row in summary_rows:
        label = str(row.get("model_label", ""))
        m = val_metrics_by_label.get(label, {})
        row["val_accuracy"] = m.get("val_accuracy")
        row["val_auc"] = m.get("val_auc")

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fold_hist_path = os.path.join(plots_dir, f"fold_{fold_id}_hist.png")
    threshold_T = summary_rows[0]["threshold_T"] if len(summary_rows) > 0 else None
    n_face = summary_rows[0]["N_face"] if len(summary_rows) > 0 else 0
    n_total = summary_rows[0]["N_total"] if len(summary_rows) > 0 else 0
    metadata_text = f"fold={fold_id}\nN_face={n_face}/{n_total}\nbins={cfg.get('hist_bins', 50)}"
    plot_focus_ratio_by_model(
        ratios_by_model,
        f"{dataset_name}_fold_{fold_id}",
        fold_hist_path,
        cfg=cfg,
        threshold_T=threshold_T,
        metadata_text=metadata_text,
        summary_by_label={str(r["model_label"]): r for r in summary_rows},
        model_order=[m["label"] for m in model_configs],
    )
    print_summary_table(summary_rows, f"{dataset_name}_fold_{fold_id}")
    print_summary_table_markdown(summary_rows, f"{dataset_name}_fold_{fold_id}")
    return summary_rows


def run_multi_fold_comparison(
    *,
    fold_ids: List[int],
    fold_datasets_dir: str,
    model_map_by_fold: Dict[int, List[Dict[str, str]]],
    output_dir: str = "artifacts/model_comparison",
    experiment_id: str = "model_comparison_folds",
    weight_type_map_by_label: Optional[Dict[str, str]] = None,
    baseline_label: str = "original",
    config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    fold_metrics_path = os.path.join(output_dir, "fold_metrics.jsonl")
    aggregate_json_path = os.path.join(output_dir, "aggregate_by_weight_type.json")
    aggregate_csv_path = os.path.join(output_dir, "aggregate_by_weight_type.csv")
    aggregate_plot_path = os.path.join(output_dir, "plots", "aggregate_weight_types.png")
    fold_boxplot_path = os.path.join(output_dir, "plots", "fold_model_boxplot.png")

    all_rows: List[Dict[str, Any]] = []
    for fold_id in fold_ids:
        model_configs = model_map_by_fold.get(fold_id, [])
        fold_json = os.path.join(fold_datasets_dir, f"fold_{fold_id}.json")
        if not os.path.isfile(fold_json):
            print(f"[WARN] fold={fold_id} missing {fold_json}, skip.")
            continue
        if len(model_configs) == 0:
            print(f"[WARN] fold={fold_id} model configs missing, skip.")
            continue

        summary_rows = run_single_fold_comparison(
            fold_id=fold_id,
            fold_datasets_dir=fold_datasets_dir,
            model_configs=model_configs,
            output_dir=output_dir,
            experiment_id=experiment_id,
            weight_type="mixed",
            baseline_label=baseline_label,
            config_override=config_override,
        )
        for row in summary_rows:
            row["weight_type"] = weight_type_map_by_label.get(row["model_label"], row["model_label"]) if weight_type_map_by_label else row["model_label"]
            append_jsonl(fold_metrics_path, row)
        all_rows.extend(summary_rows)

    aggregates = _aggregate_by_weight_type(all_rows, baseline_label=baseline_label)
    _save_aggregates(
        aggregates=aggregates,
        aggregate_json_path=aggregate_json_path,
        aggregate_csv_path=aggregate_csv_path,
    )

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plot_aggregate_metrics(aggregates, aggregate_plot_path)
    discovered_order = [m["label"] for fid in fold_ids for m in model_map_by_fold.get(fid, [])]
    model_order = list(dict.fromkeys(discovered_order))
    plot_fold_boxplot(all_rows, fold_boxplot_path, model_order=model_order)
    recommendation_path = _save_recommendation_text(
        output_dir=output_dir,
        aggregates=aggregates,
        all_rows=all_rows,
    )

    print(f"\n[SAVE] Fold metrics: {fold_metrics_path}")
    print(f"[SAVE] Aggregate JSON: {aggregate_json_path}")
    print(f"[SAVE] Aggregate CSV: {aggregate_csv_path}")
    print(f"[SAVE] Aggregate Plot: {aggregate_plot_path}")
    print(f"[SAVE] Fold Boxplot: {fold_boxplot_path}")
    print(f"[SAVE] Recommendation: {recommendation_path}")
    return {
        "fold_metrics_path": fold_metrics_path,
        "aggregate_json_path": aggregate_json_path,
        "aggregate_csv_path": aggregate_csv_path,
        "aggregate_plot_path": aggregate_plot_path,
        "fold_boxplot_path": fold_boxplot_path,
        "recommendation_path": recommendation_path,
        "rows_count": len(all_rows),
        "aggregate_count": len(aggregates),
    }


# ================== MAIN ======================
if __name__ == "__main__":
    # Backward-compatible single-run mode
    cfg = CONFIG
    data_dir = cfg["data_dir"]
    img_size = tuple(cfg["img_size"])
    dataset_name = cfg.get("dataset_name", Path(data_dir).name)
    output_dir = str(project_root / "artifacts" / "model_comparison_legacy")
    os.makedirs(output_dir, exist_ok=True)

    ratios_by_model = {}
    stats_by_model = {}
    model_paths = {}
    for m_cfg in MODEL_CONFIGS:
        label = m_cfg["label"]
        model_path = m_cfg["model_path"]
        model_paths[label] = model_path
        print(f"\n[LOAD] {label}: {model_path}")
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model path, skip: {model_path}")
            continue
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"[RUN] Collecting focus ratios for {label} ...")
        ratios, stats = collect_pred_class_focus_ratios_for_model(
            model, data_dir, img_size, cfg["class_names"], cfg=cfg
        )
        ratios_by_model[label] = ratios
        stats_by_model[label] = stats

    summary_rows = _compute_fold_summary_rows(
        fold_id=0,
        experiment_id="legacy_main",
        weight_type="mixed",
        dataset_name=dataset_name,
        data_dir=data_dir,
        img_size=img_size,
        ratios_by_model=ratios_by_model,
        stats_by_model=stats_by_model,
        model_paths=model_paths,
        cfg=cfg,
    )
    summary_path = os.path.join(output_dir, f"focus_summary_{dataset_name}.jsonl")
    for row in summary_rows:
        append_jsonl(summary_path, row)
    hist_path = os.path.join(output_dir, f"model_focus_comparison_{dataset_name}.png")
    threshold_T = summary_rows[0]["threshold_T"] if len(summary_rows) > 0 else None
    plot_focus_ratio_by_model(
        ratios_by_model,
        dataset_name,
        hist_path,
        cfg=cfg,
        threshold_T=threshold_T,
        metadata_text=f"fold=0\nbins={cfg.get('hist_bins', 50)}",
    )
    print(f"\n[SAVE] Summary appended to: {summary_path}")
    print_summary_table(summary_rows, dataset_name)
    print_summary_table_markdown(summary_rows, dataset_name)
    print("\n[DONE]")
