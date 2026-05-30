"""
GradCAM sample weights with mask energy-density gap.

Formula per image:
    weight = (energy inside ROI / ROI area) - (energy outside ROI / outside area)

Important:
- This file keeps the public function names used by the existing flow:
  compute_fold_weights, save_weights, plot_weights.
- GradCAM target is the model prediction, not the true label.
- No alpha/reference/clip optimization is applied.
- Raw weights can be negative. Use as-is only if your training pipeline intentionally
  accepts negative sample_weight values.
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add root path, matching the old auto_optimize_gradcam_weights.py style.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM
from src.ds_with_paths_pipeline import get_dataset_with_paths
from src.mask_helpers import (
    create_landmark_mask,
    create_static_mask,
    image_to_float01_rgb,
    image_to_uint8_rgb,
)


def default_gradcam_cfg() -> dict:
    """Default configuration for density-gap GradCAM weights."""
    return {
        "model_path": r"runs/baseline/models/final_model.h5",
        "data_dir": r"dataset",
        "img_size": (224, 224),
        "class_names": ["NotDrowsy", "Drowsy"],
        "landmark_box_half_size": 12,
        "fallback_to_static": True,
        "background_mask_value": 0.0,
        # ROI pixels are mask == 1.0. This threshold keeps the formula correct
        # even if background_mask_value is 0.2, 0.0, etc.
        "roi_threshold": 0.5,
        "eps": 1e-8,
    }


def save_weights(weights: dict, output_path: str) -> None:
    """Write sample-weight map to JSON. Creates parent directories if needed."""
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)


def plot_weights(weights: dict, *, save_path: str) -> None:
    """Save a histogram of raw density-gap sample weights."""
    vals = np.asarray(list(weights.values()), dtype=np.float64)
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.75)
    ax.axvline(float(np.mean(vals)), linestyle="--", label=f"Mean: {np.mean(vals):.4f}")
    ax.axvline(float(np.median(vals)), linestyle="--", label=f"Median: {np.median(vals):.4f}")
    ax.axvline(0.0, linestyle=":", label="Zero")
    ax.set_xlabel("Weight = mean_heatmap_inside_ROI - mean_heatmap_outside_ROI")
    ax.set_ylabel("Count")
    ax.set_title("GradCAM density-gap sample weight distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _predict_binary_class(model: tf.keras.Model, image_float01: np.ndarray) -> Tuple[int, float]:
    """
    Predict binary class for a single HxWx3 image.

    Assumes a single sigmoid output where output[0] is P(class 1).
    Returns:
        pred_cls: 0 or 1
        prob1: P(class 1)
    """
    pred = model(image_float01[None, ...], training=False).numpy().ravel()
    if pred.size != 1:
        raise ValueError(
            f"Expected a single-output sigmoid binary model, got output shape {pred.shape}. "
            "This density-gap file is written for binary sigmoid models."
        )
    prob1 = float(pred[0])
    pred_cls = 1 if prob1 >= 0.5 else 0
    return pred_cls, prob1


def compute_mask_density_gap(
    heatmap: np.ndarray,
    mask: np.ndarray,
    *,
    roi_threshold: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[float, Dict[str, float]]:
    """
    Training sample weight: ``1 + inside_density - outside_density``.

    Uses ``density_gap_shifted`` from ``src.focus_metrics`` (heatmap not max-normalized).
    """
    from src.focus_metrics import compute_focus_score

    weight, stats = compute_focus_score(
        heatmap,
        mask,
        metric="density_gap_shifted",
        roi_threshold=roi_threshold,
        eps=eps,
    )
    stats["weight"] = float(weight)
    return float(weight), stats


def collect_true_class_focus_ratios_for_weighting(
    model: tf.keras.Model,
    data_dir: str,
    img_size: Tuple[int, int],
    cfg: dict,
    include_rel_paths: Optional[Set[str]] = None,
):
    """
    Compatibility name kept from the old file.

    Despite the old name, this function now:
    - uses the PREDICTED class as the GradCAM target,
    - computes raw density-gap weights directly,
    - returns (weight_values, filtered_file_paths, per_sample_stats).
    """
    gradcam = CustomGradCAM(model)

    ds, file_paths = get_dataset_with_paths(
        data_dir=data_dir,
        img_size=img_size,
        class_names=cfg.get("class_names", ["NotDrowsy", "Drowsy"]),
    )

    weight_values: List[float] = []
    filtered_file_paths: List[str] = []
    sample_stats: List[Dict[str, float]] = []

    landmark_success_count = 0
    static_fallback_count = 0
    skipped_no_mask_count = 0

    data_dir_abs = os.path.abspath(data_dir)
    print("[DensityGap] Computing GradCAM density-gap weights...")
    print("[DensityGap] Formula: mean_heatmap_inside_ROI - mean_heatmap_outside_ROI")
    print("[DensityGap] GradCAM target: predicted class")
    print(f"[DensityGap] Landmark box half-size: {cfg['landmark_box_half_size']}")
    print(f"[DensityGap] Background mask value: {cfg['background_mask_value']}")
    print(f"[DensityGap] ROI threshold: {cfg.get('roi_threshold', 0.5)}")
    print(f"[DensityGap] Fallback to static: {cfg['fallback_to_static']}")

    for idx, (data_batch, path_batch) in enumerate(ds):
        images, labels = data_batch
        sample_path = path_batch.numpy()[0].decode("utf-8")
        rel_path = os.path.relpath(os.path.abspath(sample_path), data_dir_abs).replace("\\", "/")

        if include_rel_paths is not None and rel_path not in include_rel_paths:
            continue

        image = images[0].numpy()
        true_label = int(labels[0].numpy())

        image_rgb_uint8 = image_to_uint8_rgb(image)
        image_float01 = image_to_float01_rgb(image)

        pred_cls, prob1 = _predict_binary_class(model, image_float01)

        # IMPORTANT: predicted class is used here, not true_label.
        heatmap = gradcam.compute_heatmap(image_float01, class_idx=pred_cls)

        # Resize heatmap to the same spatial size as the mask / image.
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
        if mask is None:
            if cfg.get("fallback_to_static", True):
                mask = create_static_mask(
                    img_size,
                    background_mask_value=float(cfg.get("background_mask_value", 0.0)),
                )
                static_fallback_count += 1
            else:
                skipped_no_mask_count += 1
                print(f"[WARN] No face detected for {rel_path}; skipping.")
                continue
        else:
            landmark_success_count += 1

        weight, stats = compute_mask_density_gap(
            heatmap,
            mask,
            roi_threshold=float(cfg.get("roi_threshold", 0.5)),
            eps=float(cfg.get("eps", 1e-8)),
        )
        stats.update(
            {
                "true_label": float(true_label),
                "pred_cls": float(pred_cls),
                "prob1": float(prob1),
                "correct": float(pred_cls == true_label),
            }
        )

        weight_values.append(weight)
        filtered_file_paths.append(sample_path)
        sample_stats.append(stats)

        if (idx + 1) % 50 == 0:
            print(
                f"  Processed dataset index {idx + 1}/{len(file_paths)} "
                f"(kept={len(weight_values)}, landmark={landmark_success_count}, "
                f"fallback={static_fallback_count}, skipped={skipped_no_mask_count})"
            )

    print(
        f"\n[DensityGap] Summary: {landmark_success_count} landmark masks, "
        f"{static_fallback_count} static fallbacks, {skipped_no_mask_count} skipped, "
        f"{len(weight_values)} total weights"
    )

    return weight_values, filtered_file_paths, sample_stats


def _weights_from_values(
    weight_values: List[float],
    file_paths: List[str],
    cfg: dict,
) -> Dict[str, float]:
    """Convert raw per-sample values to rel_path -> weight mapping."""
    weights: Dict[str, float] = {}
    base_abs = os.path.abspath(cfg["data_dir"])
    for w, fp in zip(weight_values, file_paths):
        rel = os.path.relpath(os.path.abspath(fp), base_abs).replace("\\", "/")
        weights[rel] = float(w)
    return weights


def _print_weight_stats(weight_values: List[float], sample_stats: Optional[List[Dict[str, float]]] = None) -> None:
    vals = np.asarray(weight_values, dtype=np.float64)
    if vals.size == 0:
        print("[DensityGap] No weights to summarize.")
        return

    print(
        "[DensityGap] Weight stats: "
        f"mean={np.mean(vals):.6f}, median={np.median(vals):.6f}, std={np.std(vals):.6f}, "
        f"min={np.min(vals):.6f}, max={np.max(vals):.6f}, "
        f"negative_frac={np.mean(vals < 0):.3f}"
    )

    if sample_stats:
        inside = np.asarray([s["inside_density"] for s in sample_stats], dtype=np.float64)
        outside = np.asarray([s["outside_density"] for s in sample_stats], dtype=np.float64)
        correct = np.asarray([s["correct"] for s in sample_stats], dtype=np.float64)
        print(
            "[DensityGap] Density stats: "
            f"inside_mean={np.mean(inside):.6f}, outside_mean={np.mean(outside):.6f}, "
            f"pred_correct_frac={np.mean(correct):.3f}"
        )


# Kept only for compatibility with old imports/calls. No optimization is performed.
def choose_params(weight_values, cfg):
    vals = np.asarray(weight_values, dtype=np.float64)
    return {
        "method": "raw_mask_energy_density_gap",
        "target": "predicted_class",
        "count": int(vals.size),
        "mean": float(np.mean(vals)) if vals.size else 0.0,
        "median": float(np.median(vals)) if vals.size else 0.0,
        "std": float(np.std(vals)) if vals.size else 0.0,
        "min": float(np.min(vals)) if vals.size else 0.0,
        "max": float(np.max(vals)) if vals.size else 0.0,
    }


# Kept for compatibility. Since weight_values are already final weights, params is ignored.
def apply_weights(
    params,
    ratios,
    file_paths,
    cfg: dict,
    output_path: Optional[str] = None,
):
    weights = _weights_from_values(list(ratios), list(file_paths), cfg)
    weight_values = [float(v) for v in weights.values()]
    weight_stats = choose_params(weight_values, cfg)
    if output_path is not None:
        save_weights(weights, output_path)
    _print_weight_stats(weight_values)
    return weights, weight_values, weight_stats


def compute_fold_weights(
    train_files: List[str],
    model_path: str,
    dataset_dir: str,
    cfg: Optional[Dict] = None,
) -> dict:
    """
    GradCAM density-gap sample weights for one CV fold train split.

    Public API intentionally matches the old auto_optimize_gradcam_weights.compute_fold_weights.

    Keys in the returned dict are paths relative to dataset_dir with forward slashes
    e.g. "Drowsy/A0001.png".
    """
    run_cfg = {**default_gradcam_cfg(), **(cfg or {})}
    run_cfg["data_dir"] = os.path.abspath(dataset_dir)
    run_cfg["model_path"] = model_path

    include_rel_paths = {
        os.path.relpath(os.path.abspath(fp), run_cfg["data_dir"]).replace("\\", "/")
        for fp in train_files
    }

    model = tf.keras.models.load_model(model_path, compile=False)

    weight_values, file_paths, sample_stats = collect_true_class_focus_ratios_for_weighting(
        model,
        run_cfg["data_dir"],
        run_cfg["img_size"],
        run_cfg,
        include_rel_paths=include_rel_paths,
    )

    if len(weight_values) == 0:
        raise RuntimeError(
            "compute_fold_weights: no samples processed. Check dataset_dir, fold paths, "
            "class layout, face detection, and include_rel_paths."
        )

    weights = _weights_from_values(weight_values, file_paths, run_cfg)
    _print_weight_stats(weight_values, sample_stats)
    return weights
