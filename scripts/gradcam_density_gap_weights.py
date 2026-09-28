"""
GradCAM sample weights from mask energy-density metrics.

Supported ``focus_metric`` values (see ``src.focus_metrics``):
    inside_density      — energy inside ROI / ROI area
    density_gap         — inside density − outside density
    density_gap_shifted — 1 + inside density − outside density

Uses DDD eye+mouth landmark box masks (see ``src.mask_helpers``).
GradCAM target class follows model prediction via ``decision_threshold``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ds_with_paths_pipeline import get_dataset_with_paths
from src.focus_metrics import FOCUS_METRIC_LABELS, compute_focus_score
from src.gradcam import CustomGradCAM
from src.mask_helpers import (
    create_landmark_mask,
    create_static_mask,
    image_to_float01_rgb,
    image_to_uint8_rgb,
)


def default_gradcam_cfg() -> dict:
    return {
        "model_path": r"runs/baseline/models/final_model.h5",
        "data_dir": r"dataset",
        "img_size": (224, 224),
        "class_names": ["NotDrowsy", "Drowsy"],
        "focus_metric": "density_gap_shifted",
        "fallback_to_static": True,
        "background_mask_value": 0.0,
        "landmark_box_half_size": 12,
        "roi_threshold": 0.5,
        "decision_threshold": 0.5,
        "eps": 1e-8,
    }


def _focus_metric(cfg: dict) -> str:
    return str(cfg.get("focus_metric", "density_gap_shifted")).strip()


def save_weights(weights: dict, output_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)


def plot_weights(
    weights: dict,
    *,
    save_path: str,
    focus_metric: str = "density_gap_shifted",
) -> None:
    vals = np.asarray(list(weights.values()), dtype=np.float64)
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    label = FOCUS_METRIC_LABELS.get(focus_metric, focus_metric)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.75)
    ax.axvline(float(np.mean(vals)), linestyle="--", label=f"Mean: {np.mean(vals):.4f}")
    ax.axvline(float(np.median(vals)), linestyle="--", label=f"Median: {np.median(vals):.4f}")
    ax.axvline(0.0, linestyle=":", label="Zero")
    ax.set_xlabel(f"Weight ({focus_metric})")
    ax.set_ylabel("Count")
    ax.set_title(f"GradCAM sample weights — {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _predict_binary_class(
    model: tf.keras.Model,
    image_float01: np.ndarray,
    *,
    decision_threshold: float = 0.5,
) -> Tuple[int, float]:
    pred = model(image_float01[None, ...], training=False).numpy().ravel()
    if pred.size != 1:
        raise ValueError(
            f"Expected a single-output sigmoid binary model, got output shape {pred.shape}."
        )
    prob1 = float(pred[0])
    pred_cls = 1 if prob1 >= float(decision_threshold) else 0
    return pred_cls, prob1


def compute_sample_weights(
    model: tf.keras.Model,
    data_dir: str,
    img_size: Tuple[int, int],
    cfg: dict,
    include_rel_paths: Optional[Set[str]] = None,
) -> Tuple[List[float], List[str], List[Dict[str, float]]]:
    metric = _focus_metric(cfg)
    gradcam = CustomGradCAM(model)
    ds, file_paths = get_dataset_with_paths(
        data_dir=data_dir,
        img_size=img_size,
        class_names=cfg.get("class_names", ["NotDrowsy", "Drowsy"]),
    )

    weight_values: List[float] = []
    kept_paths: List[str] = []
    sample_stats: List[Dict[str, float]] = []
    landmark_ok = static_fallback = skipped = 0

    data_dir_abs = os.path.abspath(data_dir)
    print(f"[GradCAM weights] metric={metric}")
    print(f"[GradCAM weights] {FOCUS_METRIC_LABELS.get(metric, metric)}")
    print("[GradCAM weights] GradCAM target: predicted class")
    print(f"[GradCAM weights] Landmark box half-size: {cfg.get('landmark_box_half_size', 12)}")
    print(f"[GradCAM weights] Background mask value: {cfg['background_mask_value']}")
    print(f"[GradCAM weights] ROI threshold: {cfg.get('roi_threshold', 0.5)}")
    print(f"[GradCAM weights] Decision threshold: {cfg.get('decision_threshold', 0.5)}")
    print(f"[GradCAM weights] Fallback to static: {cfg['fallback_to_static']}")

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

        pred_cls, prob1 = _predict_binary_class(
            model,
            image_float01,
            decision_threshold=float(cfg.get("decision_threshold", 0.5)),
        )
        heatmap = gradcam.compute_heatmap(image_float01, class_idx=pred_cls)
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
                static_fallback += 1
            else:
                skipped += 1
                print(f"[WARN] No face detected for {rel_path}; skipping.")
                continue
        else:
            landmark_ok += 1

        weight, stats = compute_focus_score(
            heatmap,
            mask,
            metric=metric,
            roi_threshold=float(cfg.get("roi_threshold", 0.5)),
            eps=float(cfg.get("eps", 1e-8)),
        )
        stats["weight"] = float(weight)
        stats.update(
            {
                "true_label": float(true_label),
                "pred_cls": float(pred_cls),
                "prob1": float(prob1),
                "correct": float(pred_cls == true_label),
            }
        )

        weight_values.append(weight)
        kept_paths.append(sample_path)
        sample_stats.append(stats)

        if (idx + 1) % 50 == 0:
            print(
                f"  index {idx + 1}/{len(file_paths)} "
                f"(kept={len(weight_values)}, landmark={landmark_ok}, "
                f"fallback={static_fallback}, skipped={skipped})"
            )

    print(
        f"\n[GradCAM weights] Done: {landmark_ok} landmark, {static_fallback} static, "
        f"{skipped} skipped, {len(weight_values)} weights"
    )
    return weight_values, kept_paths, sample_stats


def _weights_from_values(
    weight_values: List[float],
    file_paths: List[str],
    cfg: dict,
) -> Dict[str, float]:
    base_abs = os.path.abspath(cfg["data_dir"])
    return {
        os.path.relpath(os.path.abspath(fp), base_abs).replace("\\", "/"): float(w)
        for w, fp in zip(weight_values, file_paths)
    }


def _print_weight_stats(
    weight_values: List[float],
    sample_stats: Optional[List[Dict[str, float]]] = None,
    *,
    focus_metric: str = "density_gap_shifted",
) -> None:
    vals = np.asarray(weight_values, dtype=np.float64)
    if vals.size == 0:
        print("[GradCAM weights] No weights to summarize.")
        return

    print(
        f"[GradCAM weights] {focus_metric} stats: "
        f"mean={np.mean(vals):.6f}, median={np.median(vals):.6f}, std={np.std(vals):.6f}, "
        f"min={np.min(vals):.6f}, max={np.max(vals):.6f}, "
        f"negative_frac={np.mean(vals < 0):.3f}"
    )
    if sample_stats and "inside_density" in sample_stats[0]:
        inside = np.asarray([s["inside_density"] for s in sample_stats], dtype=np.float64)
        outside = np.asarray([s["outside_density"] for s in sample_stats], dtype=np.float64)
        correct = np.asarray([s["correct"] for s in sample_stats], dtype=np.float64)
        print(
            "[GradCAM weights] components: "
            f"inside_mean={np.mean(inside):.6f}, outside_mean={np.mean(outside):.6f}, "
            f"pred_correct_frac={np.mean(correct):.3f}"
        )


def compute_fold_weights(
    train_files: List[str],
    model_path: str,
    dataset_dir: str,
    cfg: Optional[Dict] = None,
) -> dict:
    run_cfg = {**default_gradcam_cfg(), **(cfg or {})}
    run_cfg["data_dir"] = os.path.abspath(dataset_dir)
    run_cfg["model_path"] = model_path
    metric = _focus_metric(run_cfg)

    include_rel_paths = {
        os.path.relpath(os.path.abspath(fp), run_cfg["data_dir"]).replace("\\", "/")
        for fp in train_files
    }
    model = tf.keras.models.load_model(model_path, compile=False)

    weight_values, file_paths, sample_stats = compute_sample_weights(
        model,
        run_cfg["data_dir"],
        run_cfg["img_size"],
        run_cfg,
        include_rel_paths=include_rel_paths,
    )
    if not weight_values:
        raise RuntimeError(
            "compute_fold_weights: no samples processed. Check dataset_dir, fold paths, "
            "class layout, face detection, and include_rel_paths."
        )

    weights = _weights_from_values(weight_values, file_paths, run_cfg)
    _print_weight_stats(weight_values, sample_stats, focus_metric=metric)
    return weights
