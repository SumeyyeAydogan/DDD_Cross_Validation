from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Metric ids used by model comparison / overlap / weight scripts.
FOCUS_METRIC_CHOICES = (
    "focus_ratio",
    "density_gap",
    "density_gap_shifted",
    "inside_density",
)

FOCUS_METRIC_LABELS = {
    "focus_ratio": "Classic focus ratio (normalized heatmap, mask-weighted mass)",
    "density_gap": "Inside density − outside density (no +1; for comparing models)",
    "density_gap_shifted": "1 + inside density − outside density (same as training weights)",
    "inside_density": "Inside ROI energy / ROI area only",
}


def compute_focus_ratio(heatmap: np.ndarray, mask: np.ndarray) -> float:
    heatmap = np.maximum(heatmap, 0)
    mx = float(heatmap.max())
    if mx > 0:
        heatmap = heatmap / (mx + 1e-8)
    focus = float(np.sum(heatmap * mask))
    total = float(np.sum(heatmap) + 1e-8)
    return float(focus / total)


def compute_mask_density_components(
    heatmap: np.ndarray,
    mask: np.ndarray,
    *,
    roi_threshold: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[float, float, Dict[str, float]]:
    """
  Mask energy densities (heatmap clipped to >= 0, not max-normalized).

  ROI is ``mask >= roi_threshold``; outside is the complement.
  """
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.maximum(hm, 0.0)

    if hm.shape != mask.shape:
        raise ValueError(f"heatmap shape {hm.shape} and mask shape {mask.shape} must match")

    roi = np.asarray(mask, dtype=np.float32) >= float(roi_threshold)
    outside = ~roi

    roi_area = float(np.sum(roi))
    outside_area = float(np.sum(outside))

    inside_energy = float(np.sum(hm[roi])) if roi_area > 0 else 0.0
    outside_energy = float(np.sum(hm[outside])) if outside_area > 0 else 0.0

    inside_density = inside_energy / (roi_area + eps)
    outside_density = outside_energy / (outside_area + eps)

    stats = {
        "inside_energy": inside_energy,
        "outside_energy": outside_energy,
        "inside_area": roi_area,
        "outside_area": outside_area,
        "inside_density": float(inside_density),
        "outside_density": float(outside_density),
    }
    return float(inside_density), float(outside_density), stats


def compute_focus_score(
    heatmap: np.ndarray,
    mask: np.ndarray,
    *,
    metric: str = "focus_ratio",
    roi_threshold: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[float, Dict[str, Any]]:
    """
    Single scalar per image for model comparison / analysis.

    ``density_gap`` uses inside − outside (no +1) so models are comparable.
    ``density_gap_shifted`` matches training sample weights (1 + inside − outside).
    """
    metric = str(metric).strip()
    if metric not in FOCUS_METRIC_CHOICES:
        raise ValueError(f"Unknown focus metric '{metric}'. Choose from: {FOCUS_METRIC_CHOICES}")

    if metric == "focus_ratio":
        score = compute_focus_ratio(heatmap, mask)
        return score, {"focus_metric": metric, "score": score}

    inside_d, outside_d, stats = compute_mask_density_components(
        heatmap,
        mask,
        roi_threshold=roi_threshold,
        eps=eps,
    )
    if metric == "density_gap":
        score = inside_d - outside_d
    elif metric == "density_gap_shifted":
        score = 1.0 + inside_d - outside_d
    elif metric == "inside_density":
        score = inside_d
    else:
        raise ValueError(metric)

    stats = {**stats, "focus_metric": metric, "score": float(score)}
    return float(score), stats


def histogram_right_tail_area(ratios: np.ndarray, threshold: float, bin_edges: np.ndarray) -> float:
    ratios = np.asarray(ratios, dtype=np.float32)
    if ratios.size == 0:
        return 0.0
    hist, edges = np.histogram(ratios, bins=bin_edges, density=True)
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) * 0.5
    area_bins = hist * widths
    return float(np.sum(area_bins[centers >= threshold]))


def empirical_right_tail_probability(ratios: np.ndarray, threshold: float) -> float:
    ratios = np.asarray(ratios, dtype=np.float32)
    if ratios.size == 0:
        return 0.0
    return float(np.mean(ratios > threshold))
