from __future__ import annotations

import numpy as np


def compute_focus_ratio(heatmap: np.ndarray, mask: np.ndarray) -> float:
    heatmap = np.maximum(heatmap, 0)
    mx = float(heatmap.max())
    if mx > 0:
        heatmap = heatmap / (mx + 1e-8)
    focus = float(np.sum(heatmap * mask))
    total = float(np.sum(heatmap) + 1e-8)
    return float(focus / total)


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