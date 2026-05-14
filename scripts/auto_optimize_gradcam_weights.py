"""
Auto-optimize GradCAM sample weight parameters
with dynamic landmark-based eye+mouth ROI mask.

IMPROVEMENTS:
1. ✅ Uses dynamic landmark mask (MediaPipe FaceMesh) - better than static mask
2. ✅ Better heatmap resize (bilinear interpolation)
3. ✅ Dynamic parameters based on data distribution
4. ✅ Weight analysis with histogram and statistics

Outputs:
- optimized_gradcam_weights.json  (weights per image)
- gradcam_opt_params.json         (best params)
- gradcam_opt_report.txt          (summary report)
- gradcam_weight_histogram.png    (weight distribution)
"""

import os, json, sys, argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Optional, Set

# Add root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM
from src.analysis_pipeline import get_analysis_pipeline_from_data_dir
from src.focus_metrics import compute_focus_ratio
from src.mask_helpers import create_landmark_mask, create_static_mask


def default_gradcam_cfg() -> dict:
    """Default hyperparameters and paths for GradCAM weight optimization (no globals)."""
    return {
        "model_path": r"runs/30_epoch_baseline/models/final_model.h5",
        "data_dir": r"splitted_dataset/train",
        "img_size": (224, 224),
        "landmark_box_half_size": 12,
        "fallback_to_static": True,
        "background_mask_value": 0.2,
        "class_names": ["NotDrowsy", "Drowsy"],
        "weight_mode": "reward",
        "alpha_max": 3.0,
        "alpha_min": 0.3,
        "clip_range_factor": 0.8,
        "weight_max": 5.0,
        "weight_min": 0.1,
        "lambda_clip": 1.0,
        "lambda_mean": 0.5,
    }


def save_weights(weights: dict, output_path: str) -> None:
    """Write sample-weight map to JSON. Creates parent directories if needed."""
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)


def plot_weights(weights: dict, *, save_path: str) -> None:
    """Quick histogram of weight values (full analysis: plot_weight_histogram)."""
    vals = np.asarray(list(weights.values()), dtype=np.float64)
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Sample weight distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _parse_img_size(raw: str):
    parts = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit("--img-size must be like 224,224")
    return (parts[0], parts[1])


def _parse_class_names(raw: str):
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit("--class-names must be two comma-separated names, e.g. NotDrowsy,Drowsy")
    return [parts[0], parts[1]]


def _build_arg_parser() -> argparse.ArgumentParser:
    d = default_gradcam_cfg()
    parser = argparse.ArgumentParser(description="Auto-optimize GradCAM sample weights.")
    parser.add_argument("--model-path", type=str, default=d["model_path"])
    parser.add_argument("--data-dir", type=str, default=d["data_dir"])
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--img-size", type=str, default=f"{d['img_size'][0]},{d['img_size'][1]}")
    parser.add_argument("--class-names", type=str, default="NotDrowsy,Drowsy")
    parser.add_argument("--weight-mode", type=str, choices=["reward", "penalize"], default=d["weight_mode"])
    parser.add_argument("--roi-padding-px", type=int, default=d["landmark_box_half_size"])
    parser.add_argument("--background-mask-value", type=float, default=d["background_mask_value"])
    parser.add_argument("--fallback-to-static", type=int, choices=[0, 1], default=int(d["fallback_to_static"]))
    return parser

# ================== CORE ======================
def collect_focus_distribution(
    model,
    data_dir,
    img_size,
    cfg: dict,
    include_rel_paths: Optional[Set[str]] = None,
):
    """
    Collect focus ratios for all images using dynamic landmark masks.

    If ``include_rel_paths`` is set, only those paths (relative to ``data_dir``, forward
    slashes) are processed; used by ``compute_fold_weights`` for fold train lists.
    """
    gradcam = CustomGradCAM(model)

    ds, file_paths = get_analysis_pipeline_from_data_dir(
        data_dir=data_dir,
        img_size=img_size,
        class_names=cfg.get("class_names", ["NotDrowsy", "Drowsy"]),
    )

    ratios = []
    filtered_file_paths = []
    landmark_success_count = 0
    static_fallback_count = 0

    data_dir_abs = os.path.abspath(data_dir)
    print("[AutoOpt] Computing focus distribution...")
    print("[AutoOpt] Using landmark mask with optional static fallback")
    print(f"[AutoOpt] Landmark box half-size: {cfg['landmark_box_half_size']}")
    print(f"[AutoOpt] Fallback to static: {cfg['fallback_to_static']}")

    for idx, (data_batch, path_batch) in enumerate(ds):
        images, labels = data_batch
        sample_path = path_batch.numpy()[0].decode("utf-8")
        rel_path = os.path.relpath(os.path.abspath(sample_path), data_dir_abs).replace("\\", "/")
        if include_rel_paths is not None and rel_path not in include_rel_paths:
            print(f"[AutoOpt] Skipping {rel_path} (not in include_rel_paths)")
            continue

        image = images[0].numpy()
        label = int(labels[0].numpy())
        
        # Convert to uint8 for MediaPipe (0-255 range)
        image_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        image_normalized = image_uint8 / 255.0 if image.max() > 1.0 else image

        heatmap = gradcam.compute_heatmap(image_normalized, class_idx=label)
        
        # IMPROVEMENT: Better resize with bilinear interpolation
        # Resize heatmap to match image size with better quality
        heatmap = tf.image.resize(
            heatmap[..., None], 
            img_size, 
            method='bilinear', # Better than default
            antialias=True      # Anti-aliasing for smoother resize
        ).numpy()[..., 0]

        mask = create_landmark_mask(
            image_uint8,
            img_size,
            background_value=float(cfg.get("background_mask_value", 0.2)),
            landmark_box_half_size=int(cfg.get("landmark_box_half_size", 12)),
        )
        if mask is None:
            if cfg['fallback_to_static']:
                mask = create_static_mask(
                    img_size,
                    background_value=float(cfg.get("background_mask_value", 0.2)),
                )
                static_fallback_count += 1
            else:
                # Skip this image if no fallback
                print(f"[WARN] No face detected in image {idx+1}, skipping...")
                continue
        else:
            landmark_success_count += 1

        ratio = compute_focus_ratio(heatmap, mask)
        ratios.append(ratio)
        filtered_file_paths.append(sample_path)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(file_paths)} "
                  f"(landmark: {landmark_success_count}, fallback: {static_fallback_count})")

    print(f"\n[AutoOpt] Summary: {landmark_success_count} landmark masks, "
          f"{static_fallback_count} static fallbacks, {len(ratios)} total processed")

    return ratios, filtered_file_paths


# ================== OPTIMIZATION ======================
def choose_params(ratios, cfg):
    """
    IMPROVEMENT: Dynamic parameter optimization based on data distribution.
    
    Optimizes:
    - target_focus (based on focus distribution with std consideration)
    - alpha (based on virtual penalty sim with more candidates)
    - clip range (scaled from alpha with better bounds)
    """
    ratios = np.array(ratios)
    med = median(ratios)
    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios))
    q25 = float(np.percentile(ratios, 25))
    q75 = float(np.percentile(ratios, 75))

    print(f"[AutoOpt] Focus Ratio Stats: mean={mean_r:.3f}, median={med:.3f}, "
          f"std={std_r:.3f}, Q25={q25:.3f}, Q75={q75:.3f}")

    # --------- IMPROVEMENT: Dynamic Target Focus ---------
    # Use median + adaptive offset based on std
    # If std is high, use smaller offset (more conservative)
    # If std is low, use larger offset (more aggressive)
    adaptive_offset = min(0.15, max(0.05, std_r * 0.5))
    target_focus = min(0.92, max(0.45, med + adaptive_offset))
    
    print(f"[AutoOpt] Adaptive offset: {adaptive_offset:.3f}, Target focus: {target_focus:.3f}")

    # --------- IMPROVEMENT: Weight Distribution Based Alpha Optimization ---------
    # Evaluate alpha based on actual weight distribution, not just penalty
    alpha_max = cfg.get("alpha_max", 3.0)
    alpha_min = cfg.get("alpha_min", 0.3)
    candidates = np.linspace(alpha_min, alpha_max, 20).tolist()  # More granular search
    candidates = [round(c, 2) for c in candidates]  # Round to 2 decimals

    def evaluate_alpha(alpha, ratios, target, cfg):
        """
        Evaluate alpha based on weight distribution quality.
        
        Score = std(weights) - λ_clip * frac_clipped - λ_mean * |mean(weights) - 1|
        
        Higher score = better:
        - High std → good separation between good/bad samples
        - Low clipping → weights stay in natural range
        - Mean ≈ 1 → total loss scaling doesn't change much
        """
        # 1) Calculate clip range from alpha
        clip_range_factor_mult = cfg.get("clip_range_factor", 0.8)
        clip_range_factor = alpha * clip_range_factor_mult
        weight_max = cfg.get("weight_max", 5.0)
        weight_min = cfg.get("weight_min", 0.1)
        clip_min = max(weight_min, 1.0 - clip_range_factor)
        clip_max = min(weight_max, 1.0 + clip_range_factor)
        
        # 2) Calculate weights
        deltas = ratios - target
        weights = 1.0 + alpha * deltas
        weights = np.clip(weights, clip_min, clip_max)
        
        # 3) Calculate metrics
        w_std = float(np.std(weights))
        w_mean = float(np.mean(weights))
        # Fraction of weights that hit the clipping bounds
        frac_clipped = float(np.mean((weights <= clip_min + 1e-6) | (weights >= clip_max - 1e-6)))
        
        # 4) Calculate score with penalties
        lambda_clip = cfg.get("lambda_clip", 1.0)
        lambda_mean = cfg.get("lambda_mean", 0.5)
        
        score = (
            w_std
            - lambda_clip * frac_clipped
            - lambda_mean * abs(w_mean - 1.0)
        )
        
        return score, dict(
            std=w_std,
            mean=w_mean,
            frac_clipped=frac_clipped,
            clip_min=clip_min,
            clip_max=clip_max,
            min_weight=float(np.min(weights)),
            max_weight=float(np.max(weights)),
        )
    
    # Evaluate all candidates
    alpha_scores = {}
    alpha_stats = {}
    for a in candidates:
        score, stats = evaluate_alpha(a, ratios, target_focus, cfg)
        alpha_scores[a] = score
        alpha_stats[a] = stats
    
    # Select best alpha (highest score)
    best_alpha = max(alpha_scores, key=alpha_scores.get)
    best_stats = alpha_stats[best_alpha]
    
    print(f"[AutoOpt] Alpha candidates range: [{alpha_min}, {alpha_max}]")
    print(f"[AutoOpt] Best alpha: {best_alpha:.3f} (score: {alpha_scores[best_alpha]:.4f})")
    print(f"[AutoOpt] Best alpha stats: std={best_stats['std']:.3f}, "
          f"mean={best_stats['mean']:.3f}, frac_clipped={best_stats['frac_clipped']:.3f}")
    print(f"[AutoOpt] Best alpha weight range: [{best_stats['min_weight']:.3f}, {best_stats['max_weight']:.3f}]")
    
    # Show top 3 candidates
    top3 = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"[AutoOpt] Top 3 alpha candidates:")
    for a, score in top3:
        stats = alpha_stats[a]
        print(f"  Alpha {a:.2f}: score={score:.4f}, std={stats['std']:.3f}, "
              f"clipped={stats['frac_clipped']:.3f}, mean={stats['mean']:.3f}")

    # Clip range is already calculated in evaluate_alpha, use it from best_stats
    clip_min = best_stats['clip_min']
    clip_max = best_stats['clip_max']
    
    print(f"[AutoOpt] Final clip range: [{clip_min:.3f}, {clip_max:.3f}]")

    return dict(
        target_focus=float(target_focus),
        alpha=float(best_alpha),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
        # Additional stats for reporting
        mean_focus=float(mean_r),
        median_focus=float(med),
        std_focus=float(std_r),
        q25_focus=float(q25),
        q75_focus=float(q75),
    )


# ================== APPLY ======================
def apply_weights(
    params,
    ratios,
    file_paths,
    cfg: dict,
    output_path: Optional[str] = None,
):
    """
    Apply weights to each image based on focus ratio.
    
    Two modes:
    1. "reward": High focus ratio → High weight (reward good ROI focus)
    2. "penalize": Low focus ratio → High weight (penalize bad ROI focus)
    
    IMPROVEMENT: Returns weight statistics for analysis.
    """
    weights = {}
    weight_values = []
    mode = cfg.get("weight_mode", "reward")

    print(f"\n[AutoOpt] Applying weights (mode: {mode})...")

    for r, fp in zip(ratios, file_paths):
        if mode == "reward":
            # Reward approach: High focus ratio → High weight
            # delta = r - target_focus
            # If r > target_focus → delta > 0 → w > 1 (reward)
            # If r < target_focus → delta < 0 → w < 1 (penalize)
            delta = r - params["target_focus"]
        else:  # "penalize"
            # Penalize approach: Low focus ratio → High weight
            # delta = target_focus - r
            # If r < target_focus → delta > 0 → w > 1 (penalize)
            # If r > target_focus → delta < 0 → w < 1 (reward)
            delta = params["target_focus"] - r
        
        w = 1 + params["alpha"] * delta
        w = float(np.clip(w, params["clip_min"], params["clip_max"]))
        
        weight_values.append(w)
        rel = os.path.relpath(
            os.path.abspath(fp), os.path.abspath(cfg["data_dir"])
        ).replace("\\", "/")
        weights[rel] = w

    if output_path is not None:
        save_weights(weights, output_path)

    # Calculate statistics
    weight_stats = {
        "mean": float(np.mean(weight_values)),
        "median": float(np.median(weight_values)),
        "std": float(np.std(weight_values)),
        "min": float(np.min(weight_values)),
        "max": float(np.max(weight_values)),
        "q25": float(np.percentile(weight_values, 25)),
        "q75": float(np.percentile(weight_values, 75)),
    }
    
    print(f"[AutoOpt] Weight Stats: mean={weight_stats['mean']:.3f}, "
          f"median={weight_stats['median']:.3f}, std={weight_stats['std']:.3f}")
    print(f"[AutoOpt] Weight Range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")

    return weights, weight_values, weight_stats


def compute_fold_weights(
    train_files: List[str],
    model_path: str,
    dataset_dir: str,
    cfg: Optional[Dict] = None,
) -> dict:
    """
    GradCAM-based sample weights for one CV fold train split.

    Keys in the returned dict are paths relative to ``dataset_dir`` with forward slashes
    (e.g. ``Drowsy/A0001.png``), aligned with ``apply_weights`` / training lookup.

    ``train_files`` may be absolute paths as stored in fold JSON; they are normalized against
    ``dataset_dir`` for filtering inside ``collect_focus_distribution``.
    """
    run_cfg = {**default_gradcam_cfg(), **(cfg or {})}
    run_cfg["data_dir"] = os.path.abspath(dataset_dir)
    run_cfg["model_path"] = model_path
    include_rel_paths = {
        os.path.relpath(os.path.abspath(fp), run_cfg["data_dir"]).replace("\\", "/")
        for fp in train_files
    }
    model = tf.keras.models.load_model(model_path, compile=False)
    ratios, file_paths = collect_focus_distribution(
        model, run_cfg["data_dir"], run_cfg["img_size"], run_cfg, include_rel_paths
    )
    if len(ratios) == 0:
        raise RuntimeError(
            "compute_fold_weights: no samples processed. "
            "Check dataset_dir matches fold paths, class layout, and include_rel_paths."
        )
    params = choose_params(ratios, run_cfg)
    weights, _, _ = apply_weights(params, ratios, file_paths, run_cfg, output_path=None)
    return weights


# ================== ANALYSIS ======================
def plot_weight_histogram(weight_values, ratios, params, output_path):
    """
    IMPROVEMENT: Create histogram and analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Weight distribution histogram
    axes[0, 0].hist(weight_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(weight_values), color='r', linestyle='--', label=f'Mean: {np.mean(weight_values):.3f}')
    axes[0, 0].axvline(np.median(weight_values), color='g', linestyle='--', label=f'Median: {np.median(weight_values):.3f}')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Weight Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Focus ratio distribution
    axes[0, 1].hist(ratios, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(params["target_focus"], color='r', linestyle='--', 
                       label=f'Target: {params["target_focus"]:.3f}')
    axes[0, 1].axvline(np.median(ratios), color='g', linestyle='--', 
                       label=f'Median: {np.median(ratios):.3f}')
    axes[0, 1].set_xlabel('Focus Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Focus Ratio Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Weight vs Focus Ratio scatter
    axes[1, 0].scatter(ratios, weight_values, alpha=0.3, s=10)
    axes[1, 0].axvline(params["target_focus"], color='r', linestyle='--', 
                       label=f'Target: {params["target_focus"]:.3f}')
    axes[1, 0].set_xlabel('Focus Ratio')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Weight vs Focus Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Weight statistics text
    stats_text = f"""
Weight Statistics:
  Mean: {np.mean(weight_values):.3f}
  Median: {np.median(weight_values):.3f}
  Std: {np.std(weight_values):.3f}
  Min: {np.min(weight_values):.3f}
  Max: {np.max(weight_values):.3f}
  Q25: {np.percentile(weight_values, 25):.3f}
  Q75: {np.percentile(weight_values, 75):.3f}

Focus Ratio Statistics:
  Mean: {np.mean(ratios):.3f}
  Median: {np.median(ratios):.3f}
  Std: {np.std(ratios):.3f}
  Target: {params["target_focus"]:.3f}

Parameters:
  Alpha: {params["alpha"]:.3f}
  Clip Range: [{params["clip_min"]:.3f}, {params["clip_max"]:.3f}]
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[AutoOpt] Histogram saved: {output_path}")


# ================== MAIN ======================
if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = {**default_gradcam_cfg()}
    cfg["model_path"] = args.model_path
    cfg["data_dir"] = args.data_dir
    cfg["img_size"] = _parse_img_size(args.img_size)
    cfg["class_names"] = _parse_class_names(args.class_names)
    cfg["weight_mode"] = str(args.weight_mode)
    cfg["landmark_box_half_size"] = int(args.roi_padding_px)
    cfg["background_mask_value"] = float(args.background_mask_value)
    cfg["fallback_to_static"] = bool(int(args.fallback_to_static))
    artifacts_dir = args.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    model = tf.keras.models.load_model(cfg["model_path"], compile=False)

    ratios, file_paths = collect_focus_distribution(
        model, cfg["data_dir"], cfg["img_size"], cfg, include_rel_paths=None
    )
    if len(ratios) == 0:
        raise RuntimeError(
            "No samples were processed for optimization. "
            "Check data_dir layout, class names, and that images load correctly."
        )    

    params = choose_params(ratios, cfg)
    print("\n[AutoOpt] Best Params:", {k: v for k, v in params.items() 
                                       if k not in ['mean_focus', 'median_focus', 'std_focus', 'q25_focus', 'q75_focus']})

    # SAVE PARAMS
    with open(os.path.join(artifacts_dir, "gradcam_opt_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # APPLY & SAVE
    weights, weight_values, weight_stats = apply_weights(
        params,
        ratios,
        file_paths,
        cfg,
        os.path.join(artifacts_dir, "optimized_gradcam_weights.json"),
    )

    # IMPROVEMENT: Create histogram
    plot_weight_histogram(
        weight_values,
        ratios,
        params,
        os.path.join(artifacts_dir, "gradcam_weight_histogram.png"),
    )

    # IMPROVEMENT: Enhanced report
    with open(os.path.join(artifacts_dir, "gradcam_opt_report.txt"), "w") as f:
        f.write("=== GradCAM Optimization Report (IMPROVED) ===\n\n")
        f.write("Weight Mode:\n")
        mode = cfg.get("weight_mode", "reward")
        f.write(f"  Mode: {mode}\n")
        if mode == "reward":
            f.write("  Strategy: High focus ratio → High weight (reward good ROI focus)\n")
        else:
            f.write("  Strategy: Low focus ratio → High weight (penalize bad ROI focus)\n")
        f.write("\nFocus Ratio Statistics:\n")
        f.write(f"  Mean: {params.get('mean_focus', np.mean(ratios)):.3f}\n")
        f.write(f"  Median: {params.get('median_focus', median(ratios)):.3f}\n")
        f.write(f"  Std: {params.get('std_focus', np.std(ratios)):.3f}\n")
        f.write(f"  Q25: {params.get('q25_focus', np.percentile(ratios, 25)):.3f}\n")
        f.write(f"  Q75: {params.get('q75_focus', np.percentile(ratios, 75)):.3f}\n\n")
        f.write("Optimized Parameters:\n")
        f.write(f"  Target Focus: {params['target_focus']:.3f}\n")
        f.write(f"  Alpha: {params['alpha']:.3f}\n")
        f.write(f"  Clip Range: [{params['clip_min']:.3f}, {params['clip_max']:.3f}]\n\n")
        f.write("Weight Statistics:\n")
        for key, value in weight_stats.items():
            f.write(f"  {key.capitalize()}: {value:.3f}\n")
        f.write(f"\nTotal Samples: {len(ratios)}\n")
        f.write(f"\nMask Configuration:\n")
        f.write(f"  Background Mask Value: {cfg.get('background_mask_value', 0.0):.2f}\n")
        f.write(f"  Landmark Box Half-Size: {cfg['landmark_box_half_size']}\n")
        f.write(f"  Fallback to Static: {cfg['fallback_to_static']}\n")
        f.write(f"\nWeight Optimization Configuration:\n")
        f.write(f"  Alpha Range: [{cfg.get('alpha_min', 0.3):.2f}, {cfg.get('alpha_max', 3.0):.2f}]\n")
        f.write(f"  Clip Range Factor: {cfg.get('clip_range_factor', 0.8):.2f}\n")
        f.write(f"  Weight Bounds: [{cfg.get('weight_min', 0.1):.2f}, {cfg.get('weight_max', 5.0):.2f}]\n")
        f.write(f"  Lambda Clip (clipping penalty): {cfg.get('lambda_clip', 1.0):.2f}\n")
        f.write(f"  Lambda Mean (mean deviation penalty): {cfg.get('lambda_mean', 0.5):.2f}\n")
        f.write(f"\nAlpha Selection Method:\n")
        f.write(f"  Score = std(weights) - lambda_clip * frac_clipped - lambda_mean * |mean(weights) - 1|\n")
        f.write(f"  Higher score = better separation with balanced distribution\n")

    print(f"\n[AutoOpt] DONE. Files saved in {artifacts_dir}")
    print("  - optimized_gradcam_weights.json")
    print("  - gradcam_opt_params.json")
    print("  - gradcam_opt_report.txt")
    print("  - gradcam_weight_histogram.png")
