"""
Compute per-fold GradCAM sample weights (reward + log + exp JSON).

Example::

    python scripts/run/compute_weights.py --base-run baseline --output-dir weights_percentile --background-mask-value 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from _bootstrap import bootstrap

project_root, _ = bootstrap()

import auto_optimize_gradcam_weights as autoopt_weights
import gradcam_density_gap_weights as density_gap_weights
from log_exp_script import create_exp_weights, create_log_weights
from src.fold_functions import load_fold_datasets


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fold GradCAM weights for all CV folds.")
    parser.add_argument("--base-run", type=str, default="baseline", help="Run under runs/ with trained fold models")
    parser.add_argument("--output-dir", type=str, default="weights", help="Output directory for fold_*_weights.json")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--fold-datasets-dir", type=str, default="fold_datasets")
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-mask-value", type=float, default=0.0)
    parser.add_argument("--landmark-box-half-size", type=int, default=12)
    parser.add_argument(
        "--weight-formula",
        type=str,
        choices=["autoopt", "density_gap"],
        default="autoopt",
        help="Weight formula backend: auto-optimized focus reference or density-gap.",
    )
    parser.add_argument(
        "--read-only-reward-dir",
        type=str,
        default=None,
        help="If set, only build log/exp from existing fold_*_weights.json in this dir (no GradCAM).",
    )
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=None, help="Inclusive; default = fold-start + k - 1")
    args = parser.parse_args()

    _set_global_seeds(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    dataset_dir = args.dataset_dir or str(project_root / "dataset")
    fold_dataset_dir = (
        Path(args.fold_datasets_dir)
        if Path(args.fold_datasets_dir).is_absolute()
        else project_root / args.fold_datasets_dir
    )
    runs_root = project_root / args.runs_root
    output_weights_path = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else project_root / args.output_dir
    )
    output_weights_path.mkdir(parents=True, exist_ok=True)

    fold_end = args.fold_end if args.fold_end is not None else args.fold_start + args.k - 1
    read_only = Path(args.read_only_reward_dir).resolve() if args.read_only_reward_dir else None

    cfg = {
        "background_mask_value": float(args.background_mask_value),
        "landmark_box_half_size": int(args.landmark_box_half_size),
    }

    for fold_idx in range(args.fold_start - 1, fold_end):
        fold_n = fold_idx + 1
        out_prefix = output_weights_path / f"fold_{fold_n}"

        if read_only is not None:
            src = read_only / f"fold_{fold_n}_weights.json"
            if not src.is_file():
                raise FileNotFoundError(f"Reward JSON not found: {src}")
            with open(src, encoding="utf-8") as f:
                fold_weights = {str(k): float(v) for k, v in json.load(f).items()}
            reward_json = str(src)
            print(f"Fold {fold_n}: log+exp from {src}")
        else:
            fold_tag = f"fold_{fold_n}"
            model_path = runs_root / args.base_run / fold_tag / "models" / f"{fold_tag}.h5"
            train_files, _, _, _ = load_fold_datasets(fold_idx, str(fold_dataset_dir))
            if not model_path.is_file():
                raise FileNotFoundError(f"Model not found: {model_path}")
            backend = autoopt_weights if args.weight_formula == "autoopt" else density_gap_weights
            fold_weights = backend.compute_fold_weights(
                train_files,
                str(model_path),
                dataset_dir,
                cfg=cfg,
            )
            reward_json = str(out_prefix) + "_weights.json"
            backend.save_weights(fold_weights, reward_json)
            backend.plot_weights(fold_weights, save_path=str(out_prefix) + "_weights.png")
            print(f"Fold {fold_n}: saved {reward_json} ({args.weight_formula})")

        create_log_weights(
            input_path=reward_json,
            output_path=str(out_prefix) + "_log_weights.json",
        )
        create_exp_weights(
            input_path=reward_json,
            output_path=str(out_prefix) + "_exp_weights.json",
        )


if __name__ == "__main__":
    main()
