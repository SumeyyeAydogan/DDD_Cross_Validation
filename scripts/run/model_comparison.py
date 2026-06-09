"""
CLI wrapper for multi-fold focus-ratio model comparison.

Example::

    python scripts/run/model_comparison.py --fold-start 1 --fold-count 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

project_root, _ = bootstrap()

from scripts.model_comparison_integral import model_configs_for_fold, run_multi_fold_comparison
from src.focus_metrics import FOCUS_METRIC_CHOICES, FOCUS_METRIC_LABELS


def _parse_model_runs(text: str) -> list:
    rows = []
    for part in text.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        label, run = part.split("=", 1)
        rows.append((label.strip(), run.strip()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Focus-ratio comparison across CV folds.")
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--fold-datasets-dir", type=str, default="fold_datasets")
    parser.add_argument("--output-dir", type=str, default="artifacts/model_comparison")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="default",
        help="Subfolder name under --output-dir to keep runs separated.",
    )
    parser.add_argument("--experiment-id", type=str, default="ddd_cv_folds")
    parser.add_argument(
        "--model-runs",
        type=str,
        default="original=baseline,reward=reward,log-reward=log,exp-reward=exp",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="original",
        help="Label name in --model-runs to use as baseline for thresholds/deltas.",
    )
    parser.add_argument("--background-mask-value", type=float, default=0.0)
    parser.add_argument(
        "--focus-metric",
        type=str,
        default="focus_ratio",
        choices=list(FOCUS_METRIC_CHOICES),
        help="Per-image score: focus_ratio (classic) or density-gap variants (match training).",
    )
    args = parser.parse_args()

    fold_ids = list(range(args.fold_start, args.fold_start + args.fold_count))
    fold_datasets_dir = (
        Path(args.fold_datasets_dir)
        if Path(args.fold_datasets_dir).is_absolute()
        else project_root / args.fold_datasets_dir
    )
    output_dir_base = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else project_root / args.output_dir
    )
    output_tag = str(args.output_tag).strip() or "default"
    output_dir = output_dir_base / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    model_map = {}
    for fid in fold_ids:
        base = model_configs_for_fold(fid)
        overrides = _parse_model_runs(args.model_runs)
        if overrides:
            custom = []
            for label, run_name in overrides:
                p = project_root / "runs" / run_name / f"fold_{fid}" / "models" / f"fold_{fid}.h5"
                custom.append({"label": label, "model_path": str(p)})
            model_map[fid] = custom
        else:
            model_map[fid] = base

    weight_type_map = {label: label for label, _ in _parse_model_runs(args.model_runs)}
    baseline_label = str(args.baseline_label).strip()

    result = run_multi_fold_comparison(
        fold_ids=fold_ids,
        fold_datasets_dir=str(fold_datasets_dir),
        model_map_by_fold=model_map,
        output_dir=str(output_dir),
        experiment_id=args.experiment_id,
        weight_type_map_by_label=weight_type_map,
        baseline_label=baseline_label,
        config_override={
            "dataset_name": "fold_val",
            "class_names": ["NotDrowsy", "Drowsy"],
            "img_size": (224, 224),
            "background_mask_value": float(args.background_mask_value),
            "focus_metric": str(args.focus_metric),
        },
    )
    focus_metric = str(args.focus_metric)
    run_meta = {
        "experiment_id": str(args.experiment_id),
        "output_tag": output_tag,
        "output_dir": str(output_dir),
        "fold_start": int(args.fold_start),
        "fold_count": int(args.fold_count),
        "baseline_label": baseline_label,
        "model_runs": _parse_model_runs(args.model_runs),
        "background_mask_value": float(args.background_mask_value),
        "focus_metric": focus_metric,
        "focus_metric_description": FOCUS_METRIC_LABELS.get(focus_metric, ""),
    }
    meta_path = output_dir / "run_info.json"
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[RUN RESULT]")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"run_info_path: {meta_path}")


if __name__ == "__main__":
    main()
