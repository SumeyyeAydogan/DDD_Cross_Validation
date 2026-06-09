"""
CLI wrapper for overlap_accuracy_comparison (configurable model runs & output tag).

Example::

    python scripts/run/overlap_comparison.py --output-tag my_overlap \\
        --model-runs baseline=baseline,reward=reward,log=log,exp=exp
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

import overlap_accuracy_comparison as overlap  # noqa: E402


def _parse_model_runs(text: str) -> dict:
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected label=run, got: {part}")
        label, run = part.split("=", 1)
        out[label.strip()] = run.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlap / focus comparison on val set.")
    parser.add_argument("--output-tag", type=str, default="default")
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--background-mask-value", type=float, default=0.0)
    parser.add_argument("--landmark-box-half-size", type=int, default=12)
    parser.add_argument(
        "--model-runs",
        type=str,
        default="baseline=baseline,reward=reward,log=log,exp=exp",
        help="Comma-separated label=runs_folder",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="baseline",
        help="Label from --model-runs to use as baseline.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Re-run folds even if summary.json exists")
    parser.add_argument("--save-detail-csv", action="store_true")
    parser.add_argument(
        "--focus-metric",
        type=str,
        default="focus_ratio",
        choices=["focus_ratio", "density_gap", "density_gap_shifted", "inside_density"],
    )
    args = parser.parse_args()

    overlap.CONFIG["output_tag"] = args.output_tag
    overlap.CONFIG["fold_start"] = int(args.fold_start)
    overlap.CONFIG["fold_count"] = int(args.fold_count)
    overlap.CONFIG["background_mask_value"] = float(args.background_mask_value)
    overlap.CONFIG["landmark_box_half_size"] = int(args.landmark_box_half_size)
    overlap.CONFIG["model_runs"] = _parse_model_runs(args.model_runs)
    overlap.CONFIG["baseline_label"] = str(args.baseline_label).strip()
    overlap.CONFIG["resume_skip_fold_if_summary_exists"] = not args.no_resume
    overlap.CONFIG["save_fold_detail_csv"] = bool(args.save_detail_csv)
    overlap.CONFIG["focus_metric"] = str(args.focus_metric)

    overlap.main()


if __name__ == "__main__":
    main()
