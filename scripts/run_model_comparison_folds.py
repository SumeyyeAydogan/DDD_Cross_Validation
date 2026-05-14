"""
Multi-fold focus-ratio comparison (all folds in one shot).

Uses ``fold_datasets/fold_<k>.json`` for validation images and the same
``runs/<run_name>/fold_<k>/models/fold_<k>.h5`` layout as ``cv_main`` /
``model_comparison_integral._MODEL_RUNS``.

From ``ddd_cv`` root::

    python scripts/run_model_comparison_folds.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.model_comparison_integral import model_configs_for_fold, run_multi_fold_comparison


def main() -> None:
    fold_start = 1
    fold_count = 5
    fold_ids = list(range(fold_start, fold_start + fold_count))

    fold_datasets_dir = str(project_root / "fold_datasets")
    model_map_by_fold = {fid: model_configs_for_fold(fid) for fid in fold_ids}

    weight_type_map_by_label = {
        "original": "original",
        "reward": "reward",
        "log-reward": "log",
        "exp-reward": "exp",
    }

    output_dir = str(project_root / "artifacts" / "model_comparison")
    os.makedirs(output_dir, exist_ok=True)

    result = run_multi_fold_comparison(
        fold_ids=fold_ids,
        fold_datasets_dir=fold_datasets_dir,
        model_map_by_fold=model_map_by_fold,
        output_dir=output_dir,
        experiment_id="ddd_cv_folds",
        weight_type_map_by_label=weight_type_map_by_label,
        config_override={
            "dataset_name": "fold_val",
            "class_names": ["NotDrowsy", "Drowsy"],
            "img_size": (224, 224),
        },
    )

    print("\n[RUN RESULT]")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
