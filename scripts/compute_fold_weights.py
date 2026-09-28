import json
import os
os.environ["PYTHONHASHSEED"] = "42"
from pathlib import Path
import sys
import random
import numpy as np
import tensorflow as tf

project_root = Path(__file__).parent.parent
repo_root = project_root.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(repo_root / "scripts"))

from gradcam_density_gap_weights import compute_fold_weights, plot_weights, save_weights
from log_exp_script import create_exp_weights, create_log_weights
from src.fold_functions import load_fold_datasets
from src.threshold import load_threshold

def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
        
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        

BASE_RUN_NAME = "baseline"
FOCUS_METRIC = "density_gap_shifted"
dataset_dir = os.path.join(project_root, "dataset")
fold_dataset_dir = os.path.join(project_root, "fold_datasets_v2")
runs_root = os.path.join(str(project_root), "runs")
output_weights_path = os.path.join(project_root, f"weights_{FOCUS_METRIC}")
os.makedirs(fold_dataset_dir, exist_ok=True)
os.makedirs(output_weights_path, exist_ok=True)
K = 5
_set_global_seeds(42)

ONE_OFF_READ_WEIGHTS_DIR = str(project_root / "weights")
USE_ONE_OFF_READ_WEIGHTS = False
USE_FOLD_DECISION_THRESHOLD = True
DEFAULT_DECISION_THRESHOLD = 0.5


for fold_idx in range(K):

    fold_n = fold_idx + 1
    out_prefix = os.path.join(output_weights_path, f"fold_{fold_n}")

    if USE_ONE_OFF_READ_WEIGHTS:
        src = os.path.join(ONE_OFF_READ_WEIGHTS_DIR, f"fold_{fold_n}_weights.json")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"One-off reward JSON not found: {src}")
        with open(src, encoding="utf-8") as f:
            fold_weights = json.load(f)
        fold_weights = {str(k): float(v) for k, v in fold_weights.items()}
        reward_json = src
    else:
        fold_tag = f"fold_{fold_n}"
        fold_run_dir = os.path.join(runs_root, BASE_RUN_NAME, fold_tag)
        model_path = os.path.join(fold_run_dir, "models", f"{fold_tag}.h5")
        threshold_path = os.path.join(fold_run_dir, "threshold.json")
        train_files, _, _, _, _, _ = load_fold_datasets(fold_idx, fold_dataset_dir)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
            )
        decision_threshold = DEFAULT_DECISION_THRESHOLD
        if USE_FOLD_DECISION_THRESHOLD:
            if not os.path.isfile(threshold_path):
                raise FileNotFoundError(
                    f"Threshold not found: {threshold_path}\n"
                    "Set USE_FOLD_DECISION_THRESHOLD=False to fall back to 0.5."
                )
            decision_threshold = load_threshold(threshold_path, default=DEFAULT_DECISION_THRESHOLD)
        print(f"Fold {fold_n}: focus_metric={FOCUS_METRIC}, decision_threshold={decision_threshold:.6f}")
        fold_weights = compute_fold_weights(
            train_files,
            model_path,
            dataset_dir,
            cfg={
                "decision_threshold": decision_threshold,
                "focus_metric": FOCUS_METRIC,
            },
        )
        reward_json = f"{out_prefix}_weights.json"
        save_weights(fold_weights, reward_json)
        plot_weights(
            fold_weights,
            save_path=f"{out_prefix}_weights.png",
            focus_metric=FOCUS_METRIC,
        )
    create_log_weights(input_path=reward_json, output_path=f"{out_prefix}_log_weights.json")
    create_exp_weights(input_path=reward_json, output_path=f"{out_prefix}_exp_weights.json")

    if USE_ONE_OFF_READ_WEIGHTS:
        print(f"Fold {fold_n}: log+exp from existing reward -> {out_prefix}_*")
    else:
        print(f"Fold {fold_n}: reward+log+exp JSON + plot -> {out_prefix}_*")
