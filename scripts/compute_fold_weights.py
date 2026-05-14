import json
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
from pathlib import Path
import sys
import random
import numpy as np
import tensorflow as tf

# Add root path (ddd_cv) and repo root (for scripts/log_exp_script.py)
project_root = Path(__file__).parent.parent
repo_root = project_root.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(repo_root / "scripts"))

from auto_optimize_gradcam_weights import compute_fold_weights, plot_weights, save_weights
from log_exp_script import create_exp_weights, create_log_weights
from src.fold_functions import load_fold_datasets

def _set_global_determinism(seed: int) -> None:
    """Set Python/NumPy/TensorFlow seeds and request deterministic TF ops."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
        
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        

BASE_RUN_NAME = "ddd_cv_30_eagerly_false_set-seed_pre-model"
dataset_dir = os.path.join(project_root, "dataset")
fold_dataset_dir = os.path.join(project_root, "fold_datasets")
runs_root = os.path.join(str(project_root), "runs")
output_weights_path = os.path.join(project_root, "weights")
os.makedirs(fold_dataset_dir, exist_ok=True)
os.makedirs(output_weights_path, exist_ok=True)
K = 5
_set_global_determinism(42)

# ddd_cv/weights — portable relative to project root (same layout on any machine).
ONE_OFF_READ_WEIGHTS_DIR = str(project_root / "weights")
# True: read fold_*_weights.json from the dir above; only log/exp (no re-save / no reward plot). False: GradCAM.
USE_ONE_OFF_READ_WEIGHTS = False


# 4) Compute fold weights for each fold
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
        reward_json = src  # already on disk; skip re-save and histogram
    else:
        fold_tag = f"fold_{fold_n}"
        fold_run_dir = os.path.join(runs_root, BASE_RUN_NAME, fold_tag)
        model_path = os.path.join(fold_run_dir, "models", f"{fold_tag}.h5")
        train_files, train_labels, val_files, val_labels = load_fold_datasets(
            fold_idx, fold_dataset_dir
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
            )
        fold_weights = compute_fold_weights(train_files, model_path, dataset_dir)
        reward_json = f"{out_prefix}_weights.json"
        save_weights(fold_weights, reward_json)
        plot_weights(fold_weights, save_path=f"{out_prefix}_weights.png")
    # Log / exp variants from the same reward JSON (no second GradCAM pass)
    create_log_weights(input_path=reward_json, output_path=f"{out_prefix}_log_weights.json")
    create_exp_weights(input_path=reward_json, output_path=f"{out_prefix}_exp_weights.json")
    # Optional: save_weights(summary_dict, f"{out_prefix}_summary.json")

    if USE_ONE_OFF_READ_WEIGHTS:
        print(f"Fold {fold_n}: log+exp from existing reward -> {out_prefix}_*")
    else:
        print(f"Fold {fold_n}: reward+log+exp JSON + plot -> {out_prefix}_*")