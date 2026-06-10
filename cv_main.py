import os

# Must run before any `src.*` import that may load TensorFlow (e.g. fold_functions → cv_dataloader).
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from datetime import datetime
from typing import List

import random
import numpy as np
import tensorflow as tf

from src.run_manager import RunManager
from src.fold_functions import save_fold_datasets, create_tf_datasets_for_fold
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import plot_history, plot_metrics, save_cv_summary
from src.callbacks import get_training_callbacks, get_time_history, summarize_epoch_times

def _set_global_determinism(seed: int) -> None:
    """Set Python/NumPy/TensorFlow seeds (TF determinism env vars set before tf import)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


project_root = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_root, "dataset")
# Output: runs/<run_name>/fold_<k>/... (match scripts/model_comparison_integral.py and overlap_accuracy_comparison.py)
run_manager = RunManager(run_name="baseline")
output_dir = os.path.join(project_root, "fold_datasets_v2")
inner_val_ratio = 0.15
# Reward / GradCAM sample weights: directory that contains fold_1_weights.json … fold_K_weights.json
# (one file per fold; keys are paths relative to dataset_dir). None → default <project>/weights
REWARD_WEIGHTS_DIR = None  # e.g. os.path.join(project_root, "weights", "my_reward_run")
weights_dir = REWARD_WEIGHTS_DIR if REWARD_WEIGHTS_DIR else os.path.join(project_root, "weights")
os.makedirs(output_dir, exist_ok=True)

# 1) Save initial config
config = {
    "dataset_dir": dataset_dir,
    "class_names": ("NotDrowsy", "Drowsy"),
    "run_name": run_manager.run_name,
    "img_size": (224, 224),
    "k": 5,
    "seed": 42,
    "epochs_per_fold": 30,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "started_at": str(datetime.now()),
    "output_dir": output_dir,
    "inner_val_ratio": inner_val_ratio,
    "weights_dir": weights_dir,
}
run_manager.save_config(config)

# 2) Set global determinism (before GPU memory growth / heavy TF work)
_set_global_determinism(config["seed"])

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

# 3) Save fold datasets (run once, then comment out)
#save_fold_datasets(config["dataset_dir"], config["k"], config["img_size"], config["seed"], config["class_names"], config["output_dir"], val_ratio=config["inner_val_ratio"])

val_acc_per_fold: List[float] = []
val_auc_per_fold: List[float] = []
fold_training_times: List[dict] = []

# 4) Train model for each fold
for fold_idx in range(config["k"]):

    fold_run_manager = RunManager(run_name=os.path.join(run_manager.run_name, f"fold_{fold_idx+1}"))

    # 4.1) Create datasets for fold (optional: weights/fold_{k}_weights.json from compute_fold_weights.py)
    sw_name = f"fold_{fold_idx + 1}_weights.json"
    sw_path = os.path.join(weights_dir, sw_name)
    use_sw = sw_path if os.path.isfile(sw_path) else None
    if use_sw:
        print(f"Using sample weights: {use_sw}")
    train_fit_ds, val_monitor_ds, test_ds = create_tf_datasets_for_fold(
        fold_idx,
        config["img_size"],
        config["batch_size"],
        config["seed"],
        config["class_names"],
        config["output_dir"],
        sample_weights_path=use_sw,
        weights_base_dir=config["dataset_dir"],
    )
    print(f"Fold {fold_idx+1} datasets created successfully")

    # 4.2) Train model for fold
    model = build_model()
    callbacks = get_training_callbacks(fold_run_manager)
    history = train_model(
        model,
        train_fit_ds,
        val_monitor_ds,
        epochs=config["epochs_per_fold"],
        callbacks=callbacks,
        initial_epoch=0,
    )
    print(f"Fold {fold_idx+1} model trained successfully")

    # 4.2.1) Collect per-fold training time and persist to fold config
    fold_time = summarize_epoch_times(
        getattr(get_time_history(callbacks), "epoch_times", None)
    )
    fold_time["fold"] = fold_idx + 1
    fold_training_times.append(fold_time)
    fold_run_manager.save_config({**config, "training_time": fold_time})
    print(
        f"Fold {fold_idx+1} training time: total={fold_time['total_seconds']:.1f}s "
        f"| avg/epoch={fold_time['avg_seconds_per_epoch']:.1f}s"
    )

    # 4.3) Get best validation metrics across epochs.
    val_acc = float(np.nanmax(history.history.get("val_accuracy", [float("nan")])))
    val_auc = float(np.nanmax(history.history.get("val_auc", [float("nan")])))

    print(
        f"Fold {fold_idx + 1}: val_accuracy={val_acc:.4f} | val_auc={val_auc:.4f}"
    )

    if not np.isnan(val_acc):
        val_acc_per_fold.append(float(val_acc))
    if not np.isnan(val_auc):
        val_auc_per_fold.append(float(val_auc))

    # 4.4) Plot training graphs and save them
    print("?? Plotting training history...")
    history_plot_path = os.path.join(fold_run_manager.run_dir, "plots", "training_history.png")
    plot_history(history, save_path=history_plot_path)
    
    print("?? Plotting metrics...")
    metrics_plot_path = os.path.join(fold_run_manager.run_dir, "plots", "training_metrics.png")
    plot_metrics(history, save_path=metrics_plot_path)

    # 4.5) Evaluate on validation set
    print("?? Evaluating model on validation set...")
    evaluate_model(
        model,
        test_ds,
        plots_dir=os.path.join(fold_run_manager.run_dir, "plots"),
        ds_name="test",
    )
    print(f"Fold {fold_idx+1} model evaluated on validation set successfully")

    # 4.6) Save final model
    print("?? Saving final model...")
    model.save(os.path.join(fold_run_manager.run_dir, "models", f"fold_{fold_idx+1}.h5"))
    print(f"Fold {fold_idx+1} final model saved successfully")

print("?? All folds completed successfully!")

# 5) Save CV summary
print("?? Saving CV summary...")
total_seconds = float(sum(t["total_seconds"] for t in fold_training_times))
config["training_time"] = {
    "total_seconds": total_seconds,
    "avg_seconds_per_fold": total_seconds / len(fold_training_times) if fold_training_times else 0.0,
    "per_fold": fold_training_times,
}
run_manager.save_config(config)
save_cv_summary(run_manager.run_dir, config, val_acc_per_fold, val_auc_per_fold)