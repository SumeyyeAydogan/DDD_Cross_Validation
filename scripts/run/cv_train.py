"""
Configurable CV training (same logic as cv_main.py, CLI-driven).

Example::

    python scripts/run/cv_train.py --run-name reward_percentile_dark --weights-dir weights_percentile
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from _bootstrap import bootstrap

project_root, _ = bootstrap()

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

from src.callbacks import get_time_history, get_training_callbacks, summarize_epoch_times
from src.evaluate import evaluate_model
from src.threshold import fit_threshold_for_fold, save_threshold
from src.fold_functions import create_tf_datasets_for_fold
from src.model import build_model
from src.run_manager import RunManager
from src.train import train_model
from src.utils import plot_history, plot_metrics, save_cv_summary, metrics_at_best_val_auc


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV training with optional sample weights.")
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--weights-dir", type=str, default=None, help="Dir with fold_k_weights.json; omit for no weights")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--fold-datasets-dir", type=str, default="fold_datasets_v2")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or str(project_root / "dataset")
    output_dir = (
        Path(args.fold_datasets_dir)
        if Path(args.fold_datasets_dir).is_absolute()
        else project_root / args.fold_datasets_dir
    )
    weights_dir = None
    if args.weights_dir:
        p = Path(args.weights_dir)
        weights_dir = str(p if p.is_absolute() else project_root / p)

    fold_end = args.fold_end if args.fold_end is not None else args.fold_start + args.k - 1

    run_manager = RunManager(run_name=args.run_name)
    config = {
        "dataset_dir": dataset_dir,
        "class_names": ("NotDrowsy", "Drowsy"),
        "run_name": args.run_name,
        "img_size": (224, 224),
        "k": args.k,
        "seed": args.seed,
        "epochs_per_fold": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 1e-4,
        "started_at": str(datetime.now()),
        "output_dir": str(output_dir),
        "weights_dir": weights_dir,
    }
    run_manager.save_config(config)
    _set_global_determinism(config["seed"])

    val_acc_per_fold: List[float] = []
    val_auc_per_fold: List[float] = []
    best_epoch_per_fold: List[int] = []
    test_acc_per_fold: List[float] = []
    test_auc_per_fold: List[float] = []
    test_loss_per_fold: List[float] = []
    threshold_per_fold: List[float] = []
    fold_training_times: List[dict] = []

    for fold_idx in range(args.fold_start - 1, fold_end):
        fold_run_manager = RunManager(run_name=os.path.join(run_manager.run_name, f"fold_{fold_idx + 1}"))
        sw_path = None
        if weights_dir:
            candidate = os.path.join(weights_dir, f"fold_{fold_idx + 1}_weights.json")
            if os.path.isfile(candidate):
                sw_path = candidate
                print(f"Using sample weights: {sw_path}")

        train_fit_ds, val_monitor_ds, test_ds = create_tf_datasets_for_fold(
            fold_idx,
            config["img_size"],
            config["batch_size"],
            config["seed"],
            config["class_names"],
            config["output_dir"],
            sample_weights_path=sw_path,
            weights_base_dir=config["dataset_dir"],
        )

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

        fold_time = summarize_epoch_times(getattr(get_time_history(callbacks), "epoch_times", None))
        fold_time["fold"] = fold_idx + 1
        fold_training_times.append(fold_time)
        fold_run_manager.save_config({**config, "training_time": fold_time})

        plots_dir = os.path.join(fold_run_manager.run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_history(history.history, save_path=os.path.join(plots_dir, "training_history.png"))
        plot_metrics(history.history, save_path=os.path.join(plots_dir, "training_metrics.png"))

        th_path = os.path.join(fold_run_manager.run_dir, "threshold.json")
        threshold, th_source = fit_threshold_for_fold(
            model, fold_idx, config["img_size"], config["batch_size"], config["seed"], config["output_dir"]
        )
        save_threshold(th_path, threshold, source=th_source)
        threshold_per_fold.append(threshold)
        print(f"Fold {fold_idx + 1} threshold (train_fit): {threshold:.4f}")

        val_acc, val_auc, best_epoch = metrics_at_best_val_auc(history)
        val_acc_per_fold.append(val_acc)
        val_auc_per_fold.append(val_auc)
        best_epoch_per_fold.append(best_epoch)

        test_metrics = evaluate_model(
            model,
            test_ds,
            plots_dir=plots_dir,
            class_names=list(config["class_names"]),
            ds_name="test",
            threshold=threshold,
        )
        test_acc_per_fold.append(float(test_metrics["accuracy"]))
        test_auc_per_fold.append(float(test_metrics["roc_auc"]))
        test_loss_per_fold.append(float(test_metrics["log_loss"]))

        print(
            f"Fold {fold_idx + 1} done. "
            f"val@best_auc_epoch={best_epoch + 1} acc={val_acc:.4f} auc={val_auc:.4f} | "
            f"test acc={test_metrics['accuracy']:.4f} auc={test_metrics['roc_auc']:.4f}"
        )

    save_cv_summary(
        run_manager.run_dir,
        config,
        val_acc_per_fold,
        val_auc_per_fold,
        test_acc_per_fold=test_acc_per_fold,
        test_auc_per_fold=test_auc_per_fold,
        test_loss_per_fold=test_loss_per_fold,
        threshold_per_fold=threshold_per_fold,
        best_epoch_per_fold=best_epoch_per_fold,
    )
    run_manager.save_config({**config, "training_time": {"per_fold": fold_training_times}})
    print(f"CV complete. Run dir: {run_manager.run_dir}")


if __name__ == "__main__":
    main()
