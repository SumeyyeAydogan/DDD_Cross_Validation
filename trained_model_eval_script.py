import os

from src.cv_dataloader import make_tf_dataset_from_paths

# Before TensorFlow import: cuDNN / op determinism picks up these env vars at init.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from datetime import datetime
import tensorflow as tf

from src.fold_functions import create_tf_datasets_for_fold, load_fold_manifest
from src.model           import build_model
from src.evaluate        import evaluate_model
from src.evaluation.threshold     import fit_threshold_on_dataset, save_threshold
from src.visualization.threshold_plots import plot_threshold_results
from src.visualization.prob_test_and_histogram import test_threshold
from src.gradcam_analysis import analyze_tf_keras_gradcam
from src.run_manager     import RunManager
import random
import numpy as np

def _set_global_determinism(seed: int) -> None:
    """Set Python/NumPy/TensorFlow seeds (TF determinism env vars set before tf import)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


if __name__ == "__main__":
    print("?? Starting Drowsy Driver Detection Project...")
    print("=" * 50)
    _set_global_determinism(42)
    # Project root directory: the folder where this file is located
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    #project_root = r"D:\internship\Drowsy-Driver-Detection-Project"

    # 1) Raw data folder (what you have)
    raw_dir = os.path.join(project_root, "dataset")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"`dataset` not found: {raw_dir}")

    # 2) Folder where split data will go
    class_names = ("NotDrowsy", "Drowsy")
    output_dir = os.path.join(project_root, "fold_datasets_v2")
    os.makedirs(output_dir, exist_ok=True)
    #save_fold_datasets(raw_dir, 5, (224, 224), 42, class_names, output_dir, val_ratio=0.15)
    #split_method="stratified_round_robin"
    
    # 4) EXPERIMENT CONFIGURATION
    
    # Create run name based on configuration
    run_name = "baseline"
    
    # 5) Create run manager
    print("📁 Creating run manager...")
    run_manager = RunManager(run_name)
    print(f"✅ Run manager created: {run_manager.run_dir}")
    print(tf.__version__); print(tf.config.list_physical_devices('GPU'))


    # 6) tf.data pipelines
    # LOADER (binary: NotDrowsy=0, Drowsy=1)
    print("🔄 Loading datasets...")
    
    # 4.1) Create datasets for fold (fold_idx is 0-based → 0 == fold_1.json)
    batch_size = 32
    fold_idx = 4
    train_fit_ds, val_monitor_ds, test_ds = create_tf_datasets_for_fold(
        fold_idx,
        (224, 224),
        batch_size,
        42,
        class_names,
        output_dir,
        sample_weights_path=None,
        weights_base_dir=raw_dir,
    )
    print(f"Fold {fold_idx + 1} datasets created successfully")
    
    print("✅ Datasets loaded successfully!")

    # 6) Build and train model (single GPU; no MirroredStrategy)
    print("???  Building model...")
    from src.model import build_model

    #model = build_model()

    model = tf.keras.models.load_model(
        "/SPACE/spin02/DDD/DDD_cross_validation/runs/baseline/fold_5/models/fold_5.h5",
        compile=False
    )

    print(model)
    model.summary()
    print("? Model built successfully!")

    evaluation_experiment_dir = os.path.join(run_manager.run_dir, "plots", f"evaluation_experiment_{fold_idx+1}")
    os.makedirs(evaluation_experiment_dir, exist_ok=True)

    threshold_analysis_dir = os.path.join(run_manager.run_dir, "threshold_analysis")
    os.makedirs(threshold_analysis_dir, exist_ok=True)

    # 10) Threshold on val_monitor_ds, evaluate on test
    th_path = os.path.join(threshold_analysis_dir, "threshold.json")
    threshold, ttc = fit_threshold_on_dataset(
        model, val_monitor_ds, scoring="balanced_accuracy"
    )
    save_threshold(th_path, threshold, scoring="balanced_accuracy")
    plot_threshold_results(ttc, save_path=os.path.join(threshold_analysis_dir, "threshold_tuning.png"))
    print(f"Fold {fold_idx+1} threshold (val_monitor, balanced_accuracy): {threshold:.4f}")

    print("?? Evaluating model on train set...")
    data = load_fold_manifest(fold_idx, output_dir)
    train_files = data["train_fit"]["files"]
    train_labels = np.asarray(data["train_fit"]["labels"], dtype=np.float32)
    train_fit_eval_ds = make_tf_dataset_from_paths(
        train_files, train_labels, (224, 224), batch_size,
        augment=False, seed=42, sample_weights=None,
    )
    evaluate_model(
        model,
        train_fit_eval_ds,
        plots_dir=evaluation_experiment_dir,
        class_names=list(class_names),
        ds_name="train_fit",
        threshold=threshold,
    )
    print("? Train evaluation completed!")

    print("?? Evaluating model on val set...")
    evaluate_model(
        model,
        val_monitor_ds,
        plots_dir=evaluation_experiment_dir,
        class_names=list(class_names),
        ds_name="val_monitor",
        threshold=threshold,
    )
    print("? Val evaluation completed!")
    print("?? Evaluating model on test set...")
    evaluate_model(
        model,
        test_ds,
        plots_dir=evaluation_experiment_dir,
        class_names=list(class_names),
        ds_name="test",
        threshold=threshold,
    )
    print("? Test evaluation completed!")

    test_threshold(
        model,
        val_monitor_ds,
        test_ds,
        threshold,
        output_dir=threshold_analysis_dir
    )
    
    print("\n" + "=" * 50)
    print("?? All tasks completed successfully!")
    print(f"?? Results saved to: {threshold_analysis_dir}")
    print("Project finished! ??")