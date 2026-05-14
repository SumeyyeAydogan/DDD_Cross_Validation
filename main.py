import os

# Before TensorFlow import: cuDNN / op determinism picks up these env vars at init.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from datetime import datetime
import tensorflow as tf

from src.fold_functions import create_tf_datasets_for_fold, save_fold_datasets
from src.model           import build_model
from src.train           import train_model
from src.utils           import plot_history, plot_metrics, plot_dataset_distribution
from src.evaluate        import evaluate_model
from src.gradcam_analysis import analyze_tf_keras_gradcam
from src.run_manager     import RunManager
from src.callbacks       import get_training_callbacks, get_time_history, summarize_epoch_times
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
    output_dir = os.path.join(project_root, "fold_datasets")
    os.makedirs(output_dir, exist_ok=True)
    #save_fold_datasets(raw_dir, 5, (224, 224), 42, class_names, output_dir)
    
    # 4) EXPERIMENT CONFIGURATION
    # ============================================================
    # Training uses the base pipeline and optional GradCAM weights.
    # ============================================================

    GRADCAM_WEIGHTS_FILE = os.path.join(project_root, "weights", "fold_1_weights.json")
    
    # Create run name based on configuration
    run_name = "reward_30-fold1_eagerly-false_pre-model_cv-dataloader_seed"
    
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
    fold_idx = 0
    weights_path = GRADCAM_WEIGHTS_FILE if os.path.isfile(GRADCAM_WEIGHTS_FILE) else None
    if weights_path:
        print(f"Using sample weights: {weights_path}")
    train_ds, val_ds = create_tf_datasets_for_fold(
        fold_idx,
        (224, 224),
        batch_size,
        42,
        class_names,
        output_dir,
        sample_weights_path=weights_path,
        weights_base_dir=raw_dir,
    )
    print(f"Fold {fold_idx + 1} datasets created successfully")
    
    # Print configuration
    print("\n" + "=" * 50)
    print("📋 EXPERIMENT CONFIGURATION:")
    print("=" * 50)
    sw_info = GRADCAM_WEIGHTS_FILE if weights_path else "(none — uniform train weights)"
    print(f"  Sample weights file:    {sw_info}")
    print("=" * 50 + "\n")
    
    print("✅ Datasets loaded successfully!")
    '''
    # Debug dataset shapes (support (x,y) and (x,y,w))
    def _print_batch_info(ds, name):
        for batch in ds.take(2):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x_batch, y_batch, w_batch = batch
                print(f"{name} x:", x_batch.shape, " y:", y_batch.shape, " w:", w_batch.shape)
                print(tf.reduce_mean(y_batch), tf.reduce_mean(w_batch))
            else:
                x_batch, y_batch = batch
                print(f"{name} x:", x_batch.shape, " y:", y_batch.shape)
                print(tf.reduce_mean(y_batch))

    _print_batch_info(train_ds, "train")
    '''
    # 5.1) Plot dataset distribution
    # print("?? Analyzing dataset distribution...")
    # dist_plot_path = os.path.join(run_manager.run_dir, "plots", "dataset_distribution.png")
    # # distribution plot
    # plot_dataset_distribution(output_dir, save_path=dist_plot_path)
    # print("? Dataset distribution analyzed and saved!")

    # 6) Build and train model (single GPU; no MirroredStrategy)
    print("???  Building model...")
    model = build_model()
    print("? Model built successfully!")

    # 6.1) Check for existing checkpoint and load if available
    print("?? Checking for existing checkpoints...")
    initial_epoch = run_manager.load_latest_checkpoint(model)

    if initial_epoch > 0:
        print(f"?? Resuming training from epoch {initial_epoch + 1}")
    else:
        print("?? Starting training from scratch")

    # 7) Save initial config
    epoch_count = 30
    config = {
        "run_name": run_manager.run_name,
        "epochs": epoch_count,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        "classes": list(class_names),
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "started_at": str(datetime.now()),
        "initial_epoch": initial_epoch,
    }
    run_manager.save_config(config)

    # 8) Training with all callbacks
    print("?? Starting training...")

    gradcam_epoch_outputs = os.path.join(run_manager.run_dir, "gradcam_epoch_outputs")
    gradcam_log_file = os.path.join(run_manager.run_dir, "gradcam_debug.log")
    sample_weight_log_file = os.path.join(run_manager.run_dir, "sample_weights_stats.json")
    callbacks = get_training_callbacks(
        run_manager,
        # val_ds,
        # gradcam_epoch_outputs,
        # max_samples=3,
        # gradcam_log_file=gradcam_log_file,
        # train_ds=train_ds,
        # monitor_sample_weights=True,
        # sample_weight_log_file=sample_weight_log_file
    )

    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=epoch_count,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )
    print("? Training completed!")

    # 8.1) Collect training time statistics from TimeHistory callback
    training_time = summarize_epoch_times(
        getattr(get_time_history(callbacks), "epoch_times", None)
    )
    print(
        f"?? Training time: total={training_time['total_seconds']:.1f}s "
        f"| avg/epoch={training_time['avg_seconds_per_epoch']:.1f}s "
        f"| epochs_completed={training_time['epochs_completed']}"
    )

    # 9) Plot training graphs and save them
    print("?? Plotting training history...")
    history_plot_path = os.path.join(run_manager.run_dir, "plots", "training_history.png")
    plot_history(history, save_path=history_plot_path)
    
    print("?? Plotting metrics...")
    metrics_plot_path = os.path.join(run_manager.run_dir, "plots", "training_metrics.png")
    plot_metrics(history, save_path=metrics_plot_path)

    # 10) Evaluate on validation set
    print("?? Evaluating model on validation set...")
    evaluate_model(
        model,
        val_ds,
        plots_dir=os.path.join(run_manager.run_dir, "plots"),
        ds_name="val",
    )
    print("? Validation evaluation completed!")

    # 11) Save final model
    print("?? Saving final model...")
    run_manager.save_final_model(model)
    
    # 12) Save simple config
    config = {
        "run_name": run_manager.run_name,
        "epochs": epoch_count,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        "classes": list(class_names),
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "training_time": training_time,
    }
    
    import json
    config_path = os.path.join(run_manager.run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"? Config saved to: {config_path}")
    
    print("\n" + "=" * 50)
    print("?? All tasks completed successfully!")
    print(f"?? Results saved to: {run_manager.run_dir}")
    print("Project finished! ??")