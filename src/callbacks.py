# callbacks.py - Custom training callbacks with GradCAM visualizations
import tensorflow as tf
import os
import random
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from src.gradcam import CustomGradCAM

# ------------------------------
# 1. Checkpoint callback
# ------------------------------
class CheckpointCallback(tf.keras.callbacks.Callback):
    """Custom callback to save checkpoints and metrics after each epoch"""
    
    def __init__(self, run_manager):
        super().__init__()
        self.run_manager = run_manager
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        epoch_num = epoch + 1  # Keras epochs are 0-indexed
        
        # Save checkpoint
        self.run_manager.save_checkpoint(self.model, epoch_num)
        
        # Save metrics from Keras epoch logs
        self.run_manager.save_metrics(logs, epoch_num)
        
        print(f"✅ Epoch {epoch_num} completed and checkpoint/metrics saved!")

# ------------------------------
# 2. GradCAM visualization callback
# ------------------------------
class GradCAMEpochCallback(tf.keras.callbacks.Callback):
    """
    Callback to save GradCAM visualizations for validation dataset at the end of each epoch.
    Visualizations are saved into TP/TN/FP/FN folders.
    """
    def __init__(self, test_ds, output_dir="gradcam_epoch_outputs", max_samples=10, log_file=None):
        """
        Args:
            test_ds: tf.data.Dataset for GradCAM visualization (validation set recommended)
            output_dir: directory to save GradCAM images
            max_samples: max number of samples per epoch to save
            log_file: optional path to save debug logs
        """
        super().__init__()
        self.test_ds = test_ds
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.log_file = log_file
        os.makedirs(output_dir, exist_ok=True)
        #self.gradcam = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        print(f"\n[GradCAM] Saving visualizations for epoch {epoch_num}...")
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch_num:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Create TP/TN/FP/FN folders
        tp_dir = os.path.join(epoch_dir, "TP")
        tn_dir = os.path.join(epoch_dir, "TN")
        fp_dir = os.path.join(epoch_dir, "FP")
        fn_dir = os.path.join(epoch_dir, "FN")
        for d in [tp_dir, tn_dir, fp_dir, fn_dir]:
            os.makedirs(d, exist_ok=True)

        # Always create new GradCAM instance for each epoch to get epoch-specific logs
        # Use epoch-specific log file if provided
        if self.log_file:
            logs_root = os.path.join(os.path.dirname(self.log_file), "gradcam_logs")
            os.makedirs(logs_root, exist_ok=True)
            log_path = os.path.join(logs_root, f"epoch_{epoch_num:03d}.log")
        else:
            log_path = None
        
        # Create new gradcam instance for this epoch (log every sample)
        gradcam = CustomGradCAM(self.model, log_file=log_path, debug_every=1)
        print(f"[GradCAM Callback] Created GradCAM for epoch {epoch_num} (log: {log_path})")

        # 1️⃣ Collect all samples from dataset (support (x,y) and (x,y,w))
        all_samples = []
        for batch in self.test_ds:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                batch_images, batch_labels, _ = batch
            else:
                batch_images, batch_labels = batch
            for i in range(len(batch_images)):
                all_samples.append((batch_images[i], batch_labels[i]))
        
        # 2️⃣ Random shuffle for diversity
        random.seed(epoch_num * 42)  # Different seed per epoch but reproducible
        random.shuffle(all_samples)
        
        # 3️⃣ Select random samples
        sample_count = 0
        for image, label in all_samples:
            if sample_count >= self.max_samples:
                break
                
            image_np = image.numpy()
            # Determine true class index (binary mode: single float value)
            # Label is already 0.0 or 1.0 in binary mode
            true_idx = int(label.numpy())

            # Model prediction
            pred_vec = self.model.predict(image_np[None, ...], verbose=0)
            # Model outputs sigmoid probability [0.0-1.0]
            pred_prob = float(pred_vec.ravel()[0])
            pred_idx = 1 if pred_prob >= 0.5 else 0

            # Select folder based on TP/TN/FP/FN
            if true_idx == 1 and pred_idx == 1:
                folder = tp_dir
                status = "TP"
            elif true_idx == 0 and pred_idx == 0:
                folder = tn_dir
                status = "TN"
            elif true_idx == 0 and pred_idx == 1:
                folder = fp_dir
                status = "FP"
            elif true_idx == 1 and pred_idx == 0:
                folder = fn_dir
                status = "FN"

            # File path
            filename = f"sample_{sample_count:02d}_true{true_idx}_pred{pred_idx}.png"
            save_path = os.path.join(folder, filename)
            
            # Debug print (console)
            if sample_count < 3:  # Print first 3 samples
                print(f"  Sample {sample_count}: True={true_idx}, Pred={pred_idx} (prob={pred_prob:.3f}) -> {status}")

            # Save GradCAM visualization
            gradcam.visualize(image_np, save_path=save_path, true_idx=true_idx)

            # Write a simple per-sample line into the epoch log (if logging is enabled)
            if log_path is not None:
                # Use GradCAM internal logger to append
                gradcam._log(f"[{status}] {filename} prob={pred_prob:.3f} true={true_idx} pred={pred_idx}")

            sample_count += 1

        print(f"[GradCAM] Saved {sample_count} GradCAM samples for epoch {epoch_num}")

# ------------------------------
# 3. Sample Weight Monitoring Callback
# ------------------------------
class SampleWeightMonitorCallback(tf.keras.callbacks.Callback):
    """
    Callback to monitor and log sample_weights statistics during training.
    Helps verify that sample_weights are being used correctly.
    """
    def __init__(self, train_ds, log_file=None, log_every_n_epochs=1):
        """
        Args:
            train_ds: Training dataset (should have sample_weights)
            log_file: Optional file path to save statistics
            log_every_n_epochs: Log statistics every N epochs (default: every epoch)
        """
        super().__init__()
        self.train_ds = train_ds
        self.log_file = log_file
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_stats = []
        
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Log sample_weight statistics at the end of each epoch"""
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return
        
        epoch_num = epoch + 1
        all_weights = []
        all_labels = []
        
        # Collect sample_weights from a few batches
        batch_count = 0
        for batch in self.train_ds.take(5):  # Check first 5 batches
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                _, y_batch, w_batch = batch
                weights_np = w_batch.numpy()
                labels_np = y_batch.numpy().flatten()
                all_weights.extend(weights_np)
                all_labels.extend(labels_np)
                batch_count += 1
        
        if len(all_weights) == 0:
            print(f"[SampleWeight] Epoch {epoch_num}: No sample_weights found in dataset!")
            return
        
        # Calculate statistics
        all_weights = np.array(all_weights)
        all_labels = np.array(all_labels)
        
        stats = {
            'epoch': epoch_num,
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights)),
            'median': float(np.median(all_weights)),
            'drowsy_mean': float(np.mean(all_weights[all_labels == 1])) if np.sum(all_labels == 1) > 0 else 0.0,
            'notdrowsy_mean': float(np.mean(all_weights[all_labels == 0])) if np.sum(all_labels == 0) > 0 else 0.0,
            'samples_checked': len(all_weights)
        }
        self.epoch_stats.append(stats)
        
        # Print to console
        print(f"\n[SampleWeight] Epoch {epoch_num} Statistics:")
        print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  By class - Drowsy: {stats['drowsy_mean']:.4f}, NotDrowsy: {stats['notdrowsy_mean']:.4f}")
        print(f"  Samples checked: {stats['samples_checked']}")
        
        # Write to file if specified
        if self.log_file:
            import json
            with open(self.log_file, 'w') as f:
                json.dump(self.epoch_stats, f, indent=2)

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self._t0 = None
    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._t0
        self.epoch_times.append(dt)
        print(f"[time] epoch {epoch+1}: {dt:.1f}s | avg so far: {sum(self.epoch_times)/len(self.epoch_times):.1f}s")
# ------------------------------
# 4. Function to get all training callbacks
# ------------------------------
def get_training_callbacks(run_manager, val_ds=None, gradcam_output_dir="gradcam_epoch_outputs", max_samples=5, gradcam_log_file=None, 
                          train_ds=None, monitor_sample_weights=False, sample_weight_log_file=None):
    """
    Returns all training callbacks including checkpoint, early stopping,
    learning rate scheduler, and optional GradCAM visualizations.

    Args:
        run_manager: RunManager instance for checkpoint/metrics
        val_ds: tf.data.Dataset for GradCAM visualization (validation set recommended)
        gradcam_output_dir: folder to save GradCAM outputs
        max_samples: max number of samples per epoch to save
        gradcam_log_file: optional path to save GradCAM debug logs
        train_ds: Training dataset (required if monitor_sample_weights=True)
        monitor_sample_weights: Whether to monitor sample_weights during training
        sample_weight_log_file: Path to save sample_weight statistics (optional)

    Returns:
        list of callbacks
    """
    callbacks = [
        CheckpointCallback(run_manager),
        EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, min_lr=1e-6),
        TimeHistory(),
    ]

    # Add sample weight monitoring if requested
    if monitor_sample_weights and train_ds is not None:
        callbacks.append(
            SampleWeightMonitorCallback(train_ds, log_file=sample_weight_log_file)
        )
    
    if val_ds is not None:
        callbacks.append(
            GradCAMEpochCallback(test_ds=val_ds, output_dir=gradcam_output_dir, 
                               max_samples=max_samples, log_file=gradcam_log_file)
        )

    return callbacks


def get_time_history(callbacks):
    """Return the TimeHistory callback from a list (or None if absent)."""
    for cb in callbacks or []:
        if isinstance(cb, TimeHistory):
            return cb
    return None


def summarize_epoch_times(epoch_times):
    """Build a JSON-serializable summary of per-epoch training times."""
    if not epoch_times:
        return {
            "epochs_completed": 0,
            "total_seconds": 0.0,
            "avg_seconds_per_epoch": 0.0,
            "min_seconds_per_epoch": 0.0,
            "max_seconds_per_epoch": 0.0,
            "epoch_seconds": [],
        }
    times = [float(t) for t in epoch_times]
    return {
        "epochs_completed": len(times),
        "total_seconds": float(sum(times)),
        "avg_seconds_per_epoch": float(sum(times) / len(times)),
        "min_seconds_per_epoch": float(min(times)),
        "max_seconds_per_epoch": float(max(times)),
        "epoch_seconds": times,
    }
