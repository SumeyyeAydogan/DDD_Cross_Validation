# run_manager.py - Simple run management for saving all epochs
import os
import json
import csv
from datetime import datetime

class RunManager:
    """Simple run manager for organizing training outputs"""
    
    def __init__(self, run_name="run_001"):
        self.run_name = run_name
        self.run_dir = self._create_run_directories()
        self.epoch_count = 0
        
    def _create_run_directories(self):
        """Create run directory structure"""
        base_dir = "runs"
        run_dir = os.path.join(base_dir, self.run_name)
        
        # Create directories
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        
        print(f"📁 Created run directories: {run_dir}")
        return run_dir
    
    def save_checkpoint(self, model, epoch):
        """Save model checkpoint for specific epoch"""
        checkpoint_path = os.path.join(self.run_dir, "checkpoints", f"epoch_{epoch:03d}.weights.h5")
        model.save_weights(checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
    def save_metrics(self, metrics_logs, epoch):
        """Save training metrics to CSV (one row per epoch)."""
        csv_path = os.path.join(self.run_dir, "training_metrics.csv")
        
        # Prepare metrics data
        metrics_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add all available metrics from callback logs (already scalar values)
        if metrics_logs is not None:
            for metric_name, metric_value in metrics_logs.items():
                metrics_data[metric_name] = metric_value
        
        # Save to CSV
        self._append_to_csv(csv_path, metrics_data)
        print(f"📊 Metrics saved for epoch {epoch}")
        
    def _append_to_csv(self, csv_path, data):
        """Append data to CSV file with stable metric columns."""
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(csv_path)
        fieldnames = [
            "epoch",
            "timestamp",
            "loss",
            "accuracy",
            "precision",
            "recall",
            "auc",
            "val_loss",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_auc",
            "lr",
        ]
        row = {key: data.get(key, "") for key in fieldnames}
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write headers if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write data
            writer.writerow(row)
    
    def save_final_model(self, model):
        """Save final trained model"""
        model_path = os.path.join(self.run_dir, "models", "final_model.h5")
        model.save(model_path)
        print(f"💾 Final model saved: {model_path}")
        
    def save_best_model(self, model):
        """Save best model (if you have one)"""
        model_path = os.path.join(self.run_dir, "models", "best_model.h5")
        model.save(model_path)
        print(f"💾 Best model saved: {model_path}")
        
    def save_config(self, config):
        """Save run configuration"""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"⚙️  Config saved: {config_path}")
        
    def get_checkpoint_path(self, epoch):
        """Get path for specific epoch checkpoint"""
        return os.path.join(self.run_dir, "checkpoints", f"epoch_{epoch:03d}.h5")
        
    def get_latest_checkpoint(self):
        """Get path to latest checkpoint"""
        checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return None
            
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("epoch_")]
        if not checkpoint_files:
            return None
            
        # Sort by epoch number and get latest
        latest = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
        return os.path.join(checkpoints_dir, latest)
    
    def load_latest_checkpoint(self, model):
        """Load latest checkpoint into model and return starting epoch"""
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint is None:
            print("❌ No checkpoint found, starting from scratch")
            return 0
            
        try:
            # Load weights
            model.load_weights(latest_checkpoint)
            
            # Extract epoch number from filename
            filename = os.path.basename(latest_checkpoint)
            epoch_num = int(filename.split('_')[1].split('.')[0])
            
            print(f"✅ Loaded checkpoint from epoch {epoch_num}: {latest_checkpoint}")
            return epoch_num
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return 0