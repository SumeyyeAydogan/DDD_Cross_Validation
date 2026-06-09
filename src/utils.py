import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import os


def _load_history_from_csv(save_path):
    if not save_path:
        return {}
    run_dir = os.path.dirname(os.path.dirname(save_path))
    csv_path = os.path.join(run_dir, "training_metrics.csv")
    if not os.path.exists(csv_path):
        return {}
    hist = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key in ("epoch", "timestamp") or value in (None, ""):
                    continue
                try:
                    hist.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return hist

def create_run_directories(run_name="run_001"):
    """Create basic run directory structure"""
    base_dir = "runs"
    run_dir = os.path.join(base_dir, run_name)
    
    # Create directories
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    
    print(f"📁 Created run directories: {run_dir}")
    return run_dir

def plot_dataset_distribution(data_dir="data", save_path=None):
    """Plot distribution of drowsy vs notdrowsy in train/val/test datasets"""
    import glob
    
    datasets = ['train', 'val', 'test']
    drowsy_counts = []
    notdrowsy_counts = []
    
    for dataset in datasets:
        dataset_path = os.path.join(data_dir, dataset)
        if not os.path.exists(dataset_path):
            drowsy_counts.append(0)
            notdrowsy_counts.append(0)
            continue

        # Support two layouts:
        # 1) Old: files named with *_1.* (drowsy) and *_0.* (not drowsy)
        # 2) New: class subfolders {NotDrowsy, Drowsy} under each split
        old_style_drowsy = glob.glob(os.path.join(dataset_path, "*_1.*"))
        old_style_notdrowsy = glob.glob(os.path.join(dataset_path, "*_0.*"))

        notdrowsy_dir = os.path.join(dataset_path, 'NotDrowsy')
        drowsy_dir = os.path.join(dataset_path, 'Drowsy')

        if (len(old_style_drowsy) + len(old_style_notdrowsy)) > 0:
            drowsy_counts.append(len(old_style_drowsy))
            notdrowsy_counts.append(len(old_style_notdrowsy))
        elif os.path.isdir(notdrowsy_dir) and os.path.isdir(drowsy_dir):
            nd_count = sum(
                1 for f in os.listdir(notdrowsy_dir)
                if os.path.isfile(os.path.join(notdrowsy_dir, f))
            )
            d_count = sum(
                1 for f in os.listdir(drowsy_dir)
                if os.path.isfile(os.path.join(drowsy_dir, f))
            )
            notdrowsy_counts.append(nd_count)
            drowsy_counts.append(d_count)
        else:
            # Unknown layout; count as zero to avoid crashes
            drowsy_counts.append(0)
            notdrowsy_counts.append(0)
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    x = np.arange(len(datasets))
    width = 0.35
    
    ax1.bar(x - width/2, drowsy_counts, width, label='Drowsy', color='red')
    ax1.bar(x + width/2, notdrowsy_counts, width, label='Not Drowsy', color='blue')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Distribution: Drowsy vs Not Drowsy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True)
    
    # Add value labels on bars
    for i, (d, nd) in enumerate(zip(drowsy_counts, notdrowsy_counts)):
        ax1.text(i - width/2, d + max(drowsy_counts + notdrowsy_counts) * 0.01, str(d), 
                ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, nd + max(drowsy_counts + notdrowsy_counts) * 0.01, str(nd), 
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart for total distribution
    total_drowsy = sum(drowsy_counts)
    total_notdrowsy = sum(notdrowsy_counts)
    
    if (total_drowsy + total_notdrowsy) > 0:
        ax2.pie([total_drowsy, total_notdrowsy], 
                labels=[f'Drowsy ({total_drowsy})', f'Not Drowsy ({total_notdrowsy})'],
                autopct='%1.1f%%', startangle=90, colors=['red', 'blue'])
        ax2.set_title('Total Dataset Distribution')
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax2.set_title('Total Dataset Distribution')
        ax2.axis('off')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Dataset distribution plot saved: {save_path}")
    
    plt.show()
    plt.close()
    
    # Print summary
    print("\n📊 Dataset Distribution Summary:")
    print("=" * 40)
    for i, dataset in enumerate(datasets):
        total = drowsy_counts[i] + notdrowsy_counts[i]
        if total > 0:
            drowsy_pct = (drowsy_counts[i] / total) * 100
            notdrowsy_pct = (notdrowsy_counts[i] / total) * 100
            print(f"{dataset.capitalize():>8}: {drowsy_counts[i]:>4} drowsy ({drowsy_pct:>5.1f}%) | {notdrowsy_counts[i]:>4} not drowsy ({notdrowsy_pct:>5.1f}%) | Total: {total}")
        else:
            print(f"{dataset.capitalize():>8}: No data found")
    
    print(f"\nTotal Images: {total_drowsy + total_notdrowsy}")
    print(f"Overall Drowsy Ratio: {(total_drowsy / (total_drowsy + total_notdrowsy)) * 100:.1f}%")

def plot_history(history, save_path=None):
    """Plot training history: accuracy and loss"""
    history_dict = getattr(history, "history", {}) if history is not None else {}
    if "accuracy" not in history_dict or "loss" not in history_dict:
        csv_hist = _load_history_from_csv(save_path)
        if "accuracy" in csv_hist and "loss" in csv_hist:
            history_dict = csv_hist
        else:
            print("No plottable training history found (history object or CSV).")
            return

    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Train')
    if 'val_accuracy' in history_dict:
        plt.plot(history_dict['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Train')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()
    plt.close()

def plot_metrics(history, save_path=None):
    """Plot additional metrics: precision, recall, auc"""
    history_dict = getattr(history, "history", {}) if history is not None else {}
    if not history_dict:
        history_dict = _load_history_from_csv(save_path)

    metrics = ['precision', 'recall', 'auc']
    available_metrics = [m for m in metrics if m in history_dict]
    
    if not available_metrics:
        print("No additional metrics found in history")
        return
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(available_metrics):
        plt.subplot(1, len(available_metrics), i+1)
        plt.plot(history_dict[metric], label=f'Train {metric}')
        if f'val_{metric}' in history_dict:
            plt.plot(history_dict[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f'{metric.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()
    plt.close()

def save_evaluation_report(report, roc_auc, test_accuracy, test_loss, save_path):
    with open(save_path, "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n\n=== ROC–AUC Score ===\n")
        f.write(str(roc_auc))
        f.write(f" Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

def save_cv_summary(run_dir, config, val_acc_per_fold, val_auc_per_fold):
    cv_summary_path = os.path.join(run_dir, "cv_summary.txt")
    val_acc_mean = float(np.mean(val_acc_per_fold)) if val_acc_per_fold else float("nan")
    val_acc_std = float(np.std(val_acc_per_fold)) if val_acc_per_fold else float("nan")
    val_auc_mean = float(np.mean(val_auc_per_fold)) if val_auc_per_fold else float("nan")
    val_auc_std = float(np.std(val_auc_per_fold)) if val_auc_per_fold else float("nan")
    with open(cv_summary_path, "w", encoding="utf-8") as f:
        f.write("Cross-validation summary\n")
        f.write(f"Folds           : {config['k']}\n")
        f.write(f"Epochs/fold     : {config['epochs_per_fold']}\n")
        f.write(f"Base dir        : {config['dataset_dir']}\n\n")
        f.write(f"val_accuracy_mean = {val_acc_mean:.4f}\n")
        f.write(f"val_accuracy_std  = {val_acc_std:.4f}\n")
        f.write(f"val_auc_mean      = {val_auc_mean:.4f}\n")
        f.write(f"val_auc_std       = {val_auc_std:.4f}\n")


def plot_confusion_matrix(y_true, y_pred, class_names=['NotDrowsy', 'Drowsy'], save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()
    plt.close()
def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()
    plt.close()
def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()
    plt.close()