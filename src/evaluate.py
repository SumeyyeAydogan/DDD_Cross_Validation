import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from src.utils import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, save_evaluation_report
from src.gradcam_analysis import (
    TF_KERAS_VIS_AVAILABLE,
    analyze_tf_keras_gradcam,
    analyze_gradcam_with_ds
)
import os
import tensorflow as tf

def evaluate_model(
    model,
    test_ds,
    plots_dir=None,
    class_names=['NotDrowsy', 'Drowsy'],
    misclassified_only=False,
    ds_name="test",
    num_gradcam_samples=10,
):
    """
    Evaluate model performance on test dataset
    """
    # 1) Collect all test predictions and labels
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for batch in test_ds:
        # Handle datasets that provide sample weights
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                x_batch, y_batch, _ = batch
            elif len(batch) == 2:
                x_batch, y_batch = batch
            else:
                raise ValueError(f"Unexpected batch structure length: {len(batch)}")
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        preds = model.predict(x_batch, verbose=0)
        
        # Check for NaN or Inf in predictions
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            print(f"⚠️ WARNING: NaN or Inf detected in model predictions!")
            print(f"   NaN count: {np.sum(np.isnan(preds))}, Inf count: {np.sum(np.isinf(preds))}")
            # Clip invalid values to valid range [0, 1]
            preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
            preds = np.clip(preds, 0.0, 1.0)
        
        # For binary classification: y_batch is already 0 or 1
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend((preds > 0.5).astype(int).flatten())  # Threshold 0.5
        y_pred_proba.extend(preds.flatten())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Final validation: check for NaN/Inf in collected arrays
    nan_mask = np.isnan(y_pred_proba) | np.isinf(y_pred_proba)
    if np.any(nan_mask):
        print(f"⚠️ WARNING: Found {np.sum(nan_mask)} NaN/Inf values in predictions after collection")
        print(f"   Removing invalid predictions for evaluation...")
        valid_mask = ~nan_mask
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        y_pred_proba = y_pred_proba[valid_mask]
        # Replace any remaining invalid values
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
        y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)
    
    # Check if we have enough samples and both classes
    if len(y_true) == 0:
        raise ValueError("No valid predictions collected! Check your dataset and model.")
    
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print(f"⚠️ WARNING: Only one class found in labels: {unique_classes}")
        print(f"   Cannot compute ROC AUC. Using default value 0.5")
        roc_auc = 0.5
    else:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            if np.isnan(roc_auc) or np.isinf(roc_auc):
                print(f"⚠️ WARNING: ROC AUC is NaN/Inf. Using default value 0.5")
                roc_auc = 0.5
        except ValueError as e:
            print(f"⚠️ WARNING: Error computing ROC AUC: {e}")
            print(f"   Using default value 0.5")
            roc_auc = 0.5

    # 2) Print classification report
    print("Classification Report:")
    print("=" * 50)
    report = classification_report(y_true, y_pred, target_names=class_names)
    er_save_path = os.path.join(plots_dir, f"{ds_name}_evaluation_report.txt") if plots_dir else None
    print(report)
    eval_results = model.evaluate(test_ds, verbose=0)
    test_loss = eval_results[0]
    test_accuracy = eval_results[1]
    print(f"REAL Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    save_evaluation_report(report, roc_auc, test_accuracy, test_loss, save_path=er_save_path)

    
    # 3) Plot confusion matrix
    print("Plotting Confusion Matrix...")
    cm_save_path = os.path.join(plots_dir, f"{ds_name}_confusion_matrix.png") if plots_dir else None
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_save_path)
    
    # 4) Plot ROC curve
    print("Plotting ROC Curve...")
    roc_save_path = os.path.join(plots_dir, f"{ds_name}_roc_curve.png") if plots_dir else None
    plot_roc_curve(y_true, y_pred_proba, save_path=roc_save_path)
    
    # 5) Plot Precision-Recall curve
    print("Plotting Precision-Recall Curve...")
    pr_save_path = os.path.join(plots_dir, f"{ds_name}_precision_recall_curve.png") if plots_dir else None
    plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_save_path)
    
    # 6) Generate GradCAM visualizations for explainability
    """ try:
        os.environ["TF_DETERMINISTIC_OPS"] = "0"
    except Exception as e:
        print(f"⚠️ WARNING: Error disabling op determinism: {e}")
        print("   Continuing with GradCAM analysis...") """

    print("Generating GradCAM visualizations...")
    gradcam_dir = os.path.join(plots_dir, f"{ds_name}_gradcam") if plots_dir else f"{ds_name}_gradcam_results"
    os.makedirs(gradcam_dir, exist_ok=True)
    buckets = ("FP", "FN") if misclassified_only else None
    if TF_KERAS_VIS_AVAILABLE:
        print("[GradCAM] using tf-keras-vis on val/test dataset.")
        try:
            analyze_tf_keras_gradcam(
                model,
                test_ds,
                num_samples=num_gradcam_samples,
                output_dir=gradcam_dir,
                class_names=tuple(class_names),
                include_buckets=buckets,
            )
        except Exception as e:
            print(f"[GradCAM] GPU path failed: {e}")
            print("[GradCAM] Retrying on CPU...")
            with tf.device("/CPU:0"):
                analyze_tf_keras_gradcam(
                    model,
                    test_ds,
                    num_samples=num_gradcam_samples,
                    output_dir=gradcam_dir,
                    class_names=tuple(class_names),
                    include_buckets=buckets,
                )
    else:
        print(
            "[GradCAM] Using analyze_gradcam_with_ds function."
        )
        analyze_gradcam_with_ds(
            model=model,
            ds=test_ds,
            output_dir=gradcam_dir,
            num_samples=num_gradcam_samples,
            class_names=tuple(class_names),
            seed=42,
            log_file=None,
            include_buckets=buckets,
        )
    
    # 7) Return metrics for further analysis
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }