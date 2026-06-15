"""
GradCAM analysis functions for batch processing and subject-wise analysis.
"""

import os
import re
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from src.gradcam import CustomGradCAM

# Optional import for tf-keras-vis library
try:
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore
    TF_KERAS_VIS_AVAILABLE = True
except ImportError:
    TF_KERAS_VIS_AVAILABLE = False


def analyze_subjects_gradcam(
    model,
    test_dir,
    output_dir="gradcam_subjects",
    num_samples=20,
    class_names=('NotDrowsy', 'Drowsy'),
    img_size=(224, 224),
    seed=42,
    log_file=None,
    include_buckets=None,
    threshold=0.5,
):
    """Run subject-wise GradCAM analysis and save TP/TN/FP/FN examples."""
    rng = np.random.default_rng(seed)
    cam = CustomGradCAM(model, log_file=log_file)

    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        shuffle=False
    )
    file_paths = ds.file_paths

    ds = ds.apply(tf.data.experimental.ignore_errors())

    subj_re = re.compile(r"^([A-Za-z]+)")
    subj_to_samples = {}
    for path, (img, label) in zip(file_paths, ds.unbatch()):
        fname = os.path.basename(path)
        m = subj_re.match(fname)
        if not m:
            continue
        subj = m.group(1)
        subj_to_samples.setdefault(subj, []).append((path, int(label.numpy())))

    subjects = list(subj_to_samples.keys())
    rng.shuffle(subjects)
    cam._log(f"📂 Found {len(subjects)} subjects")

    buckets = ("TP", "TN", "FP", "FN")
    if include_buckets is not None:
        include_set = set(include_buckets)
        buckets = tuple([b for b in buckets if b in include_set])
    for sub in buckets:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    saved = 0
    sample_counter = {}  # Track sample count per bucket for unique filenames
    
    # Process all subjects - one sample per subject for diversity
    for subj in subjects:
        if saved >= num_samples:
            break
        # Try all samples of the subject in random order until we find a desired bucket
        samples = list(subj_to_samples[subj])
        rng.shuffle(samples)
        found_one = False  # Track if we found a valid sample from this subject
        for path, true_label in samples:
            if saved >= num_samples or found_one:
                break
            img = tf.keras.utils.load_img(path, target_size=img_size)
            img_arr = tf.keras.utils.img_to_array(img) / 255.0

            preds = model.predict(img_arr[None, ...], verbose=0)
            prob = float(preds[0][0])
            pred_cls = 1 if prob >= threshold else 0
            disp_prob = prob if pred_cls == 1 else (1.0 - prob)

            if true_label == 1 and pred_cls == 1:
                bucket = "TP"
            elif true_label == 0 and pred_cls == 0:
                bucket = "TN"
            elif true_label == 0 and pred_cls == 1:
                bucket = "FP"
            else:
                bucket = "FN"

            if include_buckets is None or bucket in include_buckets:
                # Create unique filename with counter
                sample_counter[bucket] = sample_counter.get(bucket, 0) + 1
                fname_base = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(output_dir, bucket, f"{fname_base}_{sample_counter[bucket]:03d}.png")
                cam.visualize(img_arr, class_names, true_idx=true_label, save_path=out_path, threshold=threshold)
                cam._log(f"🧍 {subj}: Truth={class_names[true_label]}, Pred={class_names[pred_cls]} "
                         f"({disp_prob:.2f}) -> {bucket}")
                saved += 1
                found_one = True  # Only one sample per subject for diversity

    cam._log(f"✅ GradCAM analysis completed (saved {saved} samples). Results: {output_dir}")


def analyze_tf_keras_gradcam(
    model,
    test_ds,
    output_dir="gradcam_tf_keras",
    num_samples=30,
    class_names=('NotDrowsy', 'Drowsy'),
    seed=42,
    log_file=None,
    include_buckets=None,
    threshold=0.5,
):
    """
    Run subject-wise GradCAM analysis using tf-keras-vis library.
    Saves TP/TN/FP/FN examples.
    
    Args:
        model: Trained Keras model
        test_ds: tf.data.Dataset for analysis
        output_dir: Directory to save visualizations
        num_samples: Number of samples to process
        class_names: Tuple of class names
        seed: Random seed for shuffling
        log_file: Optional log file path
        
    Raises:
        ImportError: If tf-keras-vis is not installed
    """
    if not TF_KERAS_VIS_AVAILABLE:
        raise ImportError(
            "tf-keras-vis is required for this function. Install with: pip install tf-keras-vis"
        )
    
    def _log(msg):
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}\n")
        print(msg)
    
    # Create output directories
    buckets = ("TP", "TN", "FP", "FN")
    if include_buckets is not None:
        include_set = set(include_buckets)
        buckets = tuple([b for b in buckets if b in include_set])
    for sub in buckets:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    
    # Prepare GradCAM (with linearized top for proper gradients)
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    
    # Find last Conv2D layer automatically
    penultimate_layer = None
    for idx, layer in reversed(list(enumerate(model.layers))):
        if isinstance(layer, tf.keras.layers.Conv2D):
            penultimate_layer = idx
            break
    
    if penultimate_layer is None:
        _log("❌ No Conv2D layer found in model.")
        return
    
    _log(f"📊 Using layer index {penultimate_layer} for GradCAM")
    
    # Collect all samples and shuffle for diversity
    all_samples = []
    for batch in test_ds:
        # Handle both formats: (x, y) or (x, y, sample_weight)
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            batch_images, batch_labels, _ = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch_images, batch_labels = batch
        else:
            raise ValueError(f"Unexpected batch structure: {type(batch)}")

        y_flat = np.ravel(batch_labels.numpy())
        for i in range(len(batch_images)):
            all_samples.append((batch_images[i].numpy(), int(y_flat[i])))
    
    random.seed(seed)
    random.shuffle(all_samples)
    _log(f"📂 Collected {len(all_samples)} samples from dataset")
    
    # Iterate shuffled samples and save GradCAM examples
    saved = 0
    for idx_global, (x, y_true) in enumerate(all_samples):
        if saved >= num_samples:
            break
        
        x_input = np.expand_dims(x, axis=0)
        preds = model.predict(x_input, verbose=0)
        
        # Binary sigmoid classification
        prob1 = float(preds[0][0])
        y_pred = 1 if prob1 >= threshold else 0
        prob_display = prob1 if y_pred == 1 else (1.0 - prob1)
        if y_pred == 1:
            score = lambda outputs: outputs[:, 0]
        else:
            score = lambda outputs: -outputs[:, 0]
        
        # Compute CAM
        cam = gradcam(score, x_input, penultimate_layer=penultimate_layer)
        heatmap = cam[0]
        heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        cmap = plt.get_cmap('jet')
        heatmap_color = cmap(heatmap_norm)
        heatmap_color = np.delete(heatmap_color, 3, 2)
        heatmap_color = (heatmap_color[..., :3] * 255).astype(np.uint8)
        
        orig_vis = x.copy()
        if orig_vis.max() <= 1.0:
            orig_vis = (orig_vis * 255).astype(np.uint8)
        else:
            orig_vis = orig_vis.astype(np.uint8)
        
        overlay = 0.6 * heatmap_color + 0.4 * orig_vis
        overlay = np.uint8(np.clip(overlay, 0, 255))
        
        # Decide bucket
        if y_true == 1 and y_pred == 1:
            bucket = "TP"
        elif y_true == 0 and y_pred == 0:
            bucket = "TN"
        elif y_true == 0 and y_pred == 1:
            bucket = "FP"
        else:
            bucket = "FN"
        
        if include_buckets is None or bucket in include_buckets:
            # Save combined figure with overlay text on Original
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig_vis)
            overlay_text = f"Truth: {class_names[y_true]} | Pred: {class_names[y_pred]} ({prob_display:.3f})"
            axes[0].text(
                5, 15, overlay_text,
                color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')
            )
            axes[0].set_title("Original")
            axes[0].axis('off')
            im1 = axes[1].imshow(heatmap_norm, cmap='jet')
            axes[1].set_title("Heatmap")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            plt.tight_layout()

            out_path = os.path.join(output_dir, bucket, f"example_{idx_global:04d}_p{y_pred}_t{y_true}.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            if saved < 5:  # Log first few samples
                _log(f"  Sample {saved}: Truth={class_names[y_true]}, Pred={class_names[y_pred]} "
                     f"({prob_display:.3f}) -> {bucket}")

            saved += 1
    
    if saved == 0:
        _log("⚠️  No samples were processed. Check the dataset or num_samples.")
    else:
        _log(f"✅ Saved {saved} examples under: {output_dir} (TP/TN/FP/FN)")

def analyze_gradcam_with_ds(
    model,
    ds,
    output_dir="gradcam_subjects",
    num_samples=20,
    class_names=("NotDrowsy", "Drowsy"),
    seed=42,
    log_file=None,
    include_buckets=None,
    threshold=0.5,
):
    import os
    import numpy as np
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    cam = CustomGradCAM(model, log_file=log_file)

    ds = ds.apply(tf.data.experimental.ignore_errors()).unbatch()

    buckets_all = ("TP", "TN", "FP", "FN")
    if include_buckets is None:
        buckets = buckets_all
    else:
        include_set = set(include_buckets)
        buckets = tuple(b for b in buckets_all if b in include_set)

    for b in buckets:
        os.makedirs(os.path.join(output_dir, b), exist_ok=True)

    saved = 0
    scanned = 0
    max_scan = max(num_samples * 20, 200)  # cap scan length

    for sample in ds:
        if saved >= num_samples or scanned >= max_scan:
            break
        scanned += 1

        if isinstance(sample, (tuple, list)) and len(sample) == 3:
            img, label, _ = sample
        else:
            img, label = sample

        img_np = img.numpy()
        true_label = int(np.squeeze(label.numpy()))

        pred = model.predict(img_np[None, ...], verbose=0)
        prob = float(pred.ravel()[0])
        pred_cls = 1 if prob >= threshold else 0

        if true_label == 1 and pred_cls == 1:
            bucket = "TP"
        elif true_label == 0 and pred_cls == 0:
            bucket = "TN"
        elif true_label == 0 and pred_cls == 1:
            bucket = "FP"
        else:
            bucket = "FN"

        if bucket not in buckets:
            continue

        out_path = os.path.join(output_dir, bucket, f"sample_{saved+1:03d}.png")
        cam.visualize(img_np, class_names=class_names, true_idx=true_label, save_path=out_path, threshold=threshold)
        cam._log(f"saved={out_path} true={class_names[true_label]} pred={class_names[pred_cls]} prob={prob:.4f}")
        saved += 1

    cam._log(f"GradCAM done. saved={saved}, scanned={scanned}, out={output_dir}")