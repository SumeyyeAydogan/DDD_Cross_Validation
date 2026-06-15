"""
Simplified GradCAM for binary (sigmoid) models with Conv2D layers only.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime


class CustomGradCAM:
    """Simple GradCAM for Conv2D + sigmoid models."""

    def __init__(self, model, layer_name=None, log_file=None, debug_every=1):
        # Store original model for probability prediction
        self.original_model = model
        
        # Clone model and make last activation linear for GradCAM
        self.model = tf.keras.models.clone_model(model)
        self.model.set_weights(model.get_weights())
        last_layer = self.model.layers[-1]
        if hasattr(last_layer, "activation"):
            last_layer.activation = tf.keras.activations.linear

        self.layer_name = layer_name
        self.log_file = log_file
        self.debug_every = debug_every
        self._counter = 0

        # Find last Conv2D layer
        if self.layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    self.layer_name = layer.name
                    break
        self._log(f"[GradCAM] Using layer: {self.layer_name}")

        # Grad model
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

    def _log(self, msg):
        print(msg)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}\n")

    def compute_heatmap(self, image, class_idx=0):
        """Compute GradCAM heatmap for a single image."""
        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        with tf.GradientTape() as tape:
            conv_out, logits = self.grad_model(image, training=False)
            # class 1 → +logit, class 0 → -logit
            class_channel = logits[:, 0] if class_idx == 1 else -logits[:, 0]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            self._log("[GradCAM] Gradient is None, returning zeros.")
            h, w = int(conv_out.shape[1]), int(conv_out.shape[2])
            return np.zeros((h, w), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]  # (H, W, C)
        # Weighted sum across channels
        heatmap = tf.tensordot(conv_out, pooled_grads, axes=[[2], [0]])
        #heatmap = tf.reduce_sum(conv_out * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap - tf.reduce_min(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        if self._counter % self.debug_every == 0:
            self._log(f"[DEBUG #{self._counter}] grad μ={tf.reduce_mean(grads):.6f}, "
                      f"heatmap σ={tf.math.reduce_std(heatmap):.6f}")
        self._counter += 1

        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """Overlay heatmap on image."""
        H, W = image.shape[:2]
        heatmap = tf.image.resize(heatmap[..., None], (H, W))[..., 0].numpy()
        heatmap = np.clip(heatmap, 0, 1)

        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        cmap = plt.cm.get_cmap("jet")
        heatmap_rgb = cmap(heatmap)[..., :3]
        overlay = (1 - alpha) * img + alpha * heatmap_rgb
        return np.clip(overlay, 0, 1)

    def visualize(self, image, class_names=('NotDrowsy', 'Drowsy'),
                  true_idx=None, save_path=None, threshold=0.5):
        """Generate GradCAM visualization."""
        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        # Use original model for probability prediction (with sigmoid activation)
        preds = self.original_model.predict(image, verbose=0)
        prob = float(preds[0][0])
        pred_cls = 1 if prob >= threshold else 0
        disp_prob = prob if pred_cls == 1 else (1.0 - prob)

        heatmap = self.compute_heatmap(image[0], class_idx=pred_cls)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image[0])
        txt = f"Pred: {class_names[pred_cls]} ({disp_prob:.3f})"
        if true_idx is not None:
            txt = f"Truth: {class_names[int(true_idx)]} | " + txt
        axes[0].text(5, 15, txt, color='white', fontsize=10,
                     bbox=dict(facecolor='black', alpha=0.6))
        axes[0].set_title("Original")
        axes[0].axis("off")

        im1 = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Heatmap")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1])

        overlay = self.overlay_heatmap(heatmap, image[0])
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self._log(f"[GradCAM] Saved: {save_path}")
        plt.close(fig)
        return heatmap
