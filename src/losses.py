import tensorflow as tf
from typing import Optional


class SimpleMaskedLoss(tf.keras.losses.Loss):
    def __init__(self, name="simple_masked_loss", from_logits: bool = False, **kwargs):
        # Keep reduction as NONE and compute the mean ourselves
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Per-element BCE, shape (batch, 1) or (batch,)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Validate and fix invalid predictions
        # Check for NaN/Inf in predictions
        has_nan = tf.reduce_any(tf.math.is_nan(y_pred))
        has_inf = tf.reduce_any(tf.math.is_inf(y_pred))
        
        # Clip predictions to valid range if not from_logits
        if not self.from_logits:
            # Clip to [epsilon, 1-epsilon] to avoid log(0) in BCE
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        else:
            # For logits, replace NaN/Inf with 0
            y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, 0.0)

        ce = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        
        # Replace any NaN/Inf in loss with a large finite value
        ce = tf.where(tf.math.is_finite(ce), ce, 1e6)
        
        return ce  # We do not reduce yet

    def __call__(
        self,
        y_true,
        y_pred,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs,
    ):
        ce = self.call(y_true, y_pred, **kwargs)  # (batch, 1) or (batch,)

        # Squeeze → (batch,) when we statically know there is an extra dim
        if ce.shape.rank is not None and ce.shape.rank > 1:
            ce = tf.squeeze(ce, axis=-1)
        ce = tf.cast(ce, tf.float32)

        # Check for NaN/Inf in cross-entropy values before processing
        has_nan_ce = tf.reduce_any(tf.math.is_nan(ce))
        has_inf_ce = tf.reduce_any(tf.math.is_inf(ce))
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            # Flatten and align lengths to the minimum (for safety)
            ce = tf.reshape(ce, [-1])
            sample_weight = tf.reshape(sample_weight, [-1])

            n = tf.minimum(tf.shape(ce)[0], tf.shape(sample_weight)[0])
            ce = ce[:n]
            sample_weight = sample_weight[:n]
            
            # Ensure sample weights are valid (no NaN/Inf, positive)
            sample_weight = tf.where(tf.math.is_finite(sample_weight), sample_weight, 0.0)
            sample_weight = tf.maximum(sample_weight, 0.0)  # Ensure non-negative

            tf.print(
                "[MaskedLoss] ✅ sample_weight RECEIVED | loss_len:",
                tf.shape(ce)[0],
                "| sw_len:", tf.shape(sample_weight)[0],
                "| mean:", tf.reduce_mean(sample_weight),
                "| min:", tf.reduce_min(sample_weight),
                "| max:", tf.reduce_max(sample_weight),
                "| ce_has_nan:", has_nan_ce,
                "| ce_has_inf:", has_inf_ce,
            )

            total_weight = tf.reduce_sum(sample_weight) + 1e-8
            weighted = ce * sample_weight
            loss = tf.reduce_sum(weighted) / total_weight
        else:
            tf.print(
                "[MaskedLoss] ⚠️ sample_weight is None | loss_len:",
                tf.shape(ce)[0],
                "| ce_has_nan:", has_nan_ce,
                "| ce_has_inf:", has_inf_ce,
            )
            loss = tf.reduce_mean(ce)
        
        # Final safety check: replace NaN/Inf loss with a large finite value
        loss = tf.where(tf.math.is_finite(loss), loss, 1e6)
        
        return loss


def create_simple_masked_loss():
    return SimpleMaskedLoss()
