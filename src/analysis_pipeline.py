from __future__ import annotations

from typing import List, Sequence, Tuple

import tensorflow as tf


def get_analysis_pipeline_from_data_dir(
    data_dir: str,
    img_size: Tuple[int, int],
    class_names: Sequence[str] = ("NotDrowsy", "Drowsy"),
) -> Tuple[tf.data.Dataset, List[str]]:
    """
    Build a simple per-sample dataset for analysis scripts.
    Returns a dataset of ((images, labels), paths) with batch_size=1.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=1,
        shuffle=False,
    )
    file_paths = list(getattr(ds, "file_paths", []))
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths).batch(1)
    ds = tf.data.Dataset.zip((ds, path_ds))
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds, file_paths
