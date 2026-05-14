import os
import tensorflow as tf
import numpy as np
from typing import List

def collect_file_paths(base_dir, class_names, img_size):

    ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=1,
        shuffle=False,
    )
    file_paths = list(getattr(ds, "file_paths", []))
    return file_paths

def labels_from_paths(file_paths, class_names):
    class_to_idx = {name: float(idx) for idx, name in enumerate(class_names)}
    labels_list: List[float] = []
    for path in file_paths:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class_to_idx:
            raise ValueError(f"Unknown class directory '{class_name}' for file '{path}'. Expected one of '{class_names}'")
        labels_list.append(class_to_idx[class_name])
    labels = np.array(labels_list, dtype=np.float32)
    return labels

def make_tf_dataset_from_paths(file_paths, labels, img_size, batch_size, augment=True, sample_weights=None, seed=None):
    AUTOTUNE = tf.data.AUTOTUNE
    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    # Handling sample weights if provided
    if sample_weights is not None:
        weights_ds = tf.data.Dataset.from_tensor_slices(sample_weights.astype(np.float32))
        ds = tf.data.Dataset.zip((paths_ds, labels_ds, weights_ds))
    else:
        ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    # Define preprocessing function based on sample_weights
    def _load_and_preprocess(path, label, weight=None):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        
        label = tf.expand_dims(label, axis=-1)
        
        if weight is not None:
            weight = tf.cast(weight, tf.float32)
            return img, label, weight
        return img, label

    # Use map function based on the presence of sample_weights
    if sample_weights is not None:
        ds = ds.map(lambda path, label, weight: _load_and_preprocess(path, label, weight), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda path, label: _load_and_preprocess(path, label), num_parallel_calls=AUTOTUNE)

    # Skip samples that fail to decode (e.g., corrupted image files)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    #ds = ds.cache()

    # Apply data augmentation if needed
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.1, seed=None if seed is None else seed + 1),
            tf.keras.layers.RandomZoom(0.1, seed=None if seed is None else seed + 2),
            tf.keras.layers.RandomTranslation(0.1, 0.1, seed=None if seed is None else seed + 3),
        ])
        
        # Apply augmentation to data with or without sample weights
        def _apply_augment(x, y, w=None):
            if w is not None:
                return aug(x, training=True), y, w
            return aug(x, training=True), y

        if sample_weights is not None:
            ds = ds.map(lambda x, y, w: _apply_augment(x, y, w), num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: _apply_augment(x, y), num_parallel_calls=AUTOTUNE)

    # Shuffle the dataset if augmentation is applied
    if augment:
        ds = ds.shuffle(1000, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)

    return ds