import numpy as np
import os
import json
from typing import Any, Dict, Optional

from src.cv_dataloader import collect_file_paths, labels_from_paths, make_tf_dataset_from_paths


def save_fold_datasets(base_dir, k, img_size, seed, class_names, output_dir="fold_datasets"):
    file_paths = collect_file_paths(base_dir, class_names, img_size)
    labels = labels_from_paths(file_paths, class_names)

    indices = np.arange(len(file_paths))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    os.makedirs(output_dir, exist_ok=True)

    for fold_idx in range(k):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        train_files = [file_paths[i] for i in train_idx]
        val_files = [file_paths[i] for i in val_idx]

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        payload = {
            "meta": {
                "fold": fold_idx + 1,
                "k": k,
                "seed": seed,
                "base_dir": os.path.abspath(base_dir),
                "class_names": list(class_names),
                "img_size": list(img_size) if isinstance(img_size, (tuple, list)) else img_size,
                "train_size": int(len(train_files)),
                "val_size": int(len(val_files)),
            },
            "train":
                {
                    "files": train_files,
                    "labels": train_labels.tolist()
                },
            "val":
                {
                    "files": val_files,
                    "labels": val_labels.tolist()
                }
        }

        with open(os.path.join(output_dir, f"fold_{fold_idx+1}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Fold {fold_idx+1} datasets saved to {os.path.join(output_dir, f'fold_{fold_idx+1}.json')}")


def load_fold_manifest(fold_idx: int, output_dir: str = "fold_datasets") -> Dict[str, Any]:
    path = os.path.join(output_dir, f"fold_{fold_idx+1}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fold_datasets(fold_idx, output_dir="fold_datasets"):
    data = load_fold_manifest(fold_idx, output_dir)
    train_files = data["train"]["files"]
    train_labels = data["train"]["labels"]
    val_files = data["val"]["files"]
    val_labels = data["val"]["labels"]
    return train_files, train_labels, val_files, val_labels


def sample_weights_for_train_files(
    train_files,
    weight_map: dict,
    base_dir: str,
) -> np.ndarray:
    """Align JSON keys (rel. to base_dir, forward slashes) with train file order."""
    base_abs = os.path.abspath(base_dir)
    weights = []
    for fp in train_files:
        rel = os.path.relpath(os.path.abspath(fp), base_abs).replace("\\", "/")
        weights.append(float(weight_map.get(rel, 1.0)))
    return np.asarray(weights, dtype=np.float32)


def create_tf_datasets_for_fold(
    fold_idx,
    img_size,
    batch_size,
    seed,
    class_names,
    output_dir="fold_datasets",
    sample_weights_path: Optional[str] = None,
    weights_base_dir: Optional[str] = None,
):
    """
    Build train/val tf.data pipelines from fold JSON.

    If ``sample_weights_path`` points to a JSON dict (rel_path -> weight), train batches
    become (x, y, sample_weight). Missing keys default to 1.0.

    ``weights_base_dir``: root for rel_path keys (usually the same as dataset root used
    when building weights). If None, uses ``meta.base_dir`` from the fold JSON.
    """
    data = load_fold_manifest(fold_idx, output_dir)
    train_files = data["train"]["files"]
    train_labels = np.asarray(data["train"]["labels"], dtype=np.float32)
    val_files = data["val"]["files"]
    val_labels = np.asarray(data["val"]["labels"], dtype=np.float32)

    train_sample_weights = None
    if sample_weights_path:
        if not os.path.isfile(sample_weights_path):
            raise FileNotFoundError(f"sample_weights_path not found: {sample_weights_path}")
        base = weights_base_dir or (data.get("meta") or {}).get("base_dir")
        if not base:
            raise ValueError(
                "weights_base_dir is required when fold JSON has no meta.base_dir "
                "(needed to match JSON keys like Drowsy/a.png)."
            )
        with open(sample_weights_path, "r", encoding="utf-8") as f:
            wmap = json.load(f)
        train_sample_weights = sample_weights_for_train_files(train_files, wmap, base)
        base_abs = os.path.abspath(base)
        matched = sum(
            1
            for fp in train_files
            if os.path.relpath(os.path.abspath(fp), base_abs).replace("\\", "/") in wmap
        )
        print(
            f"[fold_functions] Sample weights: {sample_weights_path} | "
            f"matched {matched}/{len(train_files)} keys (others -> 1.0)"
        )

    train_ds = make_tf_dataset_from_paths(
        train_files,
        train_labels,
        img_size,
        batch_size,
        augment=True,
        seed=seed,
        sample_weights=train_sample_weights,
    )
    val_ds = make_tf_dataset_from_paths(
        val_files,
        val_labels,
        img_size,
        batch_size,
        augment=False,
        seed=seed,
        sample_weights=None,
    )

    return train_ds, val_ds
