import numpy as np
import os
import json
from typing import Any, Dict, Optional

from src.cv_dataloader import collect_file_paths, labels_from_paths, make_tf_dataset_from_paths
from src.fold_split import build_fold_payloads


def save_fold_datasets(
    base_dir,
    k,
    img_size,
    seed,
    class_names,
    output_dir="fold_datasets",
    val_ratio: float = 0.15,
):
    file_paths = collect_file_paths(base_dir, class_names, img_size)
    labels = labels_from_paths(file_paths, class_names)

    os.makedirs(output_dir, exist_ok=True)

    payloads = build_fold_payloads(
        file_paths=file_paths,
        labels=labels,
        class_names=class_names,
        k=k,
        seed=seed,
        val_ratio=val_ratio,
        base_dir=base_dir,
        img_size=img_size,
    )

    for payload in payloads:
        fold_idx = payload["meta"]["fold"]
        out_path = os.path.join(output_dir, f"fold_{fold_idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Fold {fold_idx} saved -> {out_path}")


def load_fold_manifest(fold_idx: int, output_dir: str = "fold_datasets") -> Dict[str, Any]:
    path = os.path.join(output_dir, f"fold_{fold_idx+1}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fold_datasets(fold_idx, output_dir="fold_datasets"):
    data = load_fold_manifest(fold_idx, output_dir)

    # New schema (train_fit / val_monitor / test)
    if "train_fit" in data:
        return (
            data["train_fit"]["files"],
            data["train_fit"]["labels"],
            data["val_monitor"]["files"],
            data["val_monitor"]["labels"],
            data["test"]["files"],
            data["test"]["labels"],
        )

    # Legacy schema fallback
    return (
        data["train"]["files"],
        data["train"]["labels"],
        data["val"]["files"],
        data["val"]["labels"],
        None,
        None,
    )


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
    data = load_fold_manifest(fold_idx, output_dir)

    if "train_fit" in data:
        train_files = data["train_fit"]["files"]
        train_labels = np.asarray(data["train_fit"]["labels"], dtype=np.float32)
        val_monitor_files = data["val_monitor"]["files"]
        val_monitor_labels = np.asarray(data["val_monitor"]["labels"], dtype=np.float32)
        test_files = data["test"]["files"]
        test_labels = np.asarray(data["test"]["labels"], dtype=np.float32)
    else:
        # Legacy JSON (train / val only)
        train_files = data["train"]["files"]
        train_labels = np.asarray(data["train"]["labels"], dtype=np.float32)
        val_monitor_files = data["val"]["files"]
        val_monitor_labels = np.asarray(data["val"]["labels"], dtype=np.float32)
        test_files = data["val"]["files"]
        test_labels = np.asarray(data["val"]["labels"], dtype=np.float32)

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

    train_fit_ds = make_tf_dataset_from_paths(
        train_files, train_labels, img_size, batch_size,
        augment=True, seed=seed, sample_weights=train_sample_weights,
    )
    val_monitor_ds = make_tf_dataset_from_paths(
        val_monitor_files, val_monitor_labels, img_size, batch_size,
        augment=False, seed=seed, sample_weights=None,
    )
    test_ds = make_tf_dataset_from_paths(
        test_files, test_labels, img_size, batch_size,
        augment=False, seed=seed, sample_weights=None,
    )

    return train_fit_ds, val_monitor_ds, test_ds
