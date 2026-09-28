# src/fold_split.py
import os
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Leading letters in the filename = person id (a0002 -> "a", A0002 -> "A")
_PERSON_RE = re.compile(r"^([A-Za-z]+)")


def person_id_from_path(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    m = _PERSON_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse person id from: {file_path}")
    return m.group(1)  # case-sensitive: a != A


def class_name_from_path(file_path: str, class_names: Sequence[str]) -> str:
    parent = os.path.basename(os.path.dirname(file_path))
    if parent not in class_names:
        raise ValueError(f"Unknown class folder '{parent}' in {file_path}")
    return parent


def group_key(file_path: str, class_names: Sequence[str]) -> Tuple[str, str]:
    """Return (class, person); each group gets its own round-robin."""
    return class_name_from_path(file_path, class_names), person_id_from_path(file_path)


def round_robin_fold_assignments(
    items: List[str],
    k: int,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    items: image paths for one person
    return: {path: fold_index}  fold_index in [0, k-1]
    """
    shuffled = list(items)
    rng.shuffle(shuffled)
    return {path: i % k for i, path in enumerate(shuffled)}


def assign_outer_test_folds(
    file_paths: List[str],
    class_names: Sequence[str],
    k: int,
    seed: int,
) -> Dict[str, int]:
    """
    Independent round-robin per (class, person) group.
    Each image is assigned to exactly one test fold.
    """
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for p in file_paths:
        groups[group_key(p, class_names)].append(p)

    assignment: Dict[str, int] = {}
    for key in sorted(groups.keys()):  # deterministic order
        group_rng = np.random.default_rng(seed + hash(key) % 10_000)
        assignment.update(round_robin_fold_assignments(groups[key], k, group_rng))
    return assignment


def assign_outer_test_folds_stratified(
    file_paths: List[str],
    class_names: Sequence[str],
    k: int,
    seed: int,
) -> Dict[str, int]:
    """
    Stratified-only outer split (no person grouping).
    Each class is shuffled independently, then assigned round-robin to folds.
    """
    class_groups: Dict[str, List[str]] = defaultdict(list)
    for p in file_paths:
        class_groups[class_name_from_path(p, class_names)].append(p)

    assignment: Dict[str, int] = {}
    for class_name in sorted(class_groups.keys()):
        class_rng = np.random.default_rng(seed + hash(class_name) % 10_000)
        assignment.update(round_robin_fold_assignments(class_groups[class_name], k, class_rng))
    return assignment


def split_train_pool_inner(
    train_pool_paths: List[str],
    labels_by_path: Dict[str, float],
    class_names: Sequence[str],
    val_ratio: float,
    seed: int,
    fold_idx: int,
) -> Tuple[List[str], List[str]]:
    """
    Split train_pool into train_fit / val_monitor via per-person round-robin.
    """
    if not train_pool_paths:
        return [], []

    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for p in train_pool_paths:
        groups[group_key(p, class_names)].append(p)

    train_fit: List[str] = []
    val_monitor: List[str] = []

    for key in sorted(groups.keys()):
        items = sorted(groups[key])  # stable order
        m = len(items)
        if m == 1:
            train_fit.append(items[0])
            continue

        n_monitor = int(round(m * val_ratio))
        n_monitor = max(1, min(m - 1, n_monitor))  # keep train_fit non-empty

        group_rng = np.random.default_rng(seed + fold_idx * 1000 + hash(key) % 10_000)
        shuffled = list(items)
        group_rng.shuffle(shuffled)

        val_monitor.extend(shuffled[:n_monitor])
        train_fit.extend(shuffled[n_monitor:])

    return train_fit, val_monitor


def split_train_pool_inner_stratified(
    train_pool_paths: List[str],
    class_names: Sequence[str],
    val_ratio: float,
    seed: int,
    fold_idx: int,
) -> Tuple[List[str], List[str]]:
    """
    Split train_pool into train_fit / val_monitor with class-wise stratification.
    """
    if not train_pool_paths:
        return [], []

    class_groups: Dict[str, List[str]] = defaultdict(list)
    for p in train_pool_paths:
        class_groups[class_name_from_path(p, class_names)].append(p)

    train_fit: List[str] = []
    val_monitor: List[str] = []

    for class_name in sorted(class_groups.keys()):
        items = sorted(class_groups[class_name])
        m = len(items)
        if m == 1:
            train_fit.append(items[0])
            continue

        n_monitor = int(round(m * val_ratio))
        n_monitor = max(1, min(m - 1, n_monitor))  # keep train_fit non-empty

        class_rng = np.random.default_rng(seed + fold_idx * 1000 + hash(class_name) % 10_000)
        shuffled = list(items)
        class_rng.shuffle(shuffled)

        val_monitor.extend(shuffled[:n_monitor])
        train_fit.extend(shuffled[n_monitor:])

    return train_fit, val_monitor


def build_fold_payloads(
    file_paths: List[str],
    labels: np.ndarray,
    class_names: Sequence[str],
    k: int,
    seed: int,
    val_ratio: float,
    base_dir: str,
    img_size,
) -> List[dict]:
    labels_by_path = {p: float(l) for p, l in zip(file_paths, labels)}
    outer = assign_outer_test_folds(file_paths, class_names, k, seed)

    payloads = []
    for fold_idx in range(k):
        test_files = [p for p in file_paths if outer[p] == fold_idx]
        train_pool_files = [p for p in file_paths if outer[p] != fold_idx]

        train_fit_files, val_monitor_files = split_train_pool_inner(
            train_pool_files,
            labels_by_path,
            class_names,
            val_ratio=val_ratio,
            seed=seed,
            fold_idx=fold_idx,
        )

        def pack(paths: List[str]) -> dict:
            return {
                "files": paths,
                "labels": [labels_by_path[p] for p in paths],
            }

        payloads.append({
            "meta": {
                "fold": fold_idx + 1,
                "k": k,
                "seed": seed,
                "val_ratio": val_ratio,
                "split_method": "per_subject_round_robin",
                "base_dir": os.path.abspath(base_dir),
                "class_names": list(class_names),
                "img_size": list(img_size) if isinstance(img_size, (tuple, list)) else img_size,
                "train_fit_size": len(train_fit_files),
                "val_monitor_size": len(val_monitor_files),
                "test_size": len(test_files),
            },
            "train_fit": pack(train_fit_files),
            "val_monitor": pack(val_monitor_files),
            "test": pack(test_files),
        })
    return payloads


def _split_indices_random_inner(
    indices: np.ndarray,
    val_ratio: float,
    seed: int,
    fold_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random train_fit / val_monitor split inside the outer train pool (per-fold RNG)."""
    indices = np.asarray(indices)
    if indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(seed + fold_idx * 1000)
    shuffled = indices.copy()
    rng.shuffle(shuffled)

    n = shuffled.size
    if n == 1:
        return shuffled, np.array([], dtype=int)

    n_monitor = int(round(n * val_ratio))
    n_monitor = max(1, min(n - 1, n_monitor))
    val_monitor_idx = shuffled[:n_monitor]
    train_fit_idx = shuffled[n_monitor:]
    return train_fit_idx, val_monitor_idx


def build_fold_payloads_random(
    file_paths: List[str],
    labels: np.ndarray,
    class_names: Sequence[str],
    k: int,
    seed: int,
    val_ratio: float,
    base_dir: str,
    img_size,
) -> List[dict]:
    """
    Random outer CV folds with inner train_fit / val_monitor split (Yawning-style).

    Per fold:
      - test: held-out outer fold chunk
      - train pool: remaining chunks → randomly split into train_fit and val_monitor
    """
    labels_by_path = {p: float(l) for p, l in zip(file_paths, labels)}

    indices = np.arange(len(file_paths))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    payloads = []
    for fold_idx in range(k):
        test_idx = folds[fold_idx]
        train_pool_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
        train_fit_idx, val_monitor_idx = _split_indices_random_inner(
            train_pool_idx, val_ratio=val_ratio, seed=seed, fold_idx=fold_idx
        )

        def paths_from_indices(idxs: np.ndarray) -> List[str]:
            return [file_paths[int(i)] for i in idxs]

        test_files = paths_from_indices(test_idx)
        train_fit_files = paths_from_indices(train_fit_idx)
        val_monitor_files = paths_from_indices(val_monitor_idx)

        def pack(paths: List[str]) -> dict:
            return {
                "files": paths,
                "labels": [labels_by_path[p] for p in paths],
            }

        payloads.append({
            "meta": {
                "fold": fold_idx + 1,
                "k": k,
                "seed": seed,
                "val_ratio": val_ratio,
                "split_method": "random_outer_fold_inner_val",
                "base_dir": os.path.abspath(base_dir),
                "class_names": list(class_names),
                "img_size": list(img_size) if isinstance(img_size, (tuple, list)) else img_size,
                "train_fit_size": len(train_fit_files),
                "val_monitor_size": len(val_monitor_files),
                "test_size": len(test_files),
            },
            "train_fit": pack(train_fit_files),
            "val_monitor": pack(val_monitor_files),
            "test": pack(test_files),
        })
    return payloads


def build_fold_payloads_stratified(
    file_paths: List[str],
    labels: np.ndarray,
    class_names: Sequence[str],
    k: int,
    seed: int,
    val_ratio: float,
    base_dir: str,
    img_size,
) -> List[dict]:
    labels_by_path = {p: float(l) for p, l in zip(file_paths, labels)}
    outer = assign_outer_test_folds_stratified(file_paths, class_names, k, seed)

    payloads = []
    for fold_idx in range(k):
        test_files = [p for p in file_paths if outer[p] == fold_idx]
        train_pool_files = [p for p in file_paths if outer[p] != fold_idx]

        train_fit_files, val_monitor_files = split_train_pool_inner_stratified(
            train_pool_files,
            class_names,
            val_ratio=val_ratio,
            seed=seed,
            fold_idx=fold_idx,
        )

        def pack(paths: List[str]) -> dict:
            return {
                "files": paths,
                "labels": [labels_by_path[p] for p in paths],
            }

        payloads.append({
            "meta": {
                "fold": fold_idx + 1,
                "k": k,
                "seed": seed,
                "val_ratio": val_ratio,
                "split_method": "stratified_round_robin",
                "base_dir": os.path.abspath(base_dir),
                "class_names": list(class_names),
                "img_size": list(img_size) if isinstance(img_size, (tuple, list)) else img_size,
                "train_fit_size": len(train_fit_files),
                "val_monitor_size": len(val_monitor_files),
                "test_size": len(test_files),
            },
            "train_fit": pack(train_fit_files),
            "val_monitor": pack(val_monitor_files),
            "test": pack(test_files),
        })
    return payloads