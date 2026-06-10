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