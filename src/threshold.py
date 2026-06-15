import json
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TunedThresholdClassifierCV


class _PrefitProba(BaseEstimator, ClassifierMixin):
    """Sklearn shell; predict_proba returns precomputed scores."""

    def __init__(self, proba: np.ndarray):
        self.proba_ = proba
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.proba_


def collect_proba(model, dataset):
    y_true, scores = [], []
    for batch in dataset:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        p = np.clip(model.predict(x, verbose=0).reshape(-1), 0.0, 1.0)
        y_true.extend(y.numpy().ravel())
        scores.extend(p)
    return np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)


def fit_threshold_on_train(model, train_ds, scoring="balanced_accuracy"):
    y, scores = collect_proba(model, train_ds)
    proba = np.column_stack([1.0 - scores, scores])
    est = _PrefitProba(proba)
    est.fit(proba, y)
    ttc = TunedThresholdClassifierCV(
        estimator=est, cv="prefit", scoring=scoring, refit=False
    )
    ttc.fit(proba, y)
    return float(ttc.best_threshold_)


def fit_threshold_for_fold(model, fold_idx, img_size, batch_size, seed, output_dir="fold_datasets_v2"):
    from src.cv_dataloader import make_tf_dataset_from_paths
    from src.fold_functions import load_fold_manifest, resolve_train_fit_split

    data = load_fold_manifest(fold_idx, output_dir)
    train_files, train_labels, source = resolve_train_fit_split(data)
    ds = make_tf_dataset_from_paths(
        train_files,
        np.asarray(train_labels, dtype=np.float32),
        img_size,
        batch_size,
        augment=False,
        seed=seed,
    )
    return fit_threshold_on_train(model, ds), source


def save_threshold(path, threshold, scoring="balanced_accuracy", source="train_fit"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"threshold": threshold, "scoring": scoring, "source": source},
            f,
            indent=2,
        )


def load_threshold(path, default=0.5):
    if not path or not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def pred_class(prob: float, threshold: float = 0.5) -> int:
    return 1 if prob >= threshold else 0
