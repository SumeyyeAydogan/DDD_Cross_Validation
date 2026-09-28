import json
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TunedThresholdClassifierCV


class _PrefitProba(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.asarray(X)


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


def fit_threshold_on_dataset(
    model,
    dataset,
    scoring="balanced_accuracy"
):
    y, scores = collect_proba(model, dataset)

    proba = np.column_stack([
        1.0 - scores,
        scores
    ])

    est = _PrefitProba()
    est.fit(proba, y)

    ttc = TunedThresholdClassifierCV(
        estimator=est,
        cv="prefit",
        scoring=scoring,
        refit=False,
        thresholds=np.linspace(0.0, 1.0, 1001),
        store_cv_results=True,
    )

    ttc.fit(proba, y)
    
    return float(ttc.best_threshold_), ttc


def save_threshold(path, threshold, scoring="balanced_accuracy"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"threshold": threshold, "scoring": scoring},
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
