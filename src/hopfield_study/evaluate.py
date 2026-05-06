from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MetricRow:
    name: str
    accuracy: float
    f1: float
    auroc: float


def scalar_score_metrics(name: str, scores: np.ndarray, y: np.ndarray) -> MetricRow:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    threshold = _fit_threshold(scores, y)
    pred = (scores >= threshold).astype(int)
    return MetricRow(
        name=name,
        accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
        auroc=float(roc_auc_score(y, scores)),
    )


def cv_logreg_metrics(name: str, X: np.ndarray, y: np.ndarray, seed: int = 42) -> MetricRow:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    class_counts = np.bincount(y, minlength=2)
    n_splits = min(5, int(class_counts.min()))
    if len(y) < 10 or n_splits < 2:
        return MetricRow(name=name, accuracy=float("nan"), f1=float("nan"), auroc=float("nan"))
    probs = np.zeros(len(y), dtype=np.float64)
    preds = np.zeros(len(y), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_val_idx, test_idx in splitter.split(X, y):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.15,
            random_state=seed,
            stratify=y[train_val_idx],
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        val_probs = model.predict_proba(X[val_idx])[:, 1]
        threshold = _fit_threshold(val_probs, y[val_idx])
        test_probs = model.predict_proba(X[test_idx])[:, 1]
        probs[test_idx] = test_probs
        preds[test_idx] = (test_probs >= threshold).astype(int)

    return MetricRow(
        name=name,
        accuracy=float(accuracy_score(y, preds)),
        f1=float(f1_score(y, preds, zero_division=0)),
        auroc=float(roc_auc_score(y, probs)),
    )


def majority_metrics(y: np.ndarray) -> MetricRow:
    y = np.asarray(y, dtype=np.int64)
    majority = int(np.bincount(y).argmax())
    pred = np.full_like(y, majority)
    return MetricRow(
        name="majority",
        accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
        auroc=float("nan"),
    )


def _fit_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([scores, np.linspace(scores.min(), scores.max(), 101)]))
    best_threshold = float(np.median(scores))
    best_acc = -1.0
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)
    return best_threshold
