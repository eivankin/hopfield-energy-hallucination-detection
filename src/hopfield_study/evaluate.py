from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
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


def cv_logreg_metrics(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> MetricRow:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    class_counts = np.bincount(y, minlength=2)
    n_splits = min(5, int(class_counts.min()))
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
    if len(y) < 10 or n_splits < 2:
        return MetricRow(name=name, accuracy=float("nan"), f1=float("nan"), auroc=float("nan"))
    probs = np.zeros(len(y), dtype=np.float64)
    preds = np.zeros(len(y), dtype=np.int64)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups=groups)

    for train_val_idx, test_idx in split_iter:
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


def cv_logreg_grid_metrics(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    groups: np.ndarray | None = None,
    c_values: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0),
) -> MetricRow:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    class_counts = np.bincount(y, minlength=2)
    n_splits = min(5, int(class_counts.min()))
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
    if len(y) < 10 or n_splits < 2:
        return MetricRow(name=name, accuracy=float("nan"), f1=float("nan"), auroc=float("nan"))
    probs = np.zeros(len(y), dtype=np.float64)
    preds = np.zeros(len(y), dtype=np.int64)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups=groups)

    for train_val_idx, test_idx in split_iter:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.15,
            random_state=seed,
            stratify=y[train_val_idx],
        )
        best_model = None
        best_threshold = 0.5
        best_val_auc = -1.0
        for c_value in c_values:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=c_value,
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            model.fit(X[train_idx], y[train_idx])
            val_probs = model.predict_proba(X[val_idx])[:, 1]
            try:
                val_auc = roc_auc_score(y[val_idx], val_probs)
            except ValueError:
                val_auc = float("nan")
            if val_auc > best_val_auc:
                best_val_auc = float(val_auc)
                best_model = model
                best_threshold = _fit_threshold(val_probs, y[val_idx])
        if best_model is None:
            raise RuntimeError("No logistic model was fitted")
        test_probs = best_model.predict_proba(X[test_idx])[:, 1]
        probs[test_idx] = test_probs
        preds[test_idx] = (test_probs >= best_threshold).astype(int)

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
