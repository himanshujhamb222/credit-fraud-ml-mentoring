"""Model factories and evaluation helpers for RF / XGBoost."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from credit_fraud import config


def positive_rate(y: pd.Series | np.ndarray) -> float:
    arr = np.asarray(y)
    return float((arr == config.POSITIVE_LABEL).mean())


def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """XGBoost imbalance weight ≈ neg / pos."""
    arr = np.asarray(y)
    pos = max(int((arr == config.POSITIVE_LABEL).sum()), 1)
    neg = int((arr != config.POSITIVE_LABEL).sum())
    return float(neg / pos)


def make_random_forest(**params: Any) -> RandomForestClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    defaults.update(params)
    return RandomForestClassifier(**defaults)


def make_xgboost(y_train: pd.Series | np.ndarray | None = None, **params: Any) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    if y_train is not None and "scale_pos_weight" not in params:
        defaults["scale_pos_weight"] = scale_pos_weight(y_train)
    defaults.update(params)
    return XGBClassifier(**defaults)


def predict_proba_positive(model: Any, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    # assume binary; positive class at index matching classes_
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if config.POSITIVE_LABEL in classes:
            idx = classes.index(config.POSITIVE_LABEL)
            return proba[:, idx]
    return proba[:, -1]


def classification_metrics(y_true: pd.Series | np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def feature_importance_frame(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        raise AttributeError("Model has no feature_importances_")
    frame = pd.DataFrame({"feature": feature_names, "importance": imp})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)
