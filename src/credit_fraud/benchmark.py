"""Simple baselines before heavy tree models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_fraud.modeling import classification_metrics, predict_proba_positive


@dataclass
class BenchmarkResult:
    metrics: dict[str, dict[str, float]]
    models: dict[str, Any]


class MajorityClassifier:
    """Always predict the majority class; proba = majority rate."""

    def __init__(self) -> None:
        self.majority_label_: int = 0
        self.majority_rate_: float = 1.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> MajorityClassifier:
        counts = y.value_counts()
        self.majority_label_ = int(counts.idxmax())
        self.majority_rate_ = float(counts.max() / len(y))
        self.classes_ = np.array(sorted(y.unique()))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(shape=len(x), fill_value=self.majority_label_, dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        # column order follows classes_
        proba = np.zeros((len(x), len(self.classes_)), dtype=float)
        maj_idx = list(self.classes_).index(self.majority_label_)
        proba[:, maj_idx] = self.majority_rate_
        other = 1.0 - self.majority_rate_
        for i in range(len(self.classes_)):
            if i != maj_idx:
                proba[:, i] = other / max(len(self.classes_) - 1, 1)
        return proba


def run_benchmark(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> BenchmarkResult:
    """Majority + balanced logistic regression floor metrics."""
    majority = MajorityClassifier().fit(x_train, y_train)
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    logistic.fit(x_train, y_train)

    models = {"majority": majority, "logistic": logistic}
    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        proba = predict_proba_positive(model, x_test)
        metrics[name] = classification_metrics(y_test, proba)

    return BenchmarkResult(metrics=metrics, models=models)
