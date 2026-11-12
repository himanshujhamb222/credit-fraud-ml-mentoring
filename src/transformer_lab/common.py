"""Small formatters shared by the worked examples."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def f4(value: float) -> str:
    number = float(value)
    if math.isinf(number):
        return "-inf" if number < 0 else "inf"
    return f"{number:.4f}"


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(scores, dtype=float)
    for i, row in enumerate(scores):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        shift = row[finite] - np.max(row[finite])
        exps = np.exp(shift)
        total = float(exps.sum())
        weights[i, finite] = exps / total
    return weights


def labeled_frame(matrix: np.ndarray, labels: list[str], index_name: str = "token") -> pd.DataFrame:
    shown = np.array(matrix, dtype=object)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            shown[i, j] = f4(matrix[i, j])
    frame = pd.DataFrame(shown, index=labels, columns=labels)
    frame.index.name = index_name
    return frame


def softmax_lines(scores: np.ndarray, labels: list[str]) -> tuple[list[str], np.ndarray]:
    finite = np.isfinite(scores)
    raw = np.where(finite, np.exp(np.where(finite, scores, 0.0)), 0.0)
    total = float(raw.sum())
    weights = softmax_rows(scores.reshape(1, -1))[0]
    lines = [
        "weight = exp(score) / sum exp(score)",
        "scores: " + ", ".join(f"{name}={f4(score)}" for name, score in zip(labels, scores)),
        "exp(score): " + ", ".join(f4(item) for item in raw),
        f"sum of exp = {f4(total)}",
        "weights: " + ", ".join(f"{name}={f4(weight)}" for name, weight in zip(labels, weights)),
        f"weights sum to {f4(float(weights.sum()))}",
    ]
    return lines, weights
