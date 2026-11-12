"""Worked Naive Bayes sentiment lab: four reviews, fit, predict, log-odds."""

from naive_bayes_lab.model import fit, predict
from naive_bayes_lab.walks import (
    LABELS,
    TEXTS,
    TrainingWalk,
    predict_trace,
    training_walk,
)

__all__ = [
    "LABELS",
    "TEXTS",
    "TrainingWalk",
    "fit",
    "predict",
    "predict_trace",
    "training_walk",
]
