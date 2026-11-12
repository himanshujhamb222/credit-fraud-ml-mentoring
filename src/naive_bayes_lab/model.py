"""Minimal multinomial Naive Bayes for the teaching lab."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

Model = tuple[dict[int, Counter], Counter, Counter, set[str], int]


def fit(texts: list[str], labels: list[int]) -> Model:
    """Count docs and words per class. Returns the pieces predict() needs."""
    doc_counts = Counter(labels)
    n_docs = len(texts)

    word_counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
    total_words: Counter = Counter()
    vocab: set[str] = set()

    for text, label in zip(texts, labels):
        words = text.lower().split()
        word_counts[label].update(words)
        total_words[label] += len(words)
        vocab.update(words)

    return word_counts, total_words, doc_counts, vocab, n_docs


def predict(
    text: str,
    model: Model,
    alpha: float = 1.0,
) -> tuple[str, float]:
    """Log-odds of positive vs negative. Score > 0 → positive."""
    word_counts, total_words, doc_counts, vocab, n_docs = model
    words = text.lower().split()
    v = len(vocab)

    score = math.log(doc_counts[1] / n_docs) - math.log(doc_counts[0] / n_docs)

    for word in words:
        if word not in vocab:
            continue
        p_pos = (word_counts[1][word] + alpha) / (total_words[1] + alpha * v)
        p_neg = (word_counts[0][word] + alpha) / (total_words[0] + alpha * v)
        score += math.log(p_pos / p_neg)

    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "tie", score


def word_prob(
    word: str,
    label: int,
    model: Model,
    alpha: float = 1.0,
) -> float:
    """Laplace-smoothed P(word | class)."""
    word_counts, total_words, _, vocab, _ = model
    v = len(vocab)
    return (word_counts[label][word] + alpha) / (total_words[label] + alpha * v)


def unpack_model(model: Model) -> dict[str, Any]:
    word_counts, total_words, doc_counts, vocab, n_docs = model
    return {
        "word_counts": word_counts,
        "total_words": total_words,
        "doc_counts": doc_counts,
        "vocab": vocab,
        "n_docs": n_docs,
    }
