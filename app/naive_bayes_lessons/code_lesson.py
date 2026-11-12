"""Full fit/predict code corresponding to the derivation."""

from __future__ import annotations

import inspect

import streamlit as st

from naive_bayes_lab import model as nb_model
from naive_bayes_lab.walks import LABELS, TEXTS


SOURCE = '''import math
from collections import Counter


def fit(texts: list[str], labels: list[int]):
    # Count reviews per class: estimates P(class)
    doc_counts = Counter(labels)
    n_docs = len(texts)

    # Count word occurrences separately for each class
    word_counts = {0: Counter(), 1: Counter()}
    total_words = Counter()
    vocab = set()
    for text, label in zip(texts, labels):
        words = text.lower().split()
        word_counts[label].update(words)
        total_words[label] += len(words)
        vocab.update(words)

    return word_counts, total_words, doc_counts, vocab, n_docs


def predict(text: str, model, alpha=1):
    word_counts, total_words, doc_counts, vocab, n_docs = model
    words = text.lower().split()
    V = len(vocab)

    # 1. Prior log-odds: log(P(+) / P(-))
    score = math.log(doc_counts[1] / n_docs) \\
          - math.log(doc_counts[0] / n_docs)

    # 2. Add evidence from each word
    for word in words:
        if word not in vocab:
            continue
        # 3. Estimate P(word | positive), with smoothing
        p_pos = (word_counts[1][word] + alpha) / (
            total_words[1] + alpha * V
        )
        # 4. Estimate P(word | negative), with smoothing
        p_neg = (word_counts[0][word] + alpha) / (
            total_words[0] + alpha * V
        )
        # 5. Add log-likelihood ratio
        score += math.log(p_pos / p_neg)

    # 6. Positive log-odds means positive class is more likely
    if score > 0:
        return "positive", score
    elif score < 0:
        return "negative", score
    else:
        return "tie", score


texts = [
    "good movie",
    "great movie",
    "bad movie",
    "not good",
]
labels = [1, 1, 0, 0]
'''


def render() -> None:
    st.header("Put the math into code")
    st.markdown(
        "Every line below matches the derivation. "
        "`fit()` learns the counts; `predict()` turns those counts into a log-odds score."
    )
    st.code(SOURCE, language="python")

    st.subheader("Same functions live in the repo")
    st.caption("`src/naive_bayes_lab/model.py` — what the later trace lesson calls.")
    st.code(inspect.getsource(nb_model.fit) + "\n" + inspect.getsource(nb_model.predict), language="python")

    st.markdown("Training data used everywhere in this lab:")
    st.code(
        "texts = " + repr(TEXTS) + "\nlabels = " + repr(LABELS),
        language="python",
    )
