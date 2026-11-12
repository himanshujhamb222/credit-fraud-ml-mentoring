"""fit(): priors and per-class word counts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from naive_bayes_lab.walks import WORD_TABLE_ROWS, training_walk
from naive_bayes_lessons.ui import working
from st_compat import dataframe


def render() -> None:
    st.header("fit(): learn from the training data")
    walk = training_walk()
    priors, words = st.tabs(["Step 2.1 — Class priors", "Step 2.2 — Word counts"])

    with priors:
        st.markdown(
            "The **prior** is how likely a review is positive *before* we read its words. "
            "Count reviews in each class."
        )
        st.latex(r"P(+)=\frac{2}{4}=0.5,\qquad P(-)=\frac{2}{4}=0.5")
        st.code(
            "from collections import Counter\n"
            "doc_counts = Counter(labels)\n"
            "n_docs = len(texts)",
            language="python",
        )
        st.markdown(
            """
- `Counter(labels)` counts how many reviews belong to each class.
- `len(texts)` is the total number of reviews.
"""
        )
        working(
            [
                "doc_counts = {1: 2, 0: 2}",
                "n_docs = 4",
                f"p_pos = doc_counts[1] / n_docs  # {walk.p_pos}",
                f"p_neg = doc_counts[0] / n_docs  # {walk.p_neg}",
            ]
        )

    with words:
        st.markdown(
            r"We need \(P(w\mid +)\) and \(P(w\mid -)\): "
            "among positive reviews, how likely is a word? Same for negative."
        )
        rows = [
            {
                "Word": w,
                "Positive count": c_pos,
                "Negative count": c_neg,
            }
            for w, c_pos, c_neg in WORD_TABLE_ROWS
        ]
        rows.append(
            {
                "Word": "Total",
                "Positive count": walk.total_pos,
                "Negative count": walk.total_neg,
            }
        )
        dataframe(pd.DataFrame(rows), hide_index=True)

        st.code(
            "word_counts = {0: Counter(), 1: Counter()}\n"
            "total_words = Counter()\n"
            "vocab = set()\n"
            "for text, label in zip(texts, labels):\n"
            "    words = text.lower().split()\n"
            "    word_counts[label].update(words)\n"
            "    total_words[label] += len(words)\n"
            "    vocab.update(words)",
            language="python",
        )
        st.markdown(
            """
- `zip(texts, labels)` pairs each review with its sentiment.
- `word_counts[label].update(words)` adds words to that class's counter.
- `total_words[label] += len(words)` counts **occurrences**, not unique words.
- `vocab.update(words)` collects unique words across the dataset.
"""
        )
        working(
            [
                f"word_counts[1]  # {dict(sorted(walk.word_counts_pos.items()))}",
                f"word_counts[0]  # {dict(sorted(walk.word_counts_neg.items()))}",
                f"total_words[1]  # {walk.total_pos}",
                f"total_words[0]  # {walk.total_neg}",
                f"len(vocab)      # {walk.vocab_size}",
            ]
        )
        st.caption(
            "`movie` appears twice in positive reviews, so it contributes 2 to `total_words[1]`."
        )
