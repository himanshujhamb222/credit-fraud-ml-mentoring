"""What we want to calculate: Bayes posterior from a review."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from naive_bayes_lab.walks import LABELS, TEXTS
from st_compat import dataframe


def render() -> None:
    st.header("What are we trying to calculate?")
    st.markdown(
        "Given a new review, say **good movie**, decide whether it is "
        "positive or negative. Same four training reviews throughout the lab."
    )

    dataframe(
        pd.DataFrame(
            {
                "Review": TEXTS,
                "Sentiment": ["Positive (1)" if y == 1 else "Negative (0)" for y in LABELS],
            }
        ),
        hide_index=True,
    )

    st.subheader("Compare the two posteriors")
    st.markdown("From probability theory we want:")
    st.latex(r"P(+\mid review) \quad \text{and} \quad P(-\mid review)")

    st.markdown("Bayes' theorem:")
    st.latex(
        r"P(+\mid review)=\frac{P(review\mid +)\,P(+)}{P(review)}"
    )
    st.latex(
        r"P(-\mid review)=\frac{P(review\mid -)\,P(-)}{P(review)}"
    )

    st.markdown(
        r"The denominator \(P(review)\) is the same for both classes, "
        r"so drop it when comparing. Instead compare:"
    )
    st.latex(
        r"P(review\mid +)\,P(+) \quad\text{against}\quad P(review\mid -)\,P(-)"
    )
    st.markdown(
        "That comparison is the mathematical foundation of the classifier. "
        "Next lessons estimate each piece from the four reviews."
    )
