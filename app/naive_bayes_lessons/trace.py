"""Step-by-step predict(\"good movie\") with live try-your-own."""

from __future__ import annotations

import streamlit as st

from naive_bayes_lab.model import fit, predict
from naive_bayes_lab.walks import LABELS, TEXTS, predict_trace
from naive_bayes_lessons.ui import working


def render() -> None:
    st.header('Trace predict("good movie")')
    st.markdown(
        "`fit()` learned the counts. `predict()` uses them to accumulate evidence. "
        "Walk the arithmetic for **good movie**, then try other reviews."
    )

    default, live = st.tabs(["Worked example", "Try a review"])

    with default:
        trace = predict_trace("good movie")
        st.subheader("Prior")
        working(trace.prior_lines)
        st.latex(r"score_0=\log\frac{0.5}{0.5}=0")

        for step in trace.steps:
            st.subheader(f"Word: `{step.word}`")
            working(step.lines)
            if step.word == "good":
                st.latex(
                    r"\log\frac{P(good\mid +)}{P(good\mid -)}"
                    r"=\log\frac{2/9}{2/9}=0"
                )
            elif step.word == "movie":
                st.latex(
                    r"\log\frac{P(movie\mid +)}{P(movie\mid -)}"
                    r"=\log\frac{3/9}{2/9}=\log(1.5)\approx 0.405"
                )

        st.subheader("Decision")
        working(trace.summary_lines)
        st.latex(r"score=0+0+0.405=0.405 \Rightarrow \text{positive}")
        st.markdown(
            "The whole algorithm follows from Bayes' theorem, "
            "conditional independence, and taking logs so products become sums."
        )

    with live:
        model = fit(TEXTS, LABELS)
        text = st.text_input(
            "Review to score",
            value="good movie",
            key="nb_live_review",
        )
        alpha = st.slider("Laplace α", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        if text.strip():
            label, score = predict(text, model, alpha=alpha)
            walk = predict_trace(text, alpha=alpha)
            st.metric("Prediction", label)
            st.metric("Log-odds score", f"{score:.4f}")
            working(walk.prior_lines)
            for step in walk.steps:
                with st.expander(f"`{step.word}` → score {step.score_after:.4f}"):
                    working(step.lines)
            if not walk.steps:
                st.info("No vocabulary words found in this review (after lowercasing / split).")
            working(walk.summary_lines)
        st.caption("Known training phrases: good movie · great movie · bad movie · not good")
