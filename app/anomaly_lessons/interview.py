"""Lesson 10: Interview mastery — crisp scenario drill & formula cheat sheet."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.interview_bank import (
    CORE_QA_BY_TOPIC,
    CROSS_ALGORITHM_SCENARIOS,
    all_interview_questions,
)
from anomaly_lessons.ui import lesson_header, render_interview_qa
from st_compat import dataframe


def render() -> None:
    lesson_header(
        "10. Interview mastery",
        "Cross-algorithm comparisons, financial scenario answers, equation cheat sheet, and rapid drill.",
        priority_badge=":red-badge[Highest priority — Interview readiness]",
        interview_focus=[
            "Trade-offs: Isolation Forest vs OC-SVM; Z-score vs IQR vs Mahalanobis.",
            "Failure modes: AE overfitting, skewed amounts, scarce labels, different score scales.",
            "Hybrid production stack: rules + percentile-ranked detectors + supervised fraud model.",
        ],
    )

    tab_cross, tab_cheat, tab_drill = st.tabs(
        ["9 core scenarios", "Equation cheat sheet", "Filterable Q&A drill"]
    )

    with tab_cross:
        render_interview_qa(
            CROSS_ALGORITHM_SCENARIOS,
            key_prefix="qa_cross_all",
            heading="Cross-algorithm & scenario questions",
        )

    with tab_cheat:
        cheat = pd.DataFrame(
            [
                {"Algorithm": "Z-score", "Core equation": "(x - mu) / sigma", "Scaling": "No", "Interview hotspot": "Skew, 0% breakdown (masking), log1p / MAD"},
                {"Algorithm": "IQR", "Core equation": "[Q1 - k*IQR, Q3 + k*IQR]", "Scaling": "No", "Interview hotspot": "25% breakdown, k=1.5 (~2.7 sigma), IQR=0 trap"},
                {"Algorithm": "Mahalanobis", "Core equation": "sqrt((x-mu)^T Sigma^-1 (x-mu))", "Scaling": "No", "Interview hotspot": "Correlation ellipse, chi2(p) cutoff, singular Sigma"},
                {"Algorithm": "Isolation Forest", "Core equation": "2^(-E[h(x)] / c(n))", "Scaling": "No", "Interview hotspot": "Short path, psi=256 (swamping/masking), O(T*psi*log psi)"},
                {"Algorithm": "One-Class SVM", "Core equation": "sum alpha_i K(x, x_i) - rho", "Scaling": "Yes", "Interview hotspot": "Dual nu bound, RBF gamma, O(n^2)-O(n^3) limit"},
                {"Algorithm": "Autoencoder", "Core equation": "mean((x - x_hat)^2)", "Scaling": "Yes", "Interview hotspot": "Bottleneck k<d, normal-only train, identity overfit, per-feature MSE"},
            ]
        )
        dataframe(cheat, hide_index=True)

    with tab_drill:
        all_qas = all_interview_questions()
        topics = list(CORE_QA_BY_TOPIC.keys()) + ["Cross-algorithm & Scenarios"]
        q_types = ["Conceptual", "Mathematical", "Numerical", "Scenario"]

        f1, f2 = st.columns(2)
        with f1:
            selected_topics = st.multiselect("Topics", options=topics, default=topics, key="drill_topics")
        with f2:
            if hasattr(st, "pills"):
                selected_types = st.pills("Question types", options=q_types, default=q_types, selection_mode="multi", key="drill_types")
            else:
                selected_types = st.multiselect("Question types", options=q_types, default=q_types, key="drill_types")

        filtered = [qa for qa in all_qas if qa.topic in selected_topics and (not selected_types or qa.q_type in selected_types)]
        st.caption(f"Showing **{len(filtered)}** / {len(all_qas)} questions.")
        if filtered:
            render_interview_qa(filtered, key_prefix="qa_drill_filtered", heading="Practice cards")
        else:
            st.warning("Select at least one topic and question type.", icon=":material/filter_alt:")
