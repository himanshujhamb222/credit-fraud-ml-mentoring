"""AI/ML Financial Anomaly Detection teaching and interview-prep lab."""

from __future__ import annotations

import streamlit as st

from anomaly_lessons import (
    autoencoder,
    evaluation,
    foundations,
    interview,
    iqr,
    isolation_forest,
    mahalanobis,
    one_class_svm,
    supporting,
    zscore,
)
from st_compat import button

STEPS = [
    "1. Foundations (brief)",
    "2. Z-score (Statistical)",
    "3. IQR (Statistical)",
    "4. Mahalanobis Distance",
    "5. Isolation Forest (Deep dive)",
    "6. One-Class SVM",
    "7. Autoencoders (Deep dive)",
    "8. Supporting: LOF, Graph, Temporal",
    "9. Evaluation & comparison",
    "10. Interview mastery",
]

STEP_RENDERERS = {
    STEPS[0]: foundations.render,
    STEPS[1]: zscore.render,
    STEPS[2]: iqr.render,
    STEPS[3]: mahalanobis.render,
    STEPS[4]: isolation_forest.render,
    STEPS[5]: one_class_svm.render,
    STEPS[6]: autoencoder.render,
    STEPS[7]: supporting.render,
    STEPS[8]: evaluation.render,
    STEPS[9]: interview.render,
}


def _shift_step(delta: int) -> None:
    current = st.session_state.get("anom_step_radio", STEPS[0])
    idx = STEPS.index(current) if current in STEPS else 0
    st.session_state.anom_step_radio = STEPS[max(0, min(len(STEPS) - 1, idx + delta))]


def render() -> None:
    if st.session_state.get("anom_step_radio") not in STEPS:
        st.session_state.anom_step_radio = STEPS[0]

    st.title("Financial anomaly detection lab")
    st.caption("Equations, interactive financial demos, and interview Q&As across 6 core algorithms + LOF, Graph, and Temporal.")

    st.sidebar.caption("Anomaly detection path")
    current = st.sidebar.radio(
        "Topic",
        options=STEPS,
        index=0,
        key="anom_step_radio",
    )
    idx = STEPS.index(current) if current in STEPS else 0

    prev_col, _, next_col = st.columns([1, 4, 1])
    with prev_col:
        button(
            "Previous",
            key="anom_prev",
            disabled=idx == 0,
            on_click=_shift_step,
            args=(-1,),
        )
    with next_col:
        button(
            "Next",
            key="anom_next",
            disabled=idx == len(STEPS) - 1,
            on_click=_shift_step,
            args=(1,),
        )

    st.markdown(f"**{current}** · section {idx + 1} of {len(STEPS)}")
    STEP_RENDERERS[current]()
