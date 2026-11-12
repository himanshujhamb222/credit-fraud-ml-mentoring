"""Naive Bayes teaching lab: Bayes → fit → Laplace → log-odds → code → trace."""

from __future__ import annotations

import streamlit as st

from naive_bayes_lessons import code_lesson, fit_lesson, goal, log_odds, predict_lesson, trace
from st_compat import button

STEPS = [
    "1. What are we calculating?",
    "2. fit(): learn counts",
    "3. predict(): word probs",
    "4. Why log-odds?",
    "5. Put math into code",
    "6. Trace predict()",
]

STEP_RENDERERS = {
    STEPS[0]: goal.render,
    STEPS[1]: fit_lesson.render,
    STEPS[2]: predict_lesson.render,
    STEPS[3]: log_odds.render,
    STEPS[4]: code_lesson.render,
    STEPS[5]: trace.render,
}


def _shift_step(delta: int) -> None:
    current = st.session_state.get("nb_step_radio", STEPS[0])
    idx = STEPS.index(current) if current in STEPS else 0
    st.session_state.nb_step_radio = STEPS[max(0, min(len(STEPS) - 1, idx + delta))]


def render() -> None:
    if st.session_state.get("nb_step_radio") not in STEPS:
        st.session_state.nb_step_radio = STEPS[0]

    st.title("Naive Bayes lab")
    st.caption(
        "Four movie reviews. Derive multinomial Naive Bayes from Bayes' theorem "
        "through fit, Laplace smoothing, log-odds, and a worked predict()."
    )

    current = st.sidebar.radio(
        "Session step",
        options=STEPS,
        index=0,
        key="nb_step_radio",
    )
    idx = STEPS.index(current) if current in STEPS else 0

    prev_col, _, next_col = st.columns([1, 4, 1])
    with prev_col:
        button(
            "Previous",
            key="nb_prev",
            disabled=idx == 0,
            on_click=_shift_step,
            args=(-1,),
        )
    with next_col:
        button(
            "Next",
            key="nb_next",
            disabled=idx == len(STEPS) - 1,
            on_click=_shift_step,
            args=(1,),
        )

    st.markdown(f"**{current}** · step {idx + 1} of {len(STEPS)}")
    STEP_RENDERERS[current]()
