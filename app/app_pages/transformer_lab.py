"""Transformer teaching lab: six short lessons."""

from __future__ import annotations

import streamlit as st

from transformer_lessons import attention, bert, block, gpt, sampling, shapes

LESSONS = {
    "1. Three shapes": shapes.render,
    "2. Attention": attention.render,
    "3. Inside the block": block.render,
    "4. BERT": bert.render,
    "5. GPT": gpt.render,
    "6. Sampling": sampling.render,
}


def render() -> None:
    st.sidebar.caption("Transformer path")
    lesson = st.sidebar.radio("Lesson", list(LESSONS), key="tx_lesson")
    LESSONS[str(lesson)]()
