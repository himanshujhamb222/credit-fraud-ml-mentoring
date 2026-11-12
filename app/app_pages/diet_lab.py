"""DIET teaching lab: Dual Intent and Entity Transformer, step by step."""

from __future__ import annotations

import streamlit as st

from diet_lessons import encoder, entities, featurize, intents, masking, overview, takeaways

LESSONS = {
    "1. What is DIET?": overview.render,
    "2. Featurization": featurize.render,
    "3. Transformer body": encoder.render,
    "4. Entity CRF": entities.render,
    "5. Intent ranking": intents.render,
    "6. Mask + total loss": masking.render,
    "7. Results": takeaways.render,
}


def render() -> None:
    st.sidebar.caption("DIET path · [paper](https://arxiv.org/pdf/2004.09936)")
    lesson = st.sidebar.radio("Lesson", list(LESSONS), key="diet_lesson")
    LESSONS[str(lesson)]()
