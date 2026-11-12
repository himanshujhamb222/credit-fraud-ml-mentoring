"""Next-word training, and why generation is a different loop."""

from __future__ import annotations

import streamlit as st

from st_compat import dataframe
from transformer_lab.gpt import generation_lines, loss_lines, shift_frame
from transformer_lessons.ui import figure, working


def render() -> None:
    st.header("GPT")
    st.markdown(
        "GPT is a decoder. "
        "It scores the next word. "
        "The training pass and the writing loop are different."
    )
    what, train, generate = st.tabs(["What it does", "Training", "Generation"])

    with what:
        figure("causal-attention.svg")
        st.markdown(
            """
- The model reads words from left to right.
- At each position it scores every word in the vocabulary.
- The causal mask blocks future words.
- The same block as BERT is used, with that mask, and without cross-attention.
"""
        )

    with train:
        st.markdown(
            "The sentence is **the cat sat on**. "
            "The input is **the cat sat**. "
            "The targets are **cat sat on**. "
            "One forward pass scores all three guesses."
        )
        dataframe(shift_frame())
        working(loss_lines())

    with generate:
        st.markdown(
            "At generation time the later words do not exist yet. "
            "The model writes one word, appends it, and repeats."
        )
        working(generation_lines())
