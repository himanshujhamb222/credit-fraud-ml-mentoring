"""What BERT does, and how the mask is trained."""

from __future__ import annotations

import streamlit as st

from st_compat import dataframe
from transformer_lab.bert import head_lines, loss_lines, mask_draw
from transformer_lessons.ui import figure, working


def render() -> None:
    st.header("BERT")
    st.markdown(
        "BERT is an encoder. "
        "It reads a whole sentence in both directions. "
        "It is used when you want a meaning, a label, or a match."
    )
    what, train, tune = st.tabs(["What it does", "Training", "Fine-tune"])

    with what:
        st.markdown(
            """
- Give it a sentence. It builds one vector per word, plus a vector for the whole sentence.
- A classifier can read that vector and say positive or negative.
- A search system can compare two vectors.
- A tagger can mark which word is a name or a date.
- It does not write the next word. A decoder does that job.
"""
        )

    with train:
        figure("bert-mlm.svg")
        frame = mask_draw()
        st.markdown(
            "Training hides about 15% of the words and asks for the originals. "
            "Of the hidden words, 80% become `[MASK]`, 10% become a random word, and 10% stay. "
            "The random and unchanged cases matter because `[MASK]` is absent when you later use the model."
        )
        shown = " ".join(str(word) for word in frame["shown"])
        st.markdown(f"Sentence the model sees, seed 6: `{shown}`")
        dataframe(frame)
        working(loss_lines(frame))
        st.markdown(
            "Early BERT also guessed whether sentence B followed sentence A. "
            "Later encoders leave that second task out."
        )

    with tune:
        st.markdown(
            """
- Pretraining teaches the encoder to fill in words.
- Fine-tuning keeps that encoder.
- You add a small new layer for the task you care about.
- The loss is now the task loss. The fill-in loss is finished.
"""
        )
        working(head_lines())
