"""Encoder, decoder, and encoder-decoder."""

from __future__ import annotations

import streamlit as st

from transformer_lessons.ui import figure


def render() -> None:
    st.header("Three shapes")
    st.markdown(
        "A transformer is a stack of blocks. "
        "Each block mixes the words, then runs a small network. "
        "People use that block in three ways."
    )
    encoder, decoder, both = st.tabs(["Encoder", "Decoder", "Encoder and decoder"])

    with encoder:
        figure("rnn-vs-transformer.svg")
        st.markdown(
            """
- An older network reads one word, then the next.
- A transformer mixes every word in one step.
- BERT is an encoder. Every word can look at every other word.
- That helps when you want the meaning of a whole sentence.
- Classify it. Search with it. Pull a name out of it.
"""
        )

    with decoder:
        figure("causal-attention.svg")
        st.markdown(
            """
- GPT is a decoder. It guesses the next word.
- A word may look at itself and at earlier words.
- Later words are blocked, so the answer is not given away.
- Training still reads the whole sentence in one pass.
- Writing a reply adds one new word at a time.
"""
        )

    with both:
        figure("encoder-decoder.svg")
        st.markdown(
            """
- Translation has two jobs: read the source, write the target.
- The encoder reads the source in both directions.
- The decoder writes the target one word at a time.
- Cross-attention is the link. The decoder's question looks up the encoder's memory.
- The encoder runs once. Each new target word looks up that same memory.
"""
        )
