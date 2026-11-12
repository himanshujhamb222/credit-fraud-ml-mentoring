"""Where attention sits inside a transformer block."""

from __future__ import annotations

import streamlit as st

from transformer_lab.block import head_lines, residual_lines
from transformer_lessons.ui import figure, working


def render() -> None:
    st.header("Inside the block")
    st.markdown(
        "Attention is the mixing step. "
        "The rest of the block keeps the vector stable and lets you stack many blocks."
    )
    one, heads, link = st.tabs(["One block", "Many heads", "The link"])

    with one:
        figure("full-transformer.svg")
        st.markdown(
            """
- Attention mixes words.
- A residual add puts the earlier vector back.
- A small network then changes each word on its own.
- Norm keeps the size of the vector under control.
"""
        )
        working(residual_lines())

    with heads:
        figure("multi-head-attention.svg")
        st.markdown(
            """
- One head is one lookup.
- Several heads run at the same time.
- Each head can notice a different link.
- They are joined, then mixed by `W_o`.
"""
        )
        working(head_lines())

    with link:
        figure("encoder-decoder.svg")
        st.markdown(
            """
- Cross-attention is the only wire from encoder to decoder.
- The decoder supplies the query.
- The encoder supplies the keys and the values.
- GPT has no encoder, so that wire is left out.
- GPT keeps masked self-attention and the small network.
"""
        )
