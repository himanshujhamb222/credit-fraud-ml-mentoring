"""Sparse and dense featurization before the transformer."""

from __future__ import annotations

import streamlit as st

from diet_lab.walks import featurize_walk
from diet_lessons.ui import figure, working
from st_compat import dataframe


def render() -> None:
    st.header("Featurization")
    st.markdown(
        "Before any transformer, each token becomes a vector. "
        "DIET mixes **sparse** counts and optional **dense** embeddings."
    )
    sparse, dense, fuse = st.tabs(["Sparse", "Dense", "Fuse"])
    walk = featurize_walk()

    with sparse:
        st.markdown(
            """
- **Word one-hot.** Which vocabulary word is this token?
- **Character n-grams** (n ≤ 5), multi-hot. Catches typos and morphology.
- Character n-grams are noisy → **dropout** on sparse features.
- No huge pretrain required. Sparse-only DIET is already strong on NLU-Benchmark.
"""
        )
        st.latex(r"x_{\mathrm{sparse}} \in \{0,1\}^V \quad\text{(word)} \;+\; \{0,1\}^C \quad\text{(char n-grams)}")

    with dense:
        st.markdown(
            """
- Optional **pre-trained** vectors: GloVe, BERT, or ConveRT.
- Weights stay **frozen** in the best paper setup (no full fine-tune needed).
- `__CLS__` seed:
  - **ConveRT** → sentence embedding of the utterance
  - **BERT** → BERT's `[CLS]` output
  - **GloVe** → mean of token embeddings
- ConveRT was trained on chat data; that often beats frozen BERT prose embeddings here.
"""
        )
        st.latex(r"x_{\mathrm{dense}} = E_{\mathrm{pretrained}}(\mathrm{token})")

    with fuse:
        figure("diet-featurize.svg")
        st.markdown(
            """
1. Shared feed-forward maps sparse → same width as dense.
2. **Concatenate** projected sparse + dense.
3. Another shared feed-forward maps to transformer width (paper: **256**).
4. Shared weights across tokens keep the model small.
"""
        )
        st.latex(
            r"z_t = W_2\,\big[\,W_1 x_{\mathrm{sparse},t}\;\|\; x_{\mathrm{dense},t}\,\big]"
        )
        st.markdown("Toy numbers for `play ping pong __CLS__` (dims shrunk for the page):")
        working(walk.lines)
        dataframe(walk.projected, hide_index=True)
