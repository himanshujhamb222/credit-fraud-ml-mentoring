"""CRF entity head on top of transformer outputs."""

from __future__ import annotations

import streamlit as st

from diet_lessons.ui import figure, working


def render() -> None:
    st.header("Entity recognition (CRF)")
    st.markdown(
        "Entities are **sequences** of tags, not independent guesses. "
        "DIET puts a **CRF** on the transformer outputs — same idea as the CRF lab."
    )
    why, math, tags = st.tabs(["Why CRF", "Math", "Tags"])

    with why:
        figure("diet-entities.svg")
        st.markdown(
            """
- Softmax-per-token can emit illegal patterns (`I-game` with no `B-game`).
- A CRF scores a **whole tag path** using emission + transition potentials.
- Viterbi decoding picks the best legal path at inference.
- Joint training with intent helps when intents and entities co-occur
  (paper: `play_game` almost always has `game_name`).
"""
        )

    with math:
        st.markdown(
            r"""
- Let $\boldsymbol{a} = (a_1,\ldots,a_T)$ be transformer outputs for content tokens.
- Let $\boldsymbol{y}_{\mathrm{entity}}$ be the gold tag sequence.
- Train by **negative log-likelihood** of the gold path under the CRF:
"""
        )
        st.latex(r"L_{\mathrm{E}} = L_{\mathrm{CRF}}(\boldsymbol{a},\, \boldsymbol{y}_{\mathrm{entity}})")
        st.markdown("Unnormalized path score (classic linear-chain CRF):")
        st.latex(
            r"s(\boldsymbol{a},\boldsymbol{y}) = \sum_{t=1}^{T} \psi_{\mathrm{emit}}(a_t, y_t)"
            r" + \sum_{t=1}^{T} \psi_{\mathrm{trans}}(y_{t-1}, y_t)"
        )
        st.markdown("Probability of the gold path:")
        st.latex(
            r"P(\boldsymbol{y}\mid\boldsymbol{a})"
            r" = \frac{\exp\, s(\boldsymbol{a},\boldsymbol{y})}"
            r"{\sum_{\boldsymbol{y}'} \exp\, s(\boldsymbol{a},\boldsymbol{y}')}"
        )
        working(
            [
                "L_E = - log P(y_gold | a)",
                "Partition function Z = sum over all tag paths (forward algorithm).",
                "At test time: Viterbi finds argmax_y s(a, y).",
            ]
        )

    with tags:
        st.markdown(
            """
Example utterance **play ping pong** with entity `game_name`:

| Token | BIO tags | BILOU tags (stricter eval) |
|-------|----------|----------------------------|
| play | O | O |
| ping | B-game_name | B-game_name |
| pong | I-game_name | L-game_name |

- Paper metrics on ATIS/SNIPS use **exact span match** with BILOU.
- NLU-Benchmark allows **overlapping** spans with matching labels.
- Ablation: training entities alone drops F1 (~86% → ~83%) — joint training helps entities most.
"""
        )
