"""Dot-product intent classification head."""

from __future__ import annotations

import streamlit as st

from diet_lab.walks import intent_loss_walk
from diet_lessons.ui import figure, working
from st_compat import dataframe
from transformer_lessons.charts import bars


def render() -> None:
    st.header("Intent classification")
    st.markdown(
        "Intent is a **ranking** problem in a shared embedding space — "
        "not a plain softmax over a fixed classifier weight matrix."
    )
    idea, math, toy = st.tabs(["Idea", "Math", "Worked numbers"])
    walk = intent_loss_walk()

    with idea:
        figure("diet-intent.svg")
        st.markdown(
            r"""
- Take the transformer output for `__CLS__`: $a_{\mathrm{CLS}}$.
- Embed it and every intent label into the **same** space $h \in \mathbb{R}^{20}$.
- Push $h_{\mathrm{CLS}}$ toward the **true** intent embedding.
- Push it away from **negative** intent embeddings (in-batch / sampled).
- At inference: score every intent with a **dot product**; pick the max.
"""
        )

    with math:
        st.markdown("Embeddings:")
        st.latex(
            r"h_{\mathrm{CLS}} = E(a_{\mathrm{CLS}}),\quad"
            r"h_{\mathrm{intent}} = E(y_{\mathrm{intent}})"
        )
        st.markdown("Similarities:")
        st.latex(
            r"S_{\mathrm{I}}^{+} = h_{\mathrm{CLS}}^{\top} h_{\mathrm{intent}}^{+},"
            r"\quad"
            r"S_{\mathrm{I}}^{-} = h_{\mathrm{CLS}}^{\top} h_{\mathrm{intent}}^{-}"
        )
        st.markdown("Dot-product loss (Eq. 2 in the paper):")
        st.latex(
            r"L_{\mathrm{I}} = -\left\langle"
            r" S_{\mathrm{I}}^{+} - \log\!\left("
            r"e^{S_{\mathrm{I}}^{+}} + \sum_{\Omega_{\mathrm{I}}^{-}} e^{S_{\mathrm{I}}^{-}}"
            r"\right)"
            r"\right\rangle"
        )
        st.markdown(
            r"""
- $\langle\cdot\rangle$ averages over examples in the batch.
- $\Omega_{\mathrm{I}}^{-}$ is the set of negative intent labels.
- Same family of loss as StarSpace / response-selection work cited in the paper.
"""
        )

    with toy:
        st.markdown(
            r"Toy space $\mathbb{R}^3$ (paper uses $\mathbb{R}^{20}$). "
            "True intent is **play_game**."
        )
        working(walk.lines)
        dataframe(walk.frame, hide_index=True)
        bars(
            walk.intents,
            [walk.similarities[i] for i in walk.intents],
            "Dot-product similarities (higher = closer)",
            "diet_intent_sims",
        )
        st.caption(f"Loss for this example: **{walk.loss:.3f}**")
