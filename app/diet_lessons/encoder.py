"""Two-layer transformer with relative position attention."""

from __future__ import annotations

import streamlit as st

from diet_lessons.ui import figure


def render() -> None:
    st.header("Transformer body")
    st.markdown(
        "A short **2-layer** transformer turns token features into contextual vectors. "
        "Same idea as the Attention lesson — every token can look at every other token."
    )
    role, relative, out = st.tabs(["Role", "Relative position", "Outputs"])

    with role:
        figure("diet-transformer.svg")
        st.markdown(
            r"""
- Input: sequence of fused vectors $z_1,\ldots,z_T$ (including `__CLS__`).
- Each layer: multi-head self-attention + feed-forward + residuals (Vaswani et al., 2017).
- Depth is deliberately small — dialogue NLU utterances are short; speed matters.
- Output sequence $\boldsymbol{a} = (a_1,\ldots,a_T)$ goes to the two task heads.
"""
        )
        st.latex(r"\boldsymbol{a} = \mathrm{Transformer}_2(\boldsymbol{z})")

    with relative:
        st.markdown(
            r"""
- Absolute position embeddings say "I am token #3".
- **Relative position attention** (Shaw et al., 2018) says "you are *k* steps away from me".
- That helps entity spans: "ping" next to "pong" matters more than absolute index.
- Formally, attention logits add a learned bias that depends on the distance $i-j$.
"""
        )
        st.latex(
            r"e_{ij} = \frac{(x_i W^Q)(x_j W^K)^\top}{\sqrt{d}} + a_{ij}^{\mathrm{rel}}"
        )

    with out:
        st.markdown(
            r"""
| Vector | Used by |
|--------|---------|
| $a_t$ for each content token | Entity CRF |
| $a_{\mathrm{CLS}}$ | Intent embedding / ranking |
| $a_{\mathrm{MASK}}$ (masked positions) | Optional mask loss |

- One forward pass. Three losses can share the same $\boldsymbol{a}$.
- Any loss can be switched off — modular by design.
"""
        )
