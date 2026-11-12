"""Optional mask loss and the total multi-task objective."""

from __future__ import annotations

import streamlit as st

from diet_lab.walks import mask_loss_walk
from diet_lessons.ui import figure, working


def render() -> None:
    st.header("Masking and total loss")
    st.markdown(
        "A third objective asks the model to **rebuild masked tokens**. "
        "Then all losses add into one multi-task train signal."
    )
    mask, total, batch = st.tabs(["Mask loss", "Total loss", "Batching"])
    walk = mask_loss_walk()

    with mask:
        figure("diet-mask.svg")
        st.markdown(
            r"""
- Inspired by BERT's masked language model, but lighter.
- Select ~**15%** of input tokens.
- Of those: **70%** → `__MASK__`, **10%** → random token, **20%** keep original.
- Embed $a_{\mathrm{MASK}}$ and compare to token label embeddings with the **same**
  dot-product loss as intent.
"""
        )
        st.latex(
            r"L_{\mathrm{M}} = -\left\langle"
            r" S_{\mathrm{M}}^{+} - \log\!\left("
            r"e^{S_{\mathrm{M}}^{+}} + \sum_{\Omega_{\mathrm{M}}^{-}} e^{S_{\mathrm{M}}^{-}}"
            r"\right)"
            r"\right\rangle"
        )
        st.markdown(
            f"Worked case: hide **{walk.tokens[walk.masked_index]}** "
            f"as `{walk.masked_as}`."
        )
        working(walk.lines)
        st.markdown(
            """
- Hypothesis in the paper: mask loss **regularizes** and forces more general text features,
  not only task-discriminative ones.
- Ablation: helps sparse / GloVe setups; with ConveRT/BERT it can slightly hurt —
  so it is **optional**.
"""
        )

    with total:
        st.markdown("Train by minimizing the sum (Eq. 4):")
        st.latex(r"L_{\mathrm{total}} = L_{\mathrm{I}} + L_{\mathrm{E}} + L_{\mathrm{M}}")
        st.markdown(
            r"""
| Symbol | Head | What it teaches |
|--------|------|-----------------|
| $L_{\mathrm{I}}$ | Intent | Whole-utterance label |
| $L_{\mathrm{E}}$ | Entity CRF | Token / span tags |
| $L_{\mathrm{M}}$ | Mask | Reconstruct hidden words |

- Any term can be **turned off** in config.
- Best NLU-Benchmark model in the paper: sparse + ConveRT, **no** mask loss.
- Single-task ablations: joint training helps **entities** most; intent alone can be slightly higher.
"""
        )

    with batch:
        st.markdown(
            """
- **Balanced batching** — rare intents still appear often enough (class imbalance).
- **Increasing batch size** during training — another regularizer
  (Smith et al., 2017: grow batch instead of only decaying LR).
- Optimizer: Adam, LR 0.001; ~200 epochs on NLU-Benchmark fold ≈ 1 hour on a K80-class GPU.
- Inference ≈ **80 ms** per utterance in the paper's setup.
"""
        )
