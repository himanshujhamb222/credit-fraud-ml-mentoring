"""Worked attention matrix, scaling, and the causal mask."""

from __future__ import annotations

import streamlit as st

from st_compat import dataframe
from transformer_lab.attention import attention_walk, causal_walk, scaling_walk, vector_frame
from transformer_lessons.charts import grouped_bars, heatmap
from transformer_lessons.ui import working


def render() -> None:
    st.header("Attention")
    st.markdown(
        "Attention is a soft lookup. "
        "Each word asks a question, checks every word's label, and blends the stored values."
    )
    lookup, matrix, scale, mask = st.tabs(
        ["Lookup", "The matrix", "Why √d_k", "Full vs causal"]
    )
    walk = attention_walk()

    with lookup:
        st.markdown(
            """
- **Query.** What am I looking for?
- **Key.** What do I contain?
- **Value.** What do I pass on if I am picked?
- The score is a dot product. A larger score means a closer match.
- Softmax turns one row of scores into weights that add up to 1.
- The output is those weights times the values.
"""
        )
        st.code("Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V", language="text")

    with matrix:
        st.markdown("Sentence: **the cat sat**. Each word is two numbers. `d_k = 2`.")
        q_col, k_col, v_col = st.columns(3)
        with q_col:
            st.markdown("**Q**")
            dataframe(vector_frame(walk.q, walk.tokens))
        with k_col:
            st.markdown("**K**")
            dataframe(vector_frame(walk.k, walk.tokens))
        with v_col:
            st.markdown("**V**")
            dataframe(vector_frame(walk.v, walk.tokens))
        st.markdown("One cell, by hand. **cat** looks up **cat**.")
        working(walk.dot_lines)
        heatmap(walk.scores, walk.tokens, "Scores. Rows are queries. Columns are keys.", "attn_scores")
        st.markdown("Divide that row by √d_k.")
        working(walk.scale_lines)
        heatmap(walk.scaled, walk.tokens, "Scores after dividing by √d_k", "attn_scaled")
        st.markdown("Turn the **cat** row into weights.")
        working(walk.softmax_lines)
        heatmap(walk.weights, walk.tokens, "Attention weights. Each row sums to 1.", "attn_weights", weights=True)
        st.markdown("Blend the values with those weights.")
        working(walk.output_lines)
        dataframe(vector_frame(walk.output, walk.tokens))

    with scale:
        scaled = scaling_walk()
        st.markdown(
            "Dot products grow with `d_k`. "
            "Softmax of a huge row puts almost all the weight on one word. "
            "The fix is to divide by √d_k."
        )
        working(scaled.why_lines)
        working(scaled.divide_lines)
        left, right = st.columns(2)
        with left:
            st.markdown("**Before scaling**")
            working(scaled.raw_softmax_lines)
        with right:
            st.markdown("**After scaling**")
            working(scaled.scaled_softmax_lines)
        grouped_bars(
            walk.tokens,
            scaled.raw_weights.tolist(),
            scaled.scaled_weights.tolist(),
            "scale_bars",
        )

    with mask:
        causal = causal_walk(walk)
        st.markdown(
            "An encoder keeps every cell. "
            "A decoder adds `-inf` above the diagonal, then softmax. "
            "`exp(-inf)` is 0, so future words get no weight."
        )
        full_col, causal_col = st.columns(2)
        with full_col:
            heatmap(causal.scaled, walk.tokens, "Full scores", "mask_full_scores")
        with causal_col:
            heatmap(causal.masked, walk.tokens, "Causal scores", "mask_causal_scores")
        working(causal.lines)
        heatmap(
            causal.causal_weights,
            walk.tokens,
            "Weights after the causal mask",
            "mask_weights",
            weights=True,
        )
