"""Temperature, top-k, and top-p on one toy distribution."""

from __future__ import annotations

import streamlit as st

from st_compat import dataframe
from transformer_lab.sampling import temperature_lines, temperature_table, top_k_table, top_p_table
from transformer_lessons.charts import bars
from transformer_lessons.ui import working


def render() -> None:
    st.header("Sampling")
    st.markdown(
        "After training, the model has a score for every next word. "
        "These knobs turn the scores into a choice. "
        "The scores below are the toy logits from the GPT lesson."
    )
    temp, topk, topp = st.tabs(["Temperature", "Top-k", "Top-p"])

    with temp:
        st.code("P(word) = exp(logit / T) / sum exp(logit / T)", language="text")
        temperature = st.slider("T", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
        table = temperature_table(temperature)
        working(temperature_lines(temperature))
        dataframe(table)
        bars(
            table["token"].tolist(),
            [float(value) for value in table["probability"]],
            "Probabilities after dividing by T",
            "temp_bars",
        )
    with topk:
        st.code(
            "S = k largest logits\n"
            "P(word) = exp(logit) / sum_S exp(logit)    if word is in S\n"
            "P(word) = 0                                otherwise",
            language="text",
        )
        k = st.slider("k", min_value=1, max_value=10, value=3, step=1)
        table, lines = top_k_table(int(k))
        working(lines)
        dataframe(table)
        bars(
            table["token"].tolist(),
            [float(value) for value in table["probability"]],
            "Probabilities after keeping k words",
            "topk_bars",
        )

    with topp:
        st.code(
            "S = smallest set with sum of P >= p\n"
            "P'(word) = P(word) / sum_S P(word)    if word is in S\n"
            "P'(word) = 0                          otherwise",
            language="text",
        )
        p = st.slider("p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        table, lines = top_p_table(float(p))
        working(lines)
        dataframe(table)
        bars(
            table["token"].tolist(),
            [float(value) for value in table["after renorm"]],
            "Probabilities after the top-p cutoff",
            "topp_bars",
        )
        st.markdown(
            "Top-p keeps the smallest set of words whose probabilities add up to at least p. "
            "A newer cutoff, min-p, keeps words that are at least some fraction of the winner."
        )
