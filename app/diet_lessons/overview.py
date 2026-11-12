"""What DIET is and the dual job it solves."""

from __future__ import annotations

import streamlit as st

from diet_lab.walks import EXAMPLE_ENTITY, EXAMPLE_INTENT, EXAMPLE_UTTERANCE, pipeline_steps
from diet_lessons.ui import figure


def render() -> None:
    st.header("What is DIET?")
    st.markdown(
        "**DIET** = **D**ual **I**ntent and **E**ntity **T**ransformer "
        "([Bunk et al., 2020](https://arxiv.org/pdf/2004.09936)). "
        "One small network does **two** dialogue-NLU jobs at once."
    )

    st.subheader("Paper architecture at a glance")
    st.caption(
        "Original **Figure 1** from [Bunk et al., 2020](https://arxiv.org/pdf/2004.09936) — "
        "keep this map in mind as you walk the later lessons."
    )
    figure("diet-scheme.png")
    st.latex(r"L_{\mathrm{total}} = L_{\mathrm{I}} + L_{\mathrm{E}} + L_{\mathrm{M}}")

    what, example, flow = st.tabs(["The dual job", "Running example", "Pipeline"])

    with what:
        st.markdown(
            """
- Chatbots need to know **what the user wants** (intent) and **which words are slots** (entities).
- Training two separate models can pass errors between them.
- DIET is a **multi-task** model: one transformer body, two heads.
- It is **lightweight**: sparse features alone already beat older SOTA; dense embeddings are optional plug-ins.
- Name note: people sometimes say "DUET" — the paper name is **DIET**.
"""
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Intent** — one label for the whole sentence")
            st.markdown("- `play_game`, `set_alarm`, `book_flight`, …")
        with c2:
            st.markdown("**Entity** — a tag for each token / span")
            st.markdown("- `game_name`, `date`, `location`, …")

    with example:
        figure("diet-overview.svg")
        st.markdown(
            f"""
Utterance: **`{EXAMPLE_UTTERANCE}`**

| Piece | Value |
|-------|--------|
| Intent | `{EXAMPLE_INTENT}` |
| Entity | `{EXAMPLE_ENTITY[0]}` = "{EXAMPLE_ENTITY[1]}" |

- Tokens: `play` · `ping` · `pong` · `__CLS__`
- The model reads the whole sequence once.
- Token vectors feed the **entity CRF**.
- The `__CLS__` vector feeds **intent ranking**.
"""
        )

    with flow:
        st.markdown("Eight short stages. Later lessons unpack each one.")
        for title, body in pipeline_steps():
            st.markdown(f"**{title}.** {body}")
