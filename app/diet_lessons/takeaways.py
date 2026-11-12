"""Paper results and practical takeaways."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from diet_lessons.ui import figure
from st_compat import dataframe


def render() -> None:
    st.header("Results and takeaways")
    st.markdown(
        "Why DIET matters for chatbot NLU: **accuracy without a giant fine-tune**, "
        "and a modular recipe you can swap features in and out."
    )
    numbers, compare, remember = st.tabs(["Numbers", "vs fine-tuned BERT", "Remember"])

    with numbers:
        figure("diet-overview.svg")
        st.markdown(
            """
**NLU-Benchmark** (hard multi-domain set; micro F1):

| Model | Intent F1 | Entity F1 |
|-------|-----------|-----------|
| HERMIT (prior SOTA) | 87.6 | 84.7 |
| DIET sparse + ConveRT (no mask) | **90.2** | **86.0** |

- Sparse-only + mask loss is already competitive (~88 intent / ~85 entity).
- ConveRT (chat-trained) beats frozen BERT prose embeddings on this task.
- ATIS / SNIPS: sparse-only DIET lands within ~1–2% of Joint BERT.
"""
        )
        ablation = pd.DataFrame(
            [
                {"sparse": "✓", "dense": "—", "mask": "✓", "intent F1": 88.2, "entity F1": 85.1},
                {"sparse": "✓", "dense": "GloVe", "mask": "✓", "intent F1": 89.1, "entity F1": 86.0},
                {"sparse": "✓", "dense": "BERT", "mask": "—", "intent F1": 88.5, "entity F1": 85.3},
                {"sparse": "✓", "dense": "ConveRT", "mask": "—", "intent F1": 90.2, "entity F1": 86.0},
            ]
        )
        st.markdown("Selected ablation rows (Table 3, rounded):")
        dataframe(ablation, hide_index=True)

    with compare:
        st.markdown(
            """
Same DIET shell, different dense features on NLU-Benchmark:

| Setup | Intent F1 | Entity F1 | Train time (10 folds) |
|-------|-----------|-----------|------------------------|
| Fine-tuned BERT inside DIET | 89.7 | 85.7 | ~60 hours |
| Sparse + frozen ConveRT | **90.2** | **86.0** | ~10 hours |

- Best DIET set **beats** fine-tuned BERT and is about **6× faster** to train.
- Lesson: for intent + entity dialogue NLU, a huge LM is not automatically better.
- Plug-and-play features matter more than "always fine-tune BERT".
"""
        )

    with remember:
        st.markdown(
            r"""
1. **Dual job** — one encoder, intent head + entity CRF.
2. **Features** — sparse word/char ± dense (GloVe / BERT / ConveRT).
3. **Body** — 2-layer transformer + relative position attention.
4. **$L_{\mathrm{E}}$** — CRF NLL on token tags.
5. **$L_{\mathrm{I}}$** — dot-product ranking in a shared intent space.
6. **$L_{\mathrm{M}}$** — optional masked-token reconstruction (same loss shape).
7. **$L_{\mathrm{total}} = L_{\mathrm{I}} + L_{\mathrm{E}} + L_{\mathrm{M}}$** — turn pieces on/off as needed.
8. Paper: [arxiv.org/pdf/2004.09936](https://arxiv.org/pdf/2004.09936) · code: [RasaHQ/DIET-paper](https://github.com/RasaHQ/DIET-paper)
"""
        )
        st.info(
            "If someone says **DUET**, they usually mean this architecture — "
            "the published name is **DIET** (Dual Intent and Entity Transformer)."
        )

    st.divider()
    st.subheader("Paper architecture (end map)")
    st.caption(
        "Original **Figure 1** from the paper again — same image as lesson 1."
    )
    figure("diet-scheme.png")
    st.latex(r"L_{\mathrm{total}} = L_{\mathrm{I}} + L_{\mathrm{E}} + L_{\mathrm{M}}")
    st.markdown(
        r"""
**How to read Figure 1**

1. Tokens (`play` · `ping` · `pong` · `__CLS__`)
2. Sparse and/or dense features
3. Shared feed-forward layers (concat → width 256)
4. 2-layer transformer with relative position attention
5. Three heads: intent ($L_{\mathrm{I}}$), entity CRF ($L_{\mathrm{E}}$), optional mask ($L_{\mathrm{M}}$)
"""
    )
