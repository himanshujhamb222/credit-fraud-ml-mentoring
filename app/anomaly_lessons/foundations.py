"""Lesson 1: Foundations (short, scannable, equation & diagram first)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lessons.ui import lesson_header
from st_compat import dataframe, mermaid_chart


def render() -> None:
    lesson_header(
        "1. Foundations",
        "Anomaly ≠ fraud. Anomaly detectors learn normal behavior to catch unseen attacks.",
        priority_badge=":gray-badge[Supporting — brief]",
        interview_focus=[
            "Anomaly (rare) vs fraud (malicious).",
            "Why supervised models miss zero-day attacks.",
            "How anomaly scores feed into supervised fraud stacks.",
        ],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("**4 ways a transaction is unusual**")
            st.markdown(
                r"""
- **Point:** extreme amount (**\$2,200** on a **\$50** card)
- **Context:** bad *combo* (3 AM + 200 km + high-risk merchant)
- **Sequence:** normal **\$90** charges every 2 min
- **Network:** 5 accounts sharing 1 device
"""
            )
    with col_b:
        with st.container(border=True):
            st.markdown("**Why use anomaly detection?**")
            st.markdown(
                """
- **Rare labels:** fraud is `0.05%-1%` of traffic
- **Label delay:** chargebacks arrive `30-90 days` late
- **Adversarial drift:** fraudsters invent unseen attacks
- **Zero-day coverage:** learns *normal* without fraud labels
"""
            )

    comp = pd.DataFrame(
        [
            {"Axis": "Labels", "Supervised fraud": "Needs fraud + normal labels", "Anomaly detection": "Unlabeled or normal-only"},
            {"Axis": "Learns", "Supervised fraud": "Known fraud boundary", "Anomaly detection": "Normal behavior manifold"},
            {"Axis": "Best at", "Supervised fraud": "High precision on known fraud", "Anomaly detection": "Zero-day / novel attacks"},
            {"Axis": "Main risk", "Supervised fraud": "Misses unseen patterns", "Anomaly detection": "Flags rare legit spend (travel)"},
        ]
    )
    dataframe(comp, hide_index=True)

    st.markdown("**Hybrid production stack**")
    mermaid_chart(
        """
flowchart LR
    Tx["Transaction"] --> Feat["Features"]
    Feat --> Anom["Anomaly detectors"]
    Feat --> Sup["Supervised fraud model"]
    Anom -->|"Novelty score"| Sup
    Anom -->|"Zero-day alert"| Act["2FA / Analyst review"]
    Sup -->|"Risk score"| Act
"""
    )
