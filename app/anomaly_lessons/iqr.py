"""Lesson 3: IQR — crisp, equation-first."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.data import HAND_TX_DF
from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC
from anomaly_lab.statistical import compare_zscore_vs_iqr, iqr_analysis
from anomaly_lessons.charts import amount_threshold_chart
from anomaly_lessons.ui import lesson_header, render_interview_qa, working
from st_compat import dataframe, metric


def render() -> None:
    lesson_header(
        "3. IQR (Interquartile Range)",
        "Outlier fences built from the middle 50% of sorted values — immune to extreme spikes.",
        priority_badge=":red-badge[Highest priority — Statistical]",
        interview_focus=[
            "25% breakdown point vs Z-score's 0%.",
            "Why $k = 1.5 \\approx \\pm 2.7\\sigma$ on Gaussian data.",
            "Edge cases: $\\text{IQR}=0$, negative lower fence, 1D only.",
        ],
    )

    tab_math, tab_demo, tab_code, tab_qa = st.tabs(
        ["Intuition & math", "Interactive example", "Code & limits", "Interview prep"]
    )

    with tab_math:
        st.markdown("**Intuition:** Ignore top 25% and bottom 25% so extreme fraud cannot stretch your threshold.")
        st.latex(
            r"\text{IQR} = Q_3 - Q_1, \qquad "
            r"L = Q_1 - k \cdot \text{IQR}, \qquad "
            r"U = Q_3 + k \cdot \text{IQR}"
        )
        st.latex(r"\text{Normal Gaussian benchmark: } Q_3 + 1.5\,\text{IQR} = 0.6745 + 1.5(1.349) \approx 2.70\,\sigma")
        with st.container(border=True):
            st.markdown(
                r"""
- **$Q_1, Q_2, Q_3$:** 25th, 50th (median), 75th percentiles
- **$k = 1.5$:** standard outlier fence (~0.7% Gaussian tail)
- **$k = 3.0$:** extreme outlier fence (~4.7$\sigma$)
"""
            )

    with tab_demo:
        st.markdown(r"**Learns:** $(Q_1, Q_3)$ via sort $O(n \log n)$. **Flags:** $x < L$ or $x > U$.")
        c1, c2 = st.columns(2)
        with c1:
            k_mult = st.slider("Multiplier (k)", 1.0, 3.5, 1.5, 0.25, key="iqr_k")
        with c2:
            inject_spike = st.toggle("Inject $5,000 spike", value=True, key="iqr_spike")

        amounts = HAND_TX_DF["amount"].tolist()
        labels = HAND_TX_DF["tx_id"].tolist()
        if inject_spike:
            amounts.append(5000.0)
            labels.append("T11_SPIKE")

        res = iqr_analysis(amounts, labels=labels, k=k_mult)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric("Q1", f"${res.q1:.2f}")
        with m2:
            metric("Q3", f"${res.q3:.2f}")
        with m3:
            metric("IQR", f"${res.iqr:.2f}")
        with m4:
            metric("Upper fence", f"${res.upper_fence:.2f}")

        working(
            [
                f"IQR = Q3 - Q1 = {res.q3:.2f} - {res.q1:.2f} = {res.iqr:.2f}",
                f"Fences = [{res.lower_fence:.2f},  {res.upper_fence:.2f}]  (k = {k_mult:.2f})",
            ]
        )
        amount_threshold_chart(res.table, lower=res.lower_fence, upper=res.upper_fence, title=f"IQR upper fence = ${res.upper_fence:.1f}")

        st.markdown("**Z-score vs IQR side-by-side**")
        dataframe(compare_zscore_vs_iqr(amounts, labels=labels, z_threshold=2.5, k=k_mult), hide_index=True)

    with tab_code:
        st.code(
            """import numpy as np

def iqr_anomalies(amounts, k=1.5):
    x = np.asarray(amounts, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x < lo) | (x > hi), (lo, hi)
""",
            language="python",
        )
        lim = pd.DataFrame(
            [
                {"Limit": "1D only", "Impact": "Misses joint multi-feature anomalies"},
                {"Limit": "IQR = 0 trap", "Impact": "Sparse/discrete counts (Q1=Q3=0) flag every non-zero"},
                {"Limit": "Negative lower fence", "Impact": "Misses tiny $0.01 card-testing probes"},
                {"Limit": "Multimodal spend", "Impact": "Quartiles mix distinct regimes (coffee + rent)"},
            ]
        )
        dataframe(lim, hide_index=True)

    with tab_qa:
        render_interview_qa(CORE_QA_BY_TOPIC["IQR"], key_prefix="qa_iqr")
