"""Lesson 2: Z-score — crisp, equation-first."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.data import HAND_TX_DF
from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC
from anomaly_lab.statistical import outlier_masking_demo, zscore_analysis
from anomaly_lessons.charts import amount_threshold_chart
from anomaly_lessons.ui import lesson_header, render_interview_qa, working
from st_compat import dataframe, metric


def render() -> None:
    lesson_header(
        "2. Z-score",
        "Distance from the mean measured in standard deviations.",
        priority_badge=":red-badge[Highest priority — Statistical]",
        interview_focus=[
            "Symmetry assumption vs right-skewed spend.",
            "Outlier masking (0% breakdown point).",
            "Fixes: `log1p(amount)` or Modified Z-score (median/MAD).",
        ],
    )

    tab_math, tab_demo, tab_code, tab_qa = st.tabs(
        ["Intuition & math", "Interactive example", "Code & limits", "Interview prep"]
    )

    with tab_math:
        st.markdown("**Intuition:** Center by subtracting mean $\\mu$, scale by dividing by spread $\\sigma$.")
        st.latex(
            r"\mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad "
            r"\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \mu)^2}, \qquad "
            r"z_i = \frac{x_i - \mu}{\sigma}"
        )
        st.latex(r"\text{Anomaly if } |z_i| > \tau \qquad\qquad \text{Robust (MAD): } M_i = \frac{0.6745\,(x_i - \text{med})}{\text{MAD}}")
        with st.container(border=True):
            st.markdown(
                r"""
- **$|z| \le 2.0$:** normal (~95.4% Gaussian)
- **$|z| > 2.5$:** suspicious tail (~1.2%)
- **$|z| > 3.0$:** strong anomaly (~0.27%)
- **Skew fix:** run Z-score on $y = \ln(1 + x)$
"""
            )

    with tab_demo:
        st.markdown(r"**Learns:** $(\mu, \sigma)$ in $O(n)$. **Scores:** $|(x_{\text{new}} - \mu)/\sigma| > \tau$.")
        c1, c2 = st.columns(2)
        with c1:
            z_thresh = st.slider("Threshold (tau)", 1.5, 3.5, 2.5, 0.1, key="z_thresh")
        with c2:
            t10_amt = st.slider("T10 amount ($)", 80.0, 900.0, 490.0, 10.0, key="z_t10")

        amounts = HAND_TX_DF["amount"].tolist()
        amounts[-1] = t10_amt
        res = zscore_analysis(amounts, labels=HAND_TX_DF["tx_id"].tolist(), threshold=z_thresh)
        upper_cut = res.mean + z_thresh * res.std
        lower_cut = res.mean - z_thresh * res.std

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric("Mean (mu)", f"${res.mean:.2f}")
        with m2:
            metric("Std (sigma)", f"${res.std:.2f}")
        with m3:
            metric("Upper cutoff", f"${upper_cut:.2f}")
        with m4:
            metric("T10 Z-score", f"{res.table.iloc[-1]['z_score']:.2f}")

        working(
            [
                f"mu = {res.mean:.2f},  sigma = {res.std:.2f}",
                f"z(T10) = ({t10_amt:.1f} - {res.mean:.2f}) / {res.std:.2f} = {res.table.iloc[-1]['z_score']:.3f}",
                f"Flagged (|z| > {z_thresh:.1f}) -> {bool(res.table.iloc[-1]['is_anomaly'])}",
            ]
        )
        amount_threshold_chart(res.table, lower=lower_cut, upper=upper_cut, title=f"Z-score cutoff = ${upper_cut:.1f}")
        dataframe(res.table[["tx_id", "amount", "z_score", "modified_z", "log1p_z", "is_anomaly"]], hide_index=True)

    with tab_code:
        st.code(
            """import numpy as np

def zscore_anomalies(amounts, threshold=2.5, use_log=False):
    x = np.log1p(amounts) if use_log else np.asarray(amounts, dtype=float)
    z = (x - x.mean()) / x.std(ddof=1)
    return z, np.abs(z) > threshold
""",
            language="python",
        )
        lim = pd.DataFrame(
            [
                {"Limit": "Outlier masking", "Fix": "Use Median/MAD or IQR (1 spike inflates sigma)"},
                {"Limit": "Right-skewed spend", "Fix": "Apply log1p(amount) first"},
                {"Limit": "Behavior drift", "Fix": "Use rolling per-customer windows"},
                {"Limit": "1D only", "Fix": "Switch to Mahalanobis / ML for multi-feature"},
            ]
        )
        dataframe(lim, hide_index=True)

        extreme_val = st.slider("Live masking demo: inject 1 extreme fraud ($)", 500.0, 6000.0, 3500.0, 250.0, key="z_mask_extreme")
        before, after = outlier_masking_demo(HAND_TX_DF["amount"].tolist(), extreme_amount=extreme_val, threshold=2.5)
        t10_before = float(before.table.iloc[-1]["z_score"])
        t10_after = float(after.table.iloc[-2]["z_score"])
        k1, k2, k3 = st.columns(3)
        with k1:
            metric("T10 ($490) Z before", f"{t10_before:.2f}")
        with k2:
            metric("New std (sigma)", f"${after.std:.1f}", delta=f"+${after.std - before.std:.1f}")
        with k3:
            metric("T10 ($490) Z after", f"{t10_after:.2f}", delta="Masked!" if abs(t10_after) <= 2.5 else "Flagged")

    with tab_qa:
        render_interview_qa(CORE_QA_BY_TOPIC["Z-score"], key_prefix="qa_zscore")
