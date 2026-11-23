"""Lesson 4: Mahalanobis Distance — crisp, equation-first."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.data import HAND_2D_TX
from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC
from anomaly_lab.statistical import correlated_2d_demo, mahalanobis_analysis
from anomaly_lessons.charts import mahalanobis_2d_chart
from anomaly_lessons.ui import lesson_header, render_interview_qa, working
from st_compat import dataframe, metric


def render() -> None:
    lesson_header(
        "4. Mahalanobis Distance",
        "Multivariate Z-score that accounts for feature scale and correlation.",
        priority_badge=":red-badge[Highest priority — Statistical]",
        interview_focus=[
            "Ellipse (Mahalanobis) vs circle (Euclidean) on correlated features.",
            r"Threshold via $D_M^2 \sim \chi^2_p$.",
            r"Singular $\Sigma$ fix: ridge shrinkage $(\Sigma + \lambda I)^{-1}$ or PCA.",
        ],
    )

    tab_math, tab_demo, tab_code, tab_qa = st.tabs(
        ["Intuition & math", "Interactive ellipse", "Code & limits", "Interview prep"]
    )

    with tab_math:
        st.markdown("**Intuition:** Draws a tilted ellipse aligned with feature correlation, penalizing off-trend jumps.")
        st.latex(r"\text{1D: } z^2 = (x-\mu)(\sigma^2)^{-1}(x-\mu) \quad\longrightarrow\quad \text{Multi-D: } D_M(x) = \sqrt{(x - \mu)^T \, \Sigma^{-1} \, (x - \mu)}")
        st.latex(r"\text{Threshold: } D_M^2(x) \sim \chi^2_p \;\implies\; \tau = \sqrt{\chi^2_{p,\,1-\alpha}}")
        with st.container(border=True):
            st.markdown(
                r"""
- **$\Sigma = I$:** collapses to ordinary Euclidean distance
- **Diagonal $\Sigma$:** collapses to standardized Z-score distance $\sqrt{\sum z_j^2}$
- **Scale-invariant:** dollars vs cents gives identical $D_M(x)$
- **Regularization:** invert $(\Sigma + \lambda I)$ when features are collinear
"""
            )

    with tab_demo:
        st.markdown(r"**Learns:** $(\mu, \Sigma^{-1})$ on normal $X \in \mathbb{R}^{n \times p}$. **Flags:** $D_M(x) > \sqrt{\chi^2_{p, 1-\alpha}}$.")
        rho = st.slider("Feature correlation (rho: amount vs distance)", 0.0, 0.95, 0.88, 0.05, key="mahal_rho")
        pts_df, contours_df, res = correlated_2d_demo(rho=rho)

        along_row = pts_df[pts_df["point_role"] == "Along-trend (Legit)"].iloc[0]
        off_row = pts_df[pts_df["point_role"] == "Off-trend (Fraud)"].iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            metric("Chi2 cutoff (97.5%)", f"{res.chi2_threshold:.2f}")
        with c2:
            metric("Along-trend Legit D_M", f"{along_row['mahalanobis_dist']:.2f}", delta=f"Euclid {along_row['euclidean_z_dist']:.2f}")
        with c3:
            metric("Off-trend Fraud D_M", f"{off_row['mahalanobis_dist']:.2f}", delta=f"Euclid {off_row['euclidean_z_dist']:.2f}")

        mahalanobis_2d_chart(pts_df, contours_df)

        with st.expander("Worked 12-row numbers (M12 breaks correlation)"):
            X_small = HAND_2D_TX[["amount", "distance_km"]].to_numpy(dtype=float)
            fit_normal = (HAND_2D_TX["label"] == "Normal").to_numpy()
            small_res = mahalanobis_analysis(
                X_small,
                feature_names=["amount", "distance_km"],
                labels=HAND_2D_TX["tx_id"].tolist(),
                fit_mask=fit_normal,
            )
            working(
                [
                    f"mu = [{small_res.mean[0]:.1f}, {small_res.mean[1]:.2f}],  rho = {small_res.corr[0, 1]:.3f},  cutoff = {small_res.chi2_threshold:.2f}",
                    f"M12 ($640, 3.5 km): Euclid Z = {small_res.table.iloc[-1]['euclidean_z_dist']:.2f},  D_M = {small_res.table.iloc[-1]['mahalanobis_dist']:.2f} -> FLAGGED",
                ]
            )
            dataframe(small_res.table, hide_index=True)

    with tab_code:
        st.code(
            """import numpy as np
from scipy.stats import chi2

def mahalanobis_anomalies(X_train_normal, X_eval, alpha=0.975, reg=1e-6):
    mu = X_train_normal.mean(axis=0)
    cov = np.cov(X_train_normal, rowvar=False)
    inv_cov = np.linalg.inv(cov + reg * np.eye(cov.shape[0]))
    diff = X_eval - mu
    d_m = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
    return d_m, d_m > np.sqrt(chi2.ppf(alpha, df=X_train_normal.shape[1]))
""",
            language="python",
        )
        lim = pd.DataFrame(
            [
                {"Limit": "Single elliptical cluster", "Fix": "Use Isolation Forest / OC-SVM for curved or multi-cluster data"},
                {"Limit": "Singular covariance (n < p)", "Fix": "Ledoit-Wolf shrinkage, ridge (Sigma + lambda*I), or PCA"},
                {"Limit": "Outlier-sensitive Sigma", "Fix": "Fit on clean normal rows or use Robust MCD"},
                {"Limit": "Continuous numeric only", "Fix": "Encode or use tree/neural models for categoricals"},
            ]
        )
        dataframe(lim, hide_index=True)

    with tab_qa:
        render_interview_qa(CORE_QA_BY_TOPIC["Mahalanobis"], key_prefix="qa_mahal")
