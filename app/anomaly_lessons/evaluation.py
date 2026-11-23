"""Lesson 9: Evaluation & model comparison — crisp, equation-first."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.data import make_financial_benchmark
from anomaly_lab.evaluation import (
    CORE_MODEL_NAMES,
    SCENARIO_SELECTOR_ROWS,
    precision_at_k_table,
    run_six_model_benchmark,
    threshold_cost_sweep,
)
from anomaly_lessons.charts import cost_tradeoff_chart, evaluation_curves_chart
from anomaly_lessons.ui import lesson_header
from st_compat import choice, dataframe, metric


@st.cache_data(max_entries=12)
def _cached_benchmark_comparison(contamination: float):
    df = make_financial_benchmark()
    return df, run_six_model_benchmark(df, contamination=contamination)


def render() -> None:
    lesson_header(
        "9. Evaluation & comparison",
        "Benchmark all 6 core algorithms, optimize FP vs FN cost, align score scales, and evaluate with scarce labels.",
        priority_badge=":gray-badge[Supporting — high interview value]",
        interview_focus=[
            "Why **PR-AUC** beats ROC-AUC on rare fraud (ignores massive True Negatives).",
            r"Threshold minimizes $\text{Cost} = \text{FP}\cdot C_{\text{friction}} + \text{FN}\cdot C_{\text{fraud}}$.",
            "Align incompatible scores via empirical percentile rank (ECDF) in $[0, 1]$.",
            "Scarce labels: **Precision@K** analyst queue + synthetic anomaly injection.",
        ],
    )

    tab_bench, tab_cost, tab_scales, tab_choose = st.tabs(
        ["6-model benchmark", "FP vs FN cost", "Scales & scarce labels", "Algorithm chooser"]
    )

    with tab_bench:
        st.latex(
            r"\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \qquad "
            r"\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}, \qquad "
            r"F_1 = \frac{2\,\text{PR}}{\text{P}+\text{R}}"
        )
        contam = st.slider("Alert rate / contamination", 0.02, 0.15, 0.05, 0.01, key="eval_contam")
        df, comp = _cached_benchmark_comparison(contam)

        best = comp.metrics_df.iloc[0]
        m1, m2, m3 = st.columns(3)
        with m1:
            metric("Top by PR-AUC", str(best["Algorithm"]))
        with m2:
            metric("Best PR-AUC", f"{best['PR-AUC']:.3f}")
        with m3:
            metric("Best ROC-AUC", f"{best['ROC-AUC']:.3f}")

        dataframe(comp.metrics_df, hide_index=True)
        evaluation_curves_chart(comp.pr_curves, comp.roc_curves)

    with tab_cost:
        st.latex(r"\text{Total Cost}(\tau) = \text{FP}(\tau) \cdot C_{\text{friction}} + \text{FN}(\tau) \cdot C_{\text{fraud}}")
        df, comp = _cached_benchmark_comparison(0.05)

        c1, c2, c3 = st.columns(3)
        with c1:
            chosen_model = st.selectbox("Model", CORE_MODEL_NAMES, index=3, key="eval_cost_model")
        with c2:
            fp_cost = st.slider("FP friction cost ($)", 5.0, 100.0, 15.0, 5.0, key="eval_fp_cost")
        with c3:
            fn_cost = st.slider("FN fraud loss ($)", 100.0, 1500.0, 450.0, 50.0, key="eval_fn_cost")

        sweep = threshold_cost_sweep(
            df["is_anomaly"].to_numpy(dtype=int),
            comp.scores_df[chosen_model].to_numpy(dtype=float),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        opt_row = sweep.loc[sweep["Total cost ($)"].idxmin()]
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric("Optimal alert rate", f"{opt_row['alert_rate_pct']:.0f}%")
        with k2:
            metric("Min cost", f"${opt_row['Total cost ($)']:.0f}")
        with k3:
            metric("P / R", f"{opt_row['Precision']:.2f} / {opt_row['Recall']:.2f}")
        with k4:
            metric("FP / FN", f"{int(opt_row['FP'])} / {int(opt_row['FN'])}")

        cost_tradeoff_chart(sweep)

    with tab_scales:
        st.latex(r"\tilde{s}_m(x) = \frac{\text{rank}\bigl(s_m(x)\bigr) - 1}{N - 1} \in [0, 1] \quad\text{(empirical percentile rank)}")
        df, comp = _cached_benchmark_comparison(0.05)
        scale_mode = choice(
            "Display scale",
            options=["Raw native scores", "Percentile [0, 1]"],
            default="Percentile [0, 1]",
            key="eval_scale_mode",
            horizontal=True,
        )
        if scale_mode == "Percentile [0, 1]":
            cols = ["tx_id", "anomaly_type"] + [f"{m}__pct" for m in CORE_MODEL_NAMES]
            view = comp.scores_df[comp.scores_df["is_anomaly"] == 1][cols].head(8).copy()
            view.columns = ["tx_id", "anomaly_type", *CORE_MODEL_NAMES]
        else:
            view = comp.scores_df[comp.scores_df["is_anomaly"] == 1][["tx_id", "anomaly_type", *CORE_MODEL_NAMES]].head(8)
        dataframe(view, hide_index=True)

        st.markdown("**Scarce-label evaluation:** Top-$K$ analyst review (`Precision@K`) + synthetic anomaly injection.")
        top_k = st.slider("Analyst review queue (Top-K)", 10, 60, 24, 2, key="eval_topk")
        scores_map = {m: comp.scores_df[m].to_numpy(dtype=float) for m in CORE_MODEL_NAMES}
        dataframe(precision_at_k_table(df["is_anomaly"].to_numpy(dtype=int), scores_map, k=top_k), hide_index=True)

    with tab_choose:
        dataframe(pd.DataFrame(SCENARIO_SELECTOR_ROWS), hide_index=True)
