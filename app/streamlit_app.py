"""Streamlit dashboard: EDA trends + model feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from credit_fraud import config
from credit_fraud.data import load_raw_dataframe, split_xy
from credit_fraud.eda import means_by_class, missingness_table, run_eda, target_summary
from credit_fraud.io_utils import load_json
from credit_fraud.rank_plot import rank_and_plot

st.set_page_config(page_title="Credit Fraud Mentoring", layout="wide")
st.title("Credit Fraud — EDA & Modelling Dashboard")
st.caption("Reads `config.py` + `artifacts/`. Train first for importance / metrics tabs.")


@st.cache_data(show_spinner="Loading dataset...")
def _load_data() -> pd.DataFrame:
    return load_raw_dataframe()


def _maybe_metrics() -> dict | None:
    if config.METRICS_PATH.exists():
        return load_json(config.METRICS_PATH)
    return None


def tab_target(df: pd.DataFrame) -> None:
    x, y = split_xy(df)
    summary, imbalance = target_summary(y)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Positive rate", f"{(y == config.POSITIVE_LABEL).mean():.4%}")
    c3.metric("Imbalance neg/pos", f"{imbalance:.1f}")
    st.dataframe(summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x=y.astype(str), ax=ax, color="#2F6F8F")
    ax.set_title("Target distribution")
    ax.set_xlabel(config.TARGET_COL)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Independent variable describe"):
        st.dataframe(x.describe().T, use_container_width=True)
    with st.expander("Missingness"):
        st.dataframe(missingness_table(x), use_container_width=True)
    with st.expander("Means by class"):
        st.dataframe(means_by_class(x, y), use_container_width=True)


def tab_ranks(df: pd.DataFrame) -> None:
    x, y = split_xy(df)
    if config.RANKING_PATH.exists():
        ranking = pd.read_csv(config.RANKING_PATH)
        st.success(f"Loaded ranking from {config.RANKING_PATH.name}")
    else:
        with st.spinner("Computing univariate ranks..."):
            result = rank_and_plot(x, y, save=False)
            ranking = result.ranking

    top_n = st.slider("Top N", 5, min(30, len(ranking)), 15)
    st.dataframe(ranking.head(top_n), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=ranking.head(top_n), y="feature", x="univariate_auc", color="#2F6F8F", ax=ax)
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Univariate AUC rank")
    st.pyplot(fig)
    plt.close(fig)

    feature = st.selectbox("Inspect feature by class", ranking["feature"].tolist())
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    plot_df = pd.DataFrame({feature: x[feature], config.TARGET_COL: y.astype(str)})
    sns.boxplot(data=plot_df, x=config.TARGET_COL, y=feature, ax=ax2, color="#5B8FA8")
    ax2.set_title(f"{feature} by class")
    st.pyplot(fig2)
    plt.close(fig2)


def tab_features(df: pd.DataFrame) -> None:
    from credit_fraud.features import FeatureBuilder

    x, y = split_xy(df)
    builder = FeatureBuilder().fit(x, y)
    engineered = builder.transform(x.head(500))
    st.write("Sample of engineered features (first 500 rows after transform):")
    new_cols = [c for c in engineered.columns if c not in x.columns]
    st.write("Constructed columns:", new_cols or "(none — Amount/Time not in dataset)")
    st.dataframe(engineered.head(20), use_container_width=True)


def tab_importance() -> None:
    metrics = _maybe_metrics()
    if metrics is None:
        st.warning("No `artifacts/metrics.json` yet. Run `run_training()` first.")
        return

    st.subheader(f"Selected model: `{metrics.get('best_model')}`")
    st.json(metrics.get("best_metrics", {}))

    if config.IMPORTANCES_PATH.exists():
        imp = pd.read_csv(config.IMPORTANCES_PATH)
    else:
        st.warning("feature_importances.csv missing.")
        return

    top_n = st.slider("Top features", 5, min(40, len(imp)), 20, key="imp_top")
    st.dataframe(imp.head(top_n), use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=imp.head(top_n), y="feature", x="importance", color="#C45C26", ax=ax)
    ax.set_title("Feature importance")
    st.pyplot(fig)
    plt.close(fig)

    if "comparison" in metrics:
        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(metrics["comparison"]), use_container_width=True)


def main() -> None:
    st.sidebar.markdown("### Config")
    st.sidebar.code(
        f"TARGET_COL={config.TARGET_COL}\n"
        f"MAX_ROWS={config.MAX_ROWS}\n"
        f"N_TRIALS={config.N_TRIALS}\n"
        f"DATA_PATH={config.DATA_PATH}",
        language="text",
    )
    if st.sidebar.button("Refresh EDA artifacts"):
        df = _load_data()
        x, y = split_xy(df)
        run_eda(x, y, save=True)
        rank_and_plot(x, y, save=True)
        st.sidebar.success("Saved figures + ranking under artifacts/")

    try:
        df = _load_data()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data: {exc}")
        st.info("Set DATA_PATH / DATA_URL in config.py or drop a CSV into input/.")
        return

    tabs = st.tabs(["Target & EDA", "Rank & Plot", "Feature Engineering", "Modelling"])
    with tabs[0]:
        tab_target(df)
    with tabs[1]:
        tab_ranks(df)
    with tabs[2]:
        tab_features(df)
    with tabs[3]:
        tab_importance()


if __name__ == "__main__":
    main()
