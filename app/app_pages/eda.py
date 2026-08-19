"""EDA, rank-and-plot, features, and trained-model importance."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from st_compat import button, dataframe, plotly_chart

from credit_fraud import config
from credit_fraud.data import feature_glossary, load_raw_dataframe, split_xy
from credit_fraud.eda import missingness_table, plot_raw_distribution, run_eda, target_summary
from credit_fraud.io_utils import load_json, round_floats
from credit_fraud.rank_plot import binned_capture_table, plot_capture_by_bin, rank_and_plot


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
    c2.metric("Positive rate", f"{(y == config.POSITIVE_LABEL).mean():.2%}")
    c3.metric("Imbalance neg/pos", f"{imbalance:.2f}")
    dataframe(summary)

    fig = go.Figure(
        go.Bar(
            x=summary[config.TARGET_COL].astype(str),
            y=summary["count"],
            marker_color="#2F6F8F",
        )
    )
    fig.update_layout(
        title="Target distribution",
        xaxis_title=config.TARGET_COL,
        yaxis_title="count",
        template="plotly_white",
        height=360,
    )
    plotly_chart(fig)

    with st.expander("Feature dictionary (what each column means)"):
        dataframe(feature_glossary(list(x.columns) + [config.TARGET_COL]))
    with st.expander("Independent variable describe"):
        dataframe(round_floats(x.describe().T))
    with st.expander("Missingness"):
        dataframe(missingness_table(x))

    st.subheader("Raw variable distributions")
    feature = st.selectbox(
        "Choose independent variable",
        options=list(x.columns),
        key="eda_raw_feature",
    )
    if feature in config.FEATURE_DICTIONARY:
        st.caption(config.FEATURE_DICTIONARY[feature])
    plotly_chart(plot_raw_distribution(x, feature))


def tab_ranks(df: pd.DataFrame) -> None:
    x, y = split_xy(df)
    if config.RANKING_PATH.exists():
        ranking = round_floats(pd.read_csv(config.RANKING_PATH))
        st.success(f"Loaded ranking from {config.RANKING_PATH.name}")
    else:
        with st.spinner("Computing univariate ranks..."):
            result = rank_and_plot(x, y, save=False)
            ranking = result.ranking

    dataframe(ranking)

    st.subheader("Capture rate by feature bin")
    st.caption(
        "Pick any independent variable from the dropdown. "
        "X = feature bins · Bars = observation count · Line = capture rate "
        "(positives in bin / all positives)."
    )
    rank_lookup = dict(zip(ranking["feature"], ranking["rank_auc"]))
    glossary = config.FEATURE_DICTIONARY

    def _label(f: str) -> str:
        rank = rank_lookup.get(f, "?")
        meaning = glossary.get(f, "")
        short = (meaning[:48] + "…") if len(meaning) > 48 else meaning
        return f"{f}  (rank #{rank}) — {short}" if short else f"{f}  (rank #{rank})"

    feature = st.selectbox(
        "Choose independent variable",
        options=ranking["feature"].tolist(),
        format_func=_label,
        key="rank_feature_dropdown",
    )
    if feature in glossary:
        st.info(f"**{feature}:** {glossary[feature]}")
    n_bins = st.slider("Bins", 5, 20, config.RANK_N_BINS, key="capture_bins")
    table = binned_capture_table(x[feature], y, n_bins=n_bins)
    plotly_chart(plot_capture_by_bin(feature, table))
    dataframe(table)


def tab_features(df: pd.DataFrame) -> None:
    from credit_fraud.features import FeatureBuilder

    x, y = split_xy(df)
    builder = FeatureBuilder().fit(x, y)
    engineered = builder.transform(x.head(500))
    st.write("Sample of engineered features (first 500 rows after transform):")
    new_cols = [c for c in engineered.columns if c not in x.columns]
    st.write("Constructed columns:", new_cols or "(none for this schema)")
    if new_cols:
        dataframe(feature_glossary(new_cols))
    dataframe(round_floats(engineered.head(20)))


def tab_importance() -> None:
    metrics = _maybe_metrics()
    if metrics is None:
        st.warning("No `artifacts/metrics.json` yet. Run `run_training()` first.")
        return

    if config.IMPORTANCES_PATH.exists():
        imp = pd.read_csv(config.IMPORTANCES_PATH)
        if imp["feature"].astype(str).str.match(r"^V\d+$").any():
            st.error(
                "Feature importance still looks like the old anonymized fraud model (`V1`…). "
                "Re-run training on the UCI credit-default data:\n\n"
                '`python -c "from credit_fraud.pipeline import run_training; run_training()"`'
            )
            return
    else:
        st.warning("feature_importances.csv missing.")
        return

    st.subheader(f"Selected model: `{metrics.get('best_model')}`")
    best = {
        k: round(float(v), 2) if isinstance(v, (int, float)) else v
        for k, v in (metrics.get("best_metrics") or {}).items()
    }
    st.json(best)

    imp = round_floats(imp)
    top_n = st.slider("Top features", 5, min(40, len(imp)), min(20, len(imp)), key="imp_top")
    dataframe(imp.head(top_n))
    fig = px.bar(
        imp.head(top_n).iloc[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title="Feature importance",
        color_discrete_sequence=["#C45C26"],
    )
    fig.update_layout(template="plotly_white", height=max(360, 28 * top_n))
    plotly_chart(fig)

    if "comparison" in metrics:
        st.subheader("Model comparison")
        dataframe(round_floats(pd.DataFrame(metrics["comparison"])))


def render() -> None:
    st.title("EDA and modelling")
    st.caption("Reads `config.py` + `artifacts/`. Train first for the modelling tab.")

    st.sidebar.markdown("### Config")
    st.sidebar.code(
        f"TARGET_COL={config.TARGET_COL}\n"
        f"MAX_ROWS={config.MAX_ROWS}\n"
        f"N_TRIALS={config.N_TRIALS}\n"
        f"RANK_N_BINS={config.RANK_N_BINS}\n"
        f"DATA_PATH={config.DATA_PATH}",
        language="text",
    )
    if button("Refresh EDA artifacts", key="refresh_eda"):
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

    tabs = st.tabs(["Target and EDA", "Rank and plot", "Feature engineering", "Modelling"])
    with tabs[0]:
        tab_target(df)
    with tabs[1]:
        tab_ranks(df)
    with tabs[2]:
        tab_features(df)
    with tabs[3]:
        tab_importance()
