"""EDA for independent (X) and dependent (y) variables — Plotly charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from credit_fraud import config
from credit_fraud.io_utils import ensure_dirs, round_floats, save_json, save_plotly_figure


@dataclass
class EDAResult:
    target_summary: pd.DataFrame
    describe_x: pd.DataFrame
    missingness: pd.DataFrame
    figures: dict[str, Any]
    imbalance_ratio: float


def target_summary(y: pd.Series) -> tuple[pd.DataFrame, float]:
    """Dependent-variable summary: counts, rates, imbalance ratio (neg/pos)."""
    counts = y.value_counts().sort_index()
    rates = y.value_counts(normalize=True).sort_index()
    summary = pd.DataFrame({"count": counts, "rate": rates.round(2)})
    summary.index.name = config.TARGET_COL
    pos = int((y == config.POSITIVE_LABEL).sum())
    neg = int((y != config.POSITIVE_LABEL).sum())
    imbalance = (neg / pos) if pos else float("inf")
    return summary.reset_index(), round(float(imbalance), 2)


def describe_features(x: pd.DataFrame) -> pd.DataFrame:
    """Independent-variable numeric describe (rounded)."""
    return round_floats(x.describe().T.reset_index().rename(columns={"index": "feature"}))


def missingness_table(x: pd.DataFrame) -> pd.DataFrame:
    miss = x.isna().sum()
    return round_floats(
        pd.DataFrame(
            {"feature": miss.index, "n_missing": miss.values, "pct_missing": miss.values / len(x)}
        )
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )


def plot_target_distribution(y: pd.Series) -> go.Figure:
    summary, _ = target_summary(y)
    fig = go.Figure(
        go.Bar(
            x=summary[config.TARGET_COL].astype(str),
            y=summary["count"],
            marker_color="#2F6F8F",
            hovertemplate="class=%{x}<br>count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Target distribution (class imbalance)",
        xaxis_title=config.TARGET_COL,
        yaxis_title="count",
        template="plotly_white",
        height=360,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def plot_raw_distribution(x: pd.DataFrame, feature: str, bins: int = 40) -> go.Figure:
    """Histogram of one raw independent variable."""
    fig = px.histogram(
        x,
        x=feature,
        nbins=bins,
        title=f"Raw distribution: {feature}",
        color_discrete_sequence=["#2F6F8F"],
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        bargap=0.05,
        xaxis_title=feature,
        yaxis_title="count",
    )
    return fig


def plot_raw_distributions_dropdown(
    x: pd.DataFrame,
    features: list[str] | None = None,
    bins: int = 40,
) -> go.Figure:
    """One Plotly chart with a dropdown over raw independent variables."""
    features = features or list(x.columns)
    features = [f for f in features if f in x.columns]
    if not features:
        raise ValueError("No features for raw distribution plot.")

    fig = go.Figure()
    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=x[feature],
                nbinsx=bins,
                name=feature,
                marker_color="#2F6F8F",
                visible=(i == 0),
                hovertemplate=f"{feature}=%{{x}}<br>count=%{{y}}<extra></extra>",
            )
        )

    buttons = []
    for i, feature in enumerate(features):
        visible = [False] * len(features)
        visible[i] = True
        buttons.append(
            {
                "label": feature,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Raw distribution: {feature}",
                        "xaxis": {"title": feature},
                    },
                ],
            }
        )

    first = features[0]
    fig.update_layout(
        title=f"Raw distribution: {first}",
        xaxis_title=first,
        yaxis_title="count",
        template="plotly_white",
        height=420,
        margin=dict(l=50, r=40, t=100, b=40),
        showlegend=False,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "#F7F7F7",
                "bordercolor": "#CCCCCC",
            }
        ],
        annotations=[
            {
                "text": "Feature:",
                "x": 1.0,
                "xref": "paper",
                "y": 1.18,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "right",
                "font": {"size": 12},
            }
        ],
    )
    return fig


def run_eda(
    x: pd.DataFrame,
    y: pd.Series,
    save: bool = True,
    example_feature: str | None = None,
) -> EDAResult:
    """Run dependent + independent EDA; optionally persist summary + figures."""
    ensure_dirs()
    tgt, imbalance = target_summary(y)
    desc = describe_features(x)
    miss = missingness_table(x)

    figures: dict[str, Any] = {
        "target_distribution": plot_target_distribution(y),
        "raw_distributions": plot_raw_distributions_dropdown(x),
    }

    feat = example_feature or (
        "LIMIT_BAL"
        if "LIMIT_BAL" in x.columns
        else ("Amount" if "Amount" in x.columns else x.columns[0])
    )
    figures[f"hist_{feat}"] = plot_raw_distribution(x, feat)

    if save:
        save_plotly_figure(figures["target_distribution"], "target_distribution")
        save_plotly_figure(figures["raw_distributions"], "raw_distributions")
        save_plotly_figure(figures[f"hist_{feat}"], f"hist_{feat}")
        save_json(
            {
                "n_rows": int(len(y)),
                "n_features": int(x.shape[1]),
                "imbalance_ratio_neg_over_pos": imbalance,
                "positive_rate": round(float((y == config.POSITIVE_LABEL).mean()), 2),
                "target_counts": tgt.set_index(config.TARGET_COL)["count"].to_dict(),
            },
            config.EDA_SUMMARY_PATH,
        )

    return EDAResult(
        target_summary=tgt,
        describe_x=desc,
        missingness=miss,
        figures=figures,
        imbalance_ratio=imbalance,
    )
