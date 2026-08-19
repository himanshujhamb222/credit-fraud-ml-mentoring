"""Univariate feature ranking + Plotly capture-rate charts by feature bin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from credit_fraud import config
from credit_fraud.io_utils import ensure_dirs, round_floats, save_dataframe, save_plotly_figure


@dataclass
class RankPlotResult:
    ranking: pd.DataFrame
    figures: dict[str, go.Figure]
    bin_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def univariate_auc(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """ROC-AUC of each feature alone (flip if AUC < 0.5)."""
    scores = {}
    for col in x.columns:
        values = x[col].to_numpy(dtype=float)
        if np.isnan(values).all() or np.nanstd(values) == 0:
            scores[col] = 0.5
            continue
        filled = np.where(np.isnan(values), np.nanmedian(values), values)
        try:
            auc = float(roc_auc_score(y, filled))
            scores[col] = max(auc, 1.0 - auc)
        except ValueError:
            scores[col] = 0.5
    return pd.Series(scores, name="univariate_auc").sort_values(ascending=False)


def mutual_info_scores(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    filled = x.fillna(x.median(numeric_only=True))
    mi = mutual_info_classif(filled, y, random_state=config.RANDOM_STATE)
    return pd.Series(mi, index=x.columns, name="mutual_info").sort_values(ascending=False)


def rank_features(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    auc = univariate_auc(x, y)
    mi = mutual_info_scores(x, y)
    mi_df = mi.reset_index()
    mi_df.columns = ["feature", "mutual_info"]
    ranking = pd.DataFrame({"feature": auc.index, "univariate_auc": auc.values}).merge(
        mi_df, on="feature", how="left"
    )
    ranking = ranking.sort_values("univariate_auc", ascending=False).reset_index(drop=True)
    ranking["rank_auc"] = np.arange(1, len(ranking) + 1)
    return round_floats(ranking)


def binned_capture_table(
    feature: pd.Series,
    y: pd.Series,
    n_bins: int | None = None,
) -> pd.DataFrame:
    """
    Bin an independent variable and compute counts + capture rate.

    capture_rate = positives in bin / all positives
    event_rate   = positives in bin / observations in bin
    """
    n_bins = n_bins or config.RANK_N_BINS
    frame = pd.DataFrame({"x": pd.to_numeric(feature, errors="coerce"), "y": y.to_numpy()})
    frame = frame.dropna(subset=["x"])
    if frame.empty:
        return pd.DataFrame(
            columns=["bin", "bin_label", "n_obs", "n_pos", "capture_rate", "event_rate", "x_mid"]
        )

    n_unique = frame["x"].nunique()
    bins = min(n_bins, max(n_unique, 1))
    # Quantile bins; duplicates='drop' handles low-cardinality / heavy ties
    try:
        frame["bin"] = pd.qcut(frame["x"], q=bins, duplicates="drop")
    except ValueError:
        frame["bin"] = pd.cut(frame["x"], bins=min(bins, n_unique), include_lowest=True)

    total_pos = float((frame["y"] == config.POSITIVE_LABEL).sum())
    rows = []
    for i, (interval, grp) in enumerate(frame.groupby("bin", observed=True), start=1):
        n_obs = int(len(grp))
        n_pos = int((grp["y"] == config.POSITIVE_LABEL).sum())
        capture = (n_pos / total_pos) if total_pos > 0 else 0.0
        event = (n_pos / n_obs) if n_obs > 0 else 0.0
        left = interval.left if hasattr(interval, "left") else grp["x"].min()
        right = interval.right if hasattr(interval, "right") else grp["x"].max()
        rows.append(
            {
                "bin": i,
                "bin_label": f"{left:.2f} – {right:.2f}",
                "n_obs": n_obs,
                "n_pos": n_pos,
                "capture_rate": round(capture, 2),
                "event_rate": round(event, 2),
                "x_mid": round(float(grp["x"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_capture_by_bin(
    feature_name: str,
    bin_table: pd.DataFrame,
) -> go.Figure:
    """
    Dual-axis Plotly chart:
    - x: independent-variable bins
    - bars: observation count per bin
    - line: capture rate per bin
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=bin_table["bin_label"],
            y=bin_table["n_obs"],
            name="n_obs",
            marker_color="#5B8FA8",
            opacity=0.75,
            hovertemplate="bin=%{x}<br>n_obs=%{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bin_table["bin_label"],
            y=bin_table["capture_rate"],
            name="capture_rate",
            mode="lines+markers",
            line=dict(color="#C45C26", width=2.5),
            marker=dict(size=8),
            hovertemplate="bin=%{x}<br>capture_rate=%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{feature_name}: count (bars) + capture rate (line)",
        xaxis_title=feature_name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
        margin=dict(l=60, r=60, t=70, b=80),
        height=420,
    )
    fig.update_yaxes(title_text="count (observations in bin)", secondary_y=False)
    fig.update_yaxes(title_text="capture rate", tickformat=".2f", secondary_y=True, range=[0, 1])
    fig.update_xaxes(tickangle=-35)
    return fig


def plot_capture_with_dropdown(
    x: pd.DataFrame,
    y: pd.Series,
    features: list[str] | None = None,
    n_bins: int | None = None,
) -> tuple[go.Figure, dict[str, pd.DataFrame]]:
    """
    One Plotly chart with a dropdown to choose any independent variable.

    Bars = bin counts, line = capture rate. Visibility toggles per feature.
    """
    n_bins = n_bins or config.RANK_N_BINS
    if features is None:
        features = list(x.columns)
    features = [f for f in features if f in x.columns]
    if not features:
        raise ValueError("No features available for capture dropdown plot.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bin_tables: dict[str, pd.DataFrame] = {}

    for i, feature in enumerate(features):
        table = binned_capture_table(x[feature], y, n_bins=n_bins)
        bin_tables[feature] = table
        visible = i == 0
        fig.add_trace(
            go.Bar(
                x=table["bin_label"],
                y=table["n_obs"],
                name="n_obs",
                marker_color="#5B8FA8",
                opacity=0.75,
                visible=visible,
                showlegend=(i == 0),
                legendgroup="n_obs",
                hovertemplate="bin=%{x}<br>n_obs=%{y}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=table["bin_label"],
                y=table["capture_rate"],
                name="capture_rate",
                mode="lines+markers",
                line=dict(color="#C45C26", width=2.5),
                marker=dict(size=8),
                visible=visible,
                showlegend=(i == 0),
                legendgroup="capture_rate",
                hovertemplate="bin=%{x}<br>capture_rate=%{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    n_traces = 2 * len(features)
    buttons = []
    for i, feature in enumerate(features):
        visible = [False] * n_traces
        visible[2 * i] = True
        visible[2 * i + 1] = True
        buttons.append(
            {
                "label": feature,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"{feature}: count (bars) + capture rate (line)",
                        "xaxis": {"title": feature, "tickangle": -35},
                    },
                ],
            }
        )

    first = features[0]
    fig.update_layout(
        title=f"{first}: count (bars) + capture rate (line)",
        xaxis_title=first,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
        margin=dict(l=60, r=60, t=100, b=80),
        height=460,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
                "pad": {"r": 0, "t": 0},
                "bgcolor": "#F7F7F7",
                "bordercolor": "#CCCCCC",
            }
        ],
        annotations=[
            {
                "text": "Feature:",
                "x": 1.0,
                "xref": "paper",
                "y": 1.22,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "bottom",
                "font": {"size": 12},
            }
        ],
    )
    fig.update_yaxes(title_text="count (observations in bin)", secondary_y=False)
    fig.update_yaxes(title_text="capture rate", tickformat=".2f", secondary_y=True, range=[0, 1])
    fig.update_xaxes(tickangle=-35)
    return fig, bin_tables


def rank_and_plot(
    x: pd.DataFrame,
    y: pd.Series,
    top_n: int | None = None,
    n_bins: int | None = None,
    save: bool = True,
) -> RankPlotResult:
    """
    Rank independent variables, then plot capture charts only:
    x = feature bins, bars = counts, line = capture rate (dropdown over features).
    """
    ensure_dirs()
    top_n = top_n or config.RANK_PLOT_TOP_N
    n_bins = n_bins or config.RANK_N_BINS

    ranking = rank_features(x, y)
    feature_order = ranking["feature"].tolist()
    dropdown_fig, bin_tables = plot_capture_with_dropdown(
        x, y, features=feature_order, n_bins=n_bins
    )

    figures: dict[str, go.Figure] = {"capture_dropdown": dropdown_fig}
    for feature in feature_order[:top_n]:
        figures[feature] = plot_capture_by_bin(feature, bin_tables[feature])

    if save:
        save_dataframe(ranking, config.RANKING_PATH)
        save_plotly_figure(figures["capture_dropdown"], "capture_dropdown")
        for feature in feature_order[:top_n]:
            safe = str(feature).replace("/", "_")
            save_dataframe(bin_tables[feature], config.ARTIFACTS_DIR / f"capture_bins_{safe}.csv")
            save_plotly_figure(figures[feature], f"capture_{safe}")

    return RankPlotResult(ranking=ranking, figures=figures, bin_tables=bin_tables)
