"""Plotly figures for the Optuna teaching lab and notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_trial_history(
    frame: pd.DataFrame,
    title: str = "CV PR-AUC by trial",
    baseline: float | None = None,
) -> go.Figure:
    """Trial score vs best-so-far. Optional horizontal line = random ranking (default rate)."""
    fig = go.Figure()
    if frame.empty or "value" not in frame.columns:
        fig.update_layout(title=title, template="plotly_white", height=360)
        return fig
    completed = frame.dropna(subset=["value"])
    fig.add_trace(
        go.Scatter(
            x=completed["number"],
            y=completed["value"],
            mode="markers+lines",
            name="trial PR-AUC",
            marker={"size": 9, "color": "#2F6F8F"},
            line={"color": "#2F6F8F"},
        )
    )
    if "best_so_far" in completed.columns:
        fig.add_trace(
            go.Scatter(
                x=completed["number"],
                y=completed["best_so_far"],
                mode="lines",
                name="best so far",
                line={"color": "#C45C26", "width": 3},
            )
        )
    if baseline is not None:
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="#888888",
            annotation_text=f"random ranking ≈ {baseline:.2f}",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title=title,
        xaxis_title="trial",
        yaxis_title="PR-AUC (higher is better)",
        template="plotly_white",
        height=380,
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def plot_search_scatter(
    tpe_frame: pd.DataFrame,
    random_frame: pd.DataFrame | None = None,
    x_col: str = "max_depth",
    y_col: str = "learning_rate",
    title: str = "Where trials landed",
) -> go.Figure:
    """TPE vs random in 2D. Later TPE points cluster; star = best TPE trial."""
    fig = go.Figure()

    def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        done = frame.dropna(subset=["value"]) if not frame.empty else frame
        if done is None or done.empty:
            empty = pd.DataFrame()
            return empty, empty, empty
        early = done[done["phase"] == "early"] if "phase" in done.columns else done.iloc[0:0]
        later = done[done["phase"] != "early"] if "phase" in done.columns else done
        best = done[done["is_best"]] if "is_best" in done.columns else done.nlargest(1, "value")
        return early, later, best

    tpe_early, tpe_later, tpe_best = _split(tpe_frame)
    if not tpe_early.empty and x_col in tpe_early.columns:
        fig.add_trace(
            go.Scatter(
                x=tpe_early[x_col],
                y=tpe_early[y_col],
                mode="markers",
                name="TPE early (random)",
                marker={"size": 11, "color": "#9BB8C9", "symbol": "circle", "line": {"width": 1, "color": "white"}},
                customdata=tpe_early[["number", "value"]],
                hovertemplate="TPE early #%{customdata[0]}<br>" + x_col + "=%{x}<br>" + y_col + "=%{y}<br>PR-AUC=%{customdata[1]:.3f}<extra></extra>",
            )
        )
    if not tpe_later.empty and x_col in tpe_later.columns:
        fig.add_trace(
            go.Scatter(
                x=tpe_later[x_col],
                y=tpe_later[y_col],
                mode="markers",
                name="TPE later",
                marker={"size": 12, "color": "#C45C26", "symbol": "circle", "line": {"width": 1, "color": "white"}},
                customdata=tpe_later[["number", "value"]],
                hovertemplate="TPE later #%{customdata[0]}<br>" + x_col + "=%{x}<br>" + y_col + "=%{y}<br>PR-AUC=%{customdata[1]:.3f}<extra></extra>",
            )
        )
    if random_frame is not None and not random_frame.empty and x_col in random_frame.columns:
        rnd = random_frame.dropna(subset=["value"])
        fig.add_trace(
            go.Scatter(
                x=rnd[x_col],
                y=rnd[y_col],
                mode="markers",
                name="Random",
                marker={"size": 10, "color": "#1B4332", "symbol": "x"},
                customdata=rnd[["number", "value"]],
                hovertemplate="Random #%{customdata[0]}<br>" + x_col + "=%{x}<br>" + y_col + "=%{y}<br>PR-AUC=%{customdata[1]:.3f}<extra></extra>",
            )
        )
    if not tpe_best.empty and x_col in tpe_best.columns:
        fig.add_trace(
            go.Scatter(
                x=tpe_best[x_col],
                y=tpe_best[y_col],
                mode="markers",
                name="TPE best",
                marker={"size": 18, "color": "#C45C26", "symbol": "star", "line": {"width": 1, "color": "white"}},
                customdata=tpe_best[["number", "value"]],
                hovertemplate="BEST #%{customdata[0]}<br>" + x_col + "=%{x}<br>" + y_col + "=%{y}<br>PR-AUC=%{customdata[1]:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=460,
        legend={"orientation": "h", "y": 1.08},
    )
    if y_col == "learning_rate":
        fig.update_yaxes(type="log")
    return fig


def plot_param_scatter(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str | None = None,
) -> go.Figure:
    """Early vs later trials; star = best."""
    fig = go.Figure()
    completed = frame.dropna(subset=["value"])
    if completed.empty or x_col not in completed.columns or y_col not in completed.columns:
        fig.update_layout(template="plotly_white", height=360)
        return fig
    early = completed[completed["phase"] == "early"] if "phase" in completed.columns else completed.iloc[0:0]
    later = completed[completed["phase"] != "early"] if "phase" in completed.columns else completed
    best = completed[completed.get("is_best", False)] if "is_best" in completed.columns else completed.nlargest(1, "value")

    if not early.empty:
        fig.add_trace(
            go.Scatter(
                x=early[x_col],
                y=early[y_col],
                mode="markers+text",
                text=early["number"].astype(str),
                textposition="top center",
                name="early (random)",
                marker={"size": 12, "color": "#9BB8C9", "line": {"width": 1, "color": "white"}},
                hovertemplate="trial %{text}<br>" + x_col + "=%{x}<br>" + y_col + "=%{y}<br>PR-AUC=%{marker.color}<extra></extra>",
            )
        )
    if not later.empty:
        fig.add_trace(
            go.Scatter(
                x=later[x_col],
                y=later[y_col],
                mode="markers+text",
                text=later["number"].astype(str),
                textposition="top center",
                name="later",
                marker={
                    "size": 12,
                    "color": later["value"],
                    "colorscale": "YlOrRd",
                    "showscale": True,
                    "colorbar": {"title": "PR-AUC"},
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="trial %{text}<br>" + x_col + "=%{x}<br>" + y_col + "=%{y}<br>PR-AUC=%{marker.color:.3f}<extra></extra>",
            )
        )
    if not best.empty:
        fig.add_trace(
            go.Scatter(
                x=best[x_col],
                y=best[y_col],
                mode="markers",
                name="best",
                marker={"size": 18, "color": "#C45C26", "symbol": "star", "line": {"width": 1, "color": "white"}},
            )
        )
    fig.update_layout(
        title=title or f"{y_col} vs {x_col} — star is best",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=400,
        legend={"orientation": "h", "y": 1.08},
    )
    if y_col == "learning_rate":
        fig.update_yaxes(type="log")
    if x_col == "learning_rate":
        fig.update_xaxes(type="log")
    return fig


def plot_tpe_vs_random_history(
    tpe_frame: pd.DataFrame,
    random_frame: pd.DataFrame,
    baseline: float | None = None,
) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("TPE best so far", "Random best so far"))
    for col, frame, color in ((1, tpe_frame, "#C45C26"), (2, random_frame, "#1B4332")):
        completed = frame.dropna(subset=["value"]) if not frame.empty else frame
        if completed is None or completed.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=completed["number"],
                y=completed["value"],
                mode="markers",
                name="trial",
                marker={"color": "#2F6F8F", "size": 8},
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        if "best_so_far" in completed.columns:
            fig.add_trace(
                go.Scatter(
                    x=completed["number"],
                    y=completed["best_so_far"],
                    mode="lines",
                    name="best so far",
                    line={"color": color, "width": 3},
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        if baseline is not None:
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color="#888888",
                row=1,
                col=col,
            )
    fig.update_xaxes(title_text="trial")
    fig.update_yaxes(title_text="PR-AUC")
    fig.update_layout(template="plotly_white", height=360, legend={"orientation": "h", "y": 1.12})
    return fig


def plot_tpe_parzen_example() -> go.Figure:
    """Sketch of good vs bad Parzen densities for learning rate (teaching figure)."""
    x = np.linspace(0.01, 0.35, 400)
    good = np.exp(-0.5 * ((x - 0.06) / 0.018) ** 2)
    bad = np.exp(-0.5 * ((x - 0.22) / 0.035) ** 2)
    good = good / good.max()
    bad = bad / bad.max() * 0.85
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=good,
            name="l(x)  good  P(x | y > y*)",
            fill="tozeroy",
            line={"color": "#2F6F8F"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bad,
            name="g(x)  bad  P(x | y ≤ y*)",
            fill="tozeroy",
            line={"color": "#C45C26"},
        )
    )
    fig.add_vline(x=0.05, line_dash="dot", line_color="#2F6F8F", annotation_text="prefer ~0.05")
    fig.update_layout(
        title="Parzen densities for learning rate",
        xaxis_title="learning rate",
        yaxis_title="relative density (sketch)",
        template="plotly_white",
        height=380,
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def plot_search_methods_overview() -> go.Figure:
    """Visual comparison of grid, random, and Bayesian search."""
    methods = ["Grid", "Random", "Bayesian (TPE)"]
    tries = [100, 40, 25]
    quality = [55, 45, 85]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Trials needed (lower is better)", "Finds good region (higher is better)"),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(
            x=methods,
            y=tries,
            marker_color=["#C45C26", "#888888", "#5A8F6F"],
            text=tries,
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=methods,
            y=quality,
            marker_color=["#C45C26", "#888888", "#5A8F6F"],
            text=[f"{q}%" for q in quality],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="relative trial budget", row=1, col=1)
    fig.update_yaxes(title_text="relative success", range=[0, 100], row=1, col=2)
    fig.update_layout(
        title="Grid explodes · random wanders · TPE learns from past trials",
        template="plotly_white",
        height=360,
        margin={"t": 80, "b": 40},
    )
    return fig


def plot_pr_auc_intuition(*, positive_rate: float = 0.22) -> go.Figure:
    """PR curve sketch: random ranking vs a model that ranks defaults near the top."""
    recall = np.linspace(0.0, 1.0, 80)
    random_pr = np.full_like(recall, positive_rate)
    good_pr = positive_rate + (1.0 - positive_rate) * (1.0 - (1.0 - recall) ** 1.4)
    good_pr = np.clip(good_pr, positive_rate, 1.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=random_pr,
            mode="lines",
            name=f"random ranking ≈ {positive_rate:.0%}",
            line={"color": "#888888", "dash": "dash", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=good_pr,
            mode="lines",
            name="good model (defaults near top)",
            line={"color": "#2F6F8F", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(47, 111, 143, 0.15)",
        )
    )
    fig.add_annotation(
        x=0.55,
        y=0.52,
        text="PR-AUC ≈ shaded area",
        showarrow=True,
        ax=0.2,
        ay=-40,
        font={"size": 11, "color": "#2F6F8F"},
    )
    fig.add_annotation(
        x=0.75,
        y=positive_rate + 0.02,
        text=f"always predict 'no default'<br>accuracy ≈ {1 - positive_rate:.0%} but useless",
        showarrow=False,
        font={"size": 10, "color": "#666666"},
        xanchor="left",
    )
    fig.update_layout(
        title="Why PR-AUC on ~22% default rate",
        xaxis_title="recall (fraction of defaults caught)",
        yaxis_title="precision at that cutoff",
        template="plotly_white",
        height=380,
        legend={"orientation": "h", "y": 1.12},
        yaxis={"range": [0, 1.05]},
    )
    return fig


def plot_tuning_task_overview(*, positive_rate: float = 0.22) -> go.Figure:
    """One glance: imbalanced task, fixed features, tuned XGB knobs, PR-AUC score."""
    fig = go.Figure()
    blocks = [
        (0.0, "Task", f"~{positive_rate:.0%} default\nnext month", "#C45C26"),
        (1.35, "Fixed", "features\n(engineered)", "#888888"),
        (2.7, "Tuned", "XGB depth\nlr · trees", "#2F6F8F"),
        (4.05, "Score", "CV\nPR-AUC", "#5A8F6F"),
    ]
    for x, title, body, fill in blocks:
        fig.add_shape(
            type="rect",
            x0=x,
            y0=0.2,
            x1=x + 1.1,
            y1=1.0,
            fillcolor=fill,
            line={"color": fill, "width": 0},
            layer="below",
        )
        fig.add_annotation(
            x=x + 0.55,
            y=0.78,
            text=f"<b>{title}</b>",
            showarrow=False,
            font={"size": 12, "color": "white"},
        )
        fig.add_annotation(
            x=x + 0.55,
            y=0.45,
            text=body.replace("\n", "<br>"),
            showarrow=False,
            font={"size": 10, "color": "white"},
        )
        if x < blocks[-1][0]:
            fig.add_annotation(
                x=x + 1.15,
                y=0.6,
                ax=x + 1.25,
                ay=0.6,
                showarrow=True,
                arrowhead=2,
                arrowcolor="#aaaaaa",
                text="",
            )
    fig.update_layout(
        title="What this lab tunes — and what stays fixed",
        template="plotly_white",
        height=240,
        xaxis={"visible": False, "range": [-0.2, 5.3]},
        yaxis={"visible": False, "range": [0.0, 1.2]},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        showlegend=False,
    )
    return fig
