"""Plotly charts with room for labels."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from st_compat import plotly_chart


def heatmap(matrix: np.ndarray, labels: list[str], title: str, key: str, *, weights: bool = False) -> None:
    z = np.array(matrix, dtype=float)
    color = z.astype(float)
    text: list[list[str]] = []
    for i in range(z.shape[0]):
        row: list[str] = []
        for j in range(z.shape[1]):
            if not np.isfinite(z[i, j]):
                color[i, j] = np.nan
                row.append("mask")
            else:
                row.append(f"{z[i, j]:.3f}")
        text.append(row)
    heat_args: dict = dict(
        z=color,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale=[[0.0, "#faf6ef"], [1.0, "#c0392b"]],
        showscale=False,
        hoverinfo="skip",
        xgap=4,
        ygap=4,
    )
    if weights:
        heat_args["zmin"] = 0
        heat_args["zmax"] = 1
    fig = go.Figure(data=go.Heatmap(**heat_args))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        margin=dict(l=72, r=24, t=56, b=24),
        height=360,
        xaxis=dict(side="top", tickfont=dict(size=14), automargin=True),
        yaxis=dict(autorange="reversed", tickfont=dict(size=14), automargin=True),
    )
    plotly_chart(fig, key=key)


def bars(labels: list[str], values: list[float], title: str, key: str) -> None:
    peak = max(values) if values else 1.0
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#1F4E6A",
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        margin=dict(l=48, r=24, t=56, b=48),
        height=380,
        yaxis=dict(range=[0, max(peak * 1.35, 0.1)], title="probability", automargin=True),
        xaxis=dict(tickfont=dict(size=13), automargin=True),
        showlegend=False,
    )
    plotly_chart(fig, key=key)


def grouped_bars(labels: list[str], left: list[float], right: list[float], key: str) -> None:
    fig = go.Figure()
    fig.add_bar(name="before scaling", x=labels, y=left, marker_color="#C45C26")
    fig.add_bar(name="after scaling", x=labels, y=right, marker_color="#1F4E6A")
    fig.update_layout(
        barmode="group",
        title=dict(text="Softmax weights", font=dict(size=16)),
        margin=dict(l=48, r=24, t=72, b=48),
        height=400,
        yaxis=dict(range=[0, 1.2], title="weight", automargin=True),
        xaxis=dict(tickfont=dict(size=14), automargin=True),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    plotly_chart(fig, key=key)
