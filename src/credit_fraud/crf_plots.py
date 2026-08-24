"""Plotly figures for the HMM / Viterbi / CRF teaching lab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from credit_fraud.crf import TripletCRF
from credit_fraud.hmm import PosTaggingHMM, ToyHMM

_PALETTE = {
    "DET": "#2F6F8F",
    "NOUN": "#C45C26",
    "VERB": "#5A8F6F",
    "Happy": "#5A8F6F",
    "Sad": "#8F6FA8",
    "START": "#2F6F8F",
    "accent": "#2F6F8F",
    "observation": "#F4F4F4",
}

_HEATMAP_TEXT = {"size": 20}
_AXIS_TICK = {"size": 14}


def _tag_color(tag: str) -> str:
    return _PALETTE.get(tag, "#666666")


def _add_node(
    fig: go.Figure,
    x: float,
    y: float,
    label: str,
    *,
    fill: str,
    text_color: str = "white",
    size: int = 14,
) -> None:
    fig.add_annotation(
        x=x,
        y=y,
        text=f"<b>{label}</b>",
        showarrow=False,
        font={"size": size, "color": text_color},
        bgcolor=fill,
        bordercolor=fill,
        borderwidth=2,
        borderpad=10,
        yanchor="middle",
        xanchor="center",
    )


def _add_arrow(
    fig: go.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str,
    *,
    color: str = "#444444",
    label_offset: tuple[float, float] = (0.0, 0.0),
    label_xy: tuple[float, float] | None = None,
    curve: float = 0.0,
) -> None:
    """Draw a labeled arrow; optional curve via ax/ay bend."""
    if label_xy is None:
        label_x = (x0 + x1) / 2 + label_offset[0]
        label_y = (y0 + y1) / 2 + label_offset[1]
    else:
        label_x, label_y = label_xy
    if curve:
        ax = x0 + curve
        ay = y0 + curve
    else:
        ax, ay = x0, y0
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=ax,
        ay=ay,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=color,
        text="",
    )
    fig.add_annotation(
        x=label_x,
        y=label_y,
        text=label,
        showarrow=False,
        font={"size": 11, "color": color},
        bgcolor="#ffffff",
        bordercolor="#dddddd",
        borderwidth=1,
        borderpad=3,
        yanchor="middle",
        xanchor="center",
    )


def _label_on_arrow(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    t: float = 0.38,
    side: float = 0.0,
) -> tuple[float, float]:
    """Point at fraction t along an arrow, offset perpendicular to the line."""
    px = x0 + t * (x1 - x0)
    py = y0 + t * (y1 - y0)
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy)) or 1.0
    nx, ny = -dy / length, dx / length
    return px + side * nx, py + side * ny


def _add_self_loop(
    fig: go.Figure,
    x: float,
    y: float,
    label: str,
    *,
    color: str = "#666666",
    side: str = "left",
) -> None:
    """Small curved loop sitting directly above a mood node (no arrowhead — curve is enough)."""
    y_top = y + 0.24
    if side == "left":
        xs = [x - 0.16, x - 0.34, x - 0.08, x + 0.16]
        ys = [y_top, y + 0.52, y + 0.56, y_top]
        label_x, label_y = x - 0.18, y + 0.62
    else:
        xs = [x + 0.16, x + 0.34, x + 0.08, x - 0.16]
        ys = [y_top, y + 0.52, y + 0.56, y_top]
        label_x, label_y = x + 0.18, y + 0.62

    t = np.linspace(0.0, 1.0, 40)
    omt = 1.0 - t
    loop_x = (
        omt**3 * xs[0]
        + 3 * omt**2 * t * xs[1]
        + 3 * omt * t**2 * xs[2]
        + t**3 * xs[3]
    )
    loop_y = (
        omt**3 * ys[0]
        + 3 * omt**2 * t * ys[1]
        + 3 * omt * t**2 * ys[2]
        + t**3 * ys[3]
    )

    fig.add_trace(
        go.Scatter(
            x=loop_x,
            y=loop_y,
            mode="lines",
            line={"color": color, "width": 2.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=label_x,
        y=label_y,
        text=label,
        showarrow=False,
        font={"size": 11, "color": color},
        bgcolor="#ffffff",
        bordercolor="#dddddd",
        borderwidth=1,
        borderpad=3,
        yanchor="middle",
        xanchor="center",
    )


def plot_hmm_graph(hmm: ToyHMM, *, title: str = "Hidden mood, visible shirt colors") -> go.Figure:
    """Professor happy/sad HMM — START, cross-mood transitions, and emissions."""
    fig = go.Figure()

    state_colors = {"Happy": _PALETTE["Happy"], "Sad": _PALETTE["Sad"]}
    pos = {
        "START": (1.0, 3.2),
        "Happy": (0.0, 2.0),
        "Sad": (2.0, 2.0),
        "red": (0.0, 0.3),
        "green": (1.0, 0.3),
        "blue": (2.0, 0.3),
    }

    _add_node(fig, *pos["START"], "START", fill=_PALETTE["START"])
    for state in hmm.states:
        _add_node(fig, *pos[state], state, fill=state_colors[state])
    for obs in hmm.observations:
        _add_node(
            fig,
            *pos[obs],
            obs,
            fill=_PALETTE["observation"],
            text_color="#222222",
            size=13,
        )

    happy_i, sad_i = hmm.states.index("Happy"), hmm.states.index("Sad")
    start_probs = {hmm.states[i]: float(hmm.start[i]) for i in range(len(hmm.states))}
    trans = hmm.trans

    happy_top = pos["Happy"][1] + 0.25
    sad_top = pos["Sad"][1] + 0.25
    start_x, start_y = pos["START"]

    # Emissions first (bottom layer) — labels near the source mood side.
    emit_side: dict[tuple[str, str], float] = {
        ("Happy", "red"): -0.14,
        ("Sad", "red"): 0.14,
        ("Happy", "green"): -0.18,
        ("Sad", "green"): 0.18,
        ("Happy", "blue"): -0.14,
        ("Sad", "blue"): 0.14,
    }
    for s_idx, state in enumerate(hmm.states):
        sx, sy = pos[state]
        for o_idx, obs in enumerate(hmm.observations):
            ox, oy = pos[obs]
            prob = float(hmm.emit[s_idx, o_idx])
            ex0, ey0 = sx, sy - 0.25
            ex1, ey1 = ox, oy + 0.25
            label_xy = _label_on_arrow(ex0, ey0, ex1, ey1, side=emit_side[(state, obs)])
            _add_arrow(
                fig,
                ex0,
                ey0,
                ex1,
                ey1,
                f"{prob:.0%}",
                color=state_colors[state],
                label_xy=label_xy,
            )

    _add_arrow(
        fig,
        pos["Happy"][0] + 0.35,
        pos["Happy"][1],
        pos["Sad"][0] - 0.35,
        pos["Sad"][1],
        f"{trans[happy_i, sad_i]:.0%}",
        color="#666666",
        label_offset=(0.0, 0.18),
    )
    _add_arrow(
        fig,
        pos["Sad"][0] - 0.35,
        pos["Sad"][1] - 0.12,
        pos["Happy"][0] + 0.35,
        pos["Happy"][1] - 0.12,
        f"{trans[sad_i, happy_i]:.0%}",
        color="#666666",
        label_offset=(0.0, -0.22),
    )

    _add_self_loop(
        fig,
        pos["Happy"][0],
        pos["Happy"][1],
        f"{trans[happy_i, happy_i]:.0%}",
        color=_PALETTE["Happy"],
        side="left",
    )
    _add_self_loop(
        fig,
        pos["Sad"][0],
        pos["Sad"][1],
        f"{trans[sad_i, sad_i]:.0%}",
        color=_PALETTE["Sad"],
        side="right",
    )

    # START arrows last — labels sit on the diagonal from START, not on self-loops.
    _add_arrow(
        fig,
        start_x - 0.15,
        start_y - 0.2,
        pos["Happy"][0],
        happy_top,
        f"{start_probs['Happy']:.0%}",
        color=_PALETTE["Happy"],
        label_xy=(0.28, 2.88),
    )
    _add_arrow(
        fig,
        start_x + 0.15,
        start_y - 0.2,
        pos["Sad"][0],
        sad_top,
        f"{start_probs['Sad']:.0%}",
        color=_PALETTE["Sad"],
        label_xy=(1.72, 2.88),
    )

    fig.add_annotation(
        x=1.0,
        y=-0.55,
        text=(
            "<b>Assumptions:</b> mood tomorrow ← mood today only · "
            "shirt today ← mood today only · curved loops = stay in same mood"
        ),
        showarrow=False,
        font={"size": 11, "color": "#555555"},
        xanchor="center",
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=560,
        xaxis={"visible": False, "range": [-0.8, 2.8]},
        yaxis={"visible": False, "range": [-0.8, 3.5]},
        margin={"l": 20, "r": 20, "t": 60, "b": 60},
        showlegend=False,
    )
    return fig


def plot_hmm_matrices(
    hmm: ToyHMM,
    *,
    title: str = "Transition and emission probabilities",
    highlight_obs: str | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Transition — P(next mood | current mood) [Y: Prev → X: Next]",
            "Emission — P(shirt color | mood)",
        ),
        horizontal_spacing=0.18,
    )
    fig.add_trace(
        go.Heatmap(
            z=hmm.trans,
            x=[f"→ {s} (Next)" for s in hmm.states],
            y=[f"(Prev) {s} →" for s in hmm.states],
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            text=np.round(hmm.trans, 2),
            texttemplate="%{text}",
            textfont=_HEATMAP_TEXT,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=hmm.emit,
            x=hmm.observations,
            y=hmm.states,
            colorscale="Greens",
            zmin=0.0,
            zmax=1.0,
            text=np.round(hmm.emit, 2),
            texttemplate="%{text}",
            textfont=_HEATMAP_TEXT,
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=560,
        margin={"t": 100, "b": 80},
        annotations=[
            *list(fig.layout.annotations),
            dict(
                text="Rows (Y axis) = Previous mood · Cols (X axis) = Next mood · each row sums to 1",
                xref="paper",
                yref="paper",
                x=0.22,
                y=-0.14,
                showarrow=False,
                font={"size": 11, "color": "#666666"},
            ),
            dict(
                text="Each row = mood · each column = shirt color · each row sums to 1",
                xref="paper",
                yref="paper",
                x=0.78,
                y=-0.14,
                showarrow=False,
                font={"size": 11, "color": "#666666"},
            ),
        ],
    )
    for col in (1, 2):
        fig.update_xaxes(tickfont=_AXIS_TICK, row=1, col=col)
        fig.update_yaxes(tickfont=_AXIS_TICK, row=1, col=col)

    if highlight_obs and highlight_obs in hmm.observations:
        obs_idx = hmm.observations.index(highlight_obs)
        fig.add_annotation(
            x=obs_idx,
            y=-0.55,
            xref="x2",
            yref="y2 domain",
            text=f"← today: {highlight_obs}",
            showarrow=False,
            font={"size": 12, "color": "#C45C26"},
            xanchor="center",
        )
        for mood_idx in range(len(hmm.states)):
            fig.add_shape(
                type="rect",
                x0=obs_idx - 0.5,
                y0=mood_idx - 0.5,
                x1=obs_idx + 0.5,
                y1=mood_idx + 0.5,
                xref="x2",
                yref="y2",
                line={"color": "#C45C26", "width": 3},
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

    return fig


def plot_hmm_two_paths_compare(
    hmm: ToyHMM,
    obs_seq: list[str],
    path_a: list[str],
    path_b: list[str],
    *,
    title: str = "Two mood paths — joint probability comparison",
) -> go.Figure:
    """Side-by-side bar chart for the top two mood paths."""
    labels = [" → ".join(path_a), " → ".join(path_b)]
    probs = [
        hmm.path_joint_probability(obs_seq, path_a),
        hmm.path_joint_probability(obs_seq, path_b),
    ]
    colors = [_tag_color(path_a[0]), "#d8d8d8"]
    if probs[1] > probs[0]:
        colors = ["#d8d8d8", _tag_color(path_b[0])]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=probs,
            marker_color=colors,
            text=[f"{p:.4f}" for p in probs],
            textposition="outside",
            textfont={"size": 13},
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=360,
        yaxis_title="joint probability",
        margin={"t": 70, "b": 60},
    )
    return fig


def plot_hmm_all_paths(
    hmm: ToyHMM,
    obs_seq: list[str],
    *,
    best_path: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Bar chart of every mood path's joint probability (brute force)."""
    frame = hmm.enumerate_all_paths(obs_seq)
    if title is None:
        n_paths = len(frame)
        title = f"All {n_paths} mood paths — multiply table probabilities along each path"
    if best_path is None:
        best_path, _ = hmm.viterbi(obs_seq)
    best_label = " → ".join(best_path)
    colors = [
        "#1a1a1a" if path == best_label else _tag_color(path.split(" → ")[0])
        for path in frame["mood path"]
    ]
    fig = go.Figure(
        go.Bar(
            y=frame["mood path"],
            x=frame["joint probability"],
            orientation="h",
            marker_color=colors,
            text=[f"{p:.4f}" for p in frame["joint probability"]],
            textposition="outside",
            textfont={"size": 13},
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(360, 46 * len(frame)),
        xaxis_title="joint probability (higher = better fit)",
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 120, "r": 40, "t": 70, "b": 40},
    )
    return fig


def plot_hmm_day_viterbi(
    hmm: ToyHMM,
    obs_seq: list[str],
    day_index: int,
    candidates: pd.DataFrame,
) -> go.Figure:
    """One day's candidate scores as bars; winners highlighted."""
    obs = obs_seq[day_index]
    day_label = f"Day {day_index + 1} — shirt: {obs}"

    if day_index == 0:
        labels = [str(m) for m in candidates["mood"]]
        colors = [_tag_color(m) for m in candidates["mood"]]
    else:
        labels = [f"{row['mood']} ← {row['from']}" for _, row in candidates.iterrows()]
        winners = candidates.loc[candidates.groupby("mood")["probability"].idxmax()]
        winner_pairs = set(zip(winners["mood"], winners["from"]))
        colors = [
            _tag_color(row["mood"])
            if (row["mood"], row["from"]) in winner_pairs
            else "#d8d8d8"
            for _, row in candidates.iterrows()
        ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=candidates["probability"],
            marker_color=colors,
            text=[f"{p:.3f}" for p in candidates["probability"]],
            textposition="outside",
            textfont={"size": 13},
        )
    )
    winner_note = ""
    if day_index > 0:
        winners = candidates.loc[candidates.groupby("mood")["probability"].idxmax()]
        parts = [f"{r['mood']}←{r['from']} ({r['probability']:.3f})" for _, r in winners.iterrows()]
        winner_note = " · keep best per mood: " + " · ".join(parts)
    fig.update_layout(
        title=f"{day_label}{winner_note}",
        template="plotly_white",
        height=380,
        yaxis_title="probability",
        xaxis_title="candidate path ending here",
        margin={"t": 90, "b": 60},
    )
    return fig


def plot_hmm_viterbi_pruning(
    hmm: ToyHMM,
    obs_seq: list[str],
    day_index: int,
) -> go.Figure:
    """Show two routes into the same mood; only the stronger partial path can win."""
    candidates = hmm.viterbi_day_candidates_prob(obs_seq, day_index)
    mood = str(candidates.loc[candidates["probability"].idxmax(), "mood"])
    group = candidates[candidates["mood"] == mood].sort_values("probability", ascending=False)
    top = group.iloc[0]
    second = group.iloc[1] if len(group) > 1 else None

    fig = go.Figure()
    kept_label = f"{top['from']} → {mood}"
    drop_label = f"{second['from']} → {mood}" if second is not None else None
    fig.add_trace(
        go.Bar(
            x=[kept_label, drop_label] if second is not None else [kept_label],
            y=[top["probability"], second["probability"]] if second is not None else [top["probability"]],
            marker_color=[_tag_color(mood), "#d8d8d8"] if second is not None else [_tag_color(mood)],
            text=[
                f"{top['probability']:.3f} ✓ keep",
                f"{second['probability']:.3f} ✗ drop" if second is not None else None,
            ],
            textposition="outside",
            textfont={"size": 13},
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"Day {day_index + 1}: scores into {mood} — keep vs drop",
        template="plotly_white",
        height=340,
        yaxis_title="partial joint probability",
        margin={"t": 80, "b": 50},
    )
    return fig


def plot_hmm_viterbi_keep_drop(
    hmm: ToyHMM,
    obs_seq: list[str],
    day_index: int,
) -> go.Figure:
    """Graph of two incoming routes: one kept, one dropped because later days share factors."""
    info = hmm.viterbi_pruning_pair(obs_seq, day_index)
    mood = str(info["mood"])
    kept_from = str(info["kept_from"])
    drop_from = str(info["drop_from"])
    day = int(info["day"])
    prev_day = day - 1
    obs = str(info["shirt"])
    prev_obs = str(info["prev_shirt"])

    fig = go.Figure()
    pos = {
        "kept_src": (0.0, 1.85),
        "drop_src": (0.0, 0.35),
        "dest": (2.15, 1.1),
        "future": (4.15, 1.1),
    }

    def _line(x0: float, y0: float, x1: float, y1: float, *, color: str, dash: str, width: float) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"color": color, "width": width, "dash": dash},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    keep_color = _tag_color(mood)
    _line(*pos["kept_src"], *pos["dest"], color=keep_color, dash="solid", width=4)
    _line(*pos["drop_src"], *pos["dest"], color="#9a9a9a", dash="dash", width=2.5)
    _line(*pos["dest"], *pos["future"], color="#2F6F8F", dash="solid", width=3)

    _add_node(
        fig,
        *pos["kept_src"],
        f"{kept_from}<br>day {prev_day}",
        fill=_tag_color(kept_from),
        size=13,
    )
    _add_node(
        fig,
        *pos["drop_src"],
        f"{drop_from}<br>day {prev_day}",
        fill="#b0b0b0",
        size=13,
    )
    _add_node(fig, *pos["dest"], f"{mood}<br>day {day}", fill=keep_color, size=13)
    _add_node(
        fig,
        *pos["future"],
        "later days",
        fill="#2F6F8F",
        size=13,
    )

    fig.add_annotation(
        x=1.05,
        y=1.72,
        text=f"<b>✓ keep</b> {kept_from} → {mood} ({info['kept_prob']:.3f})",
        showarrow=False,
        font={"size": 13, "color": keep_color},
        bgcolor="#ffffff",
        bordercolor=keep_color,
        borderwidth=1,
        borderpad=4,
    )
    fig.add_annotation(
        x=1.05,
        y=0.52,
        text=f"<b>✗ drop</b> {drop_from} → {mood} ({info['drop_prob']:.3f})",
        showarrow=False,
        font={"size": 13, "color": "#666666"},
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=4,
    )
    fig.add_annotation(
        x=3.15,
        y=1.52,
        text="same multipliers",
        showarrow=False,
        font={"size": 12, "color": "#2F6F8F"},
        bgcolor="#ffffff",
        bordercolor="#dddddd",
        borderwidth=1,
        borderpad=3,
    )
    fig.add_annotation(
        x=2.15,
        y=-0.25,
        text=(
            f"Shirt day {prev_day}: <b>{prev_obs}</b> · shirt day {day}: <b>{obs}</b><br>"
            f"Both routes end in <b>{mood}</b>, so day {day + 1}+ multiplies both by the same numbers. "
            f"The weaker route can never catch up."
        ),
        showarrow=False,
        font={"size": 12, "color": "#444444"},
        xanchor="center",
    )

    fig.update_layout(
        title=f"Why keep {kept_from} → {mood} and drop {drop_from} → {mood}",
        template="plotly_white",
        height=420,
        xaxis={"visible": False, "range": [-0.7, 5.0]},
        yaxis={"visible": False, "range": [-0.55, 2.55]},
        margin={"l": 20, "r": 20, "t": 60, "b": 70},
        showlegend=False,
    )
    return fig


def plot_hmm_winning_path(
    hmm: ToyHMM,
    obs_seq: list[str],
    path: list[str],
) -> go.Figure:
    """Timeline of winning moods, shirts, and running joint probability."""
    steps = hmm.viterbi_step_table_prob(obs_seq, path)
    labels = [f"Day {i + 1}<br>{obs}<br>{mood}" for i, (obs, mood) in enumerate(zip(obs_seq, path))]
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=("Winning mood each day", "Running joint probability"),
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(path))),
            y=[0.0] * len(path),
            mode="markers+text",
            text=path,
            textposition="top center",
            marker={
                "size": 28,
                "color": [_tag_color(m) for m in path],
                "line": {"width": 2, "color": "#1a1a1a"},
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(steps))),
            y=steps["running probability"],
            mode="lines+markers+text",
            text=[f"{p:.3f}" for p in steps["running probability"]],
            textposition="top center",
            line={"color": "#2F6F8F", "width": 3},
            marker={"size": 12, "color": "#2F6F8F"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(path))),
        ticktext=labels,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(steps))),
        ticktext=[f"Day {i + 1}" for i in range(len(steps))],
        row=1,
        col=2,
    )
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text="P(path, shirts so far)", row=1, col=2)
    fig.update_layout(
        title=f"Best path: {' → '.join(path)}",
        template="plotly_white",
        height=380,
        margin={"t": 80, "b": 60},
    )
    return fig


def _annotate_trellis_path(
    fig: go.Figure,
    obs_labels: list[str],
    path: list[str],
    trellis: np.ndarray,
    states: list[str],
    *,
    value_fmt: str = ".3f",
    value_label: str = "P",
) -> None:
    """Label winning path nodes with state name and trellis value."""
    n = len(obs_labels)
    for i in range(n):
        state = path[i]
        s_idx = states.index(state)
        score = float(trellis[i, s_idx])
        fig.add_annotation(
            x=i,
            y=score,
            text=f"<b>{state}</b><br>{value_label}={score:{value_fmt}}",
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=-35,
            font={"size": 10, "color": _tag_color(state)},
            bgcolor="#ffffff",
            bordercolor=_tag_color(state),
            borderwidth=1,
            borderpad=3,
        )


def _add_trellis_winning_path(
    fig: go.Figure,
    path: list[str],
    trellis: np.ndarray,
    states: list[str],
) -> None:
    """Overlay a thick line connecting only the winning path nodes."""
    n = len(path)
    path_y = [float(trellis[i, states.index(path[i])]) for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=path_y,
            mode="lines+markers",
            name="Best path",
            line={"color": "#1a1a1a", "width": 5},
            marker={
                "size": 16,
                "color": [_tag_color(path[i]) for i in range(n)],
                "line": {"width": 2.5, "color": "#1a1a1a"},
            },
        )
    )


def plot_hmm_viterbi_trellis(
    hmm: ToyHMM,
    obs_seq: list[str],
    path: list[str] | None = None,
    *,
    title: str = "Viterbi trellis — best partial probability per mood",
    use_probability: bool = True,
) -> go.Figure:
    if path is None:
        path, log_trellis = hmm.viterbi(obs_seq)
    else:
        _, log_trellis = hmm.viterbi(obs_seq)

    trellis = np.exp(log_trellis) if use_probability else log_trellis
    y_title = (
        "best partial joint probability (higher = more likely)"
        if use_probability
        else "log score (higher = more likely)"
    )
    value_fmt = ".3f" if use_probability else ".2f"
    value_label = "P" if use_probability else "log"

    states = hmm.states
    n = len(obs_seq)
    fig = go.Figure()
    for s_idx, state in enumerate(states):
        y = trellis[:, s_idx]
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=y,
                mode="lines+markers",
                name=f"{state} (hidden mood)",
                line={"color": _tag_color(state), "width": 2.5},
                marker={"size": 10, "color": _tag_color(state)},
            )
        )
    _add_trellis_winning_path(fig, path, trellis, states)
    _annotate_trellis_path(
        fig,
        obs_seq,
        path,
        trellis,
        states,
        value_fmt=value_fmt,
        value_label=value_label,
    )

    ticktext = [f"Day {i + 1}<br>shirt: {obs}" for i, obs in enumerate(obs_seq)]
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=480,
        margin={"t": 90},
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(n)),
            "ticktext": ticktext,
            "title": "day (you only see the shirt color)",
        },
        yaxis_title=y_title,
        legend={
            "orientation": "h",
            "y": 1.12,
            "title": {"text": "Thick line = best overall path from step-by-step pruning"},
        },
    )
    return fig


def plot_pos_transition_matrix(hmm: PosTaggingHMM) -> go.Figure:
    """Tag-to-tag transition heatmap only."""
    fig = go.Figure(
        go.Heatmap(
            z=hmm.trans,
            x=[f"→ {t}" for t in hmm.tags],
            y=[f"{t} →" for t in hmm.tags],
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            text=np.round(hmm.trans, 2),
            texttemplate="%{text}",
            textfont=_HEATMAP_TEXT,
            showscale=False,
        )
    )
    fig.update_layout(
        title="Transition — P(next tag | previous tag)",
        template="plotly_white",
        height=460,
        margin={"t": 70, "b": 40},
        xaxis_title="next tag",
        yaxis_title="previous tag",
    )
    fig.update_xaxes(tickfont=_AXIS_TICK)
    fig.update_yaxes(tickfont=_AXIS_TICK)
    return fig


def plot_pos_emission_matrix(hmm: PosTaggingHMM) -> go.Figure:
    """Tag-to-word emission heatmap only."""
    fig = go.Figure(
        go.Heatmap(
            z=hmm.emit,
            x=[f"{w} ({i + 1})" for i, w in enumerate(hmm.sentence)],
            y=hmm.tags,
            colorscale="Greens",
            zmin=0.0,
            zmax=1.0,
            text=np.round(hmm.emit, 2),
            texttemplate="%{text}",
            textfont=_HEATMAP_TEXT,
            showscale=False,
        )
    )
    fig.update_layout(
        title="Emission — P(word | tag at this word)",
        template="plotly_white",
        height=460,
        margin={"t": 70, "b": 40},
        xaxis_title="word position in sentence",
        yaxis_title="tag (if this word has that tag)",
    )
    fig.update_xaxes(tickfont=_AXIS_TICK)
    fig.update_yaxes(tickfont=_AXIS_TICK)
    return fig


def plot_pos_hmm_matrices(hmm: PosTaggingHMM) -> go.Figure:
    """Combined transition + emission (legacy); prefer split plots in the lab."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Transition — P(next tag | previous tag)",
            "Emission — P(word | tag at this word)",
        ),
        horizontal_spacing=0.18,
    )
    fig.add_trace(
        go.Heatmap(
            z=hmm.trans,
            x=[f"→ {t}" for t in hmm.tags],
            y=[f"{t} →" for t in hmm.tags],
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            text=np.round(hmm.trans, 2),
            texttemplate="%{text}",
            textfont=_HEATMAP_TEXT,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=hmm.emit,
            x=[f"{w} ({i + 1})" for i, w in enumerate(hmm.sentence)],
            y=hmm.tags,
            colorscale="Greens",
            zmin=0.0,
            zmax=1.0,
            text=np.round(hmm.emit, 2),
            texttemplate="%{text}",
            textfont=_HEATMAP_TEXT,
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Probability tables — *the fans watch the race*",
        template="plotly_white",
        height=520,
        margin={"t": 100, "b": 60},
    )
    return fig


def plot_pos_viterbi_trellis(
    hmm: PosTaggingHMM,
    path: list[str] | None = None,
) -> go.Figure:
    computed_path, trellis = hmm.viterbi()
    path = path or computed_path

    tags = hmm.tags
    sentence = hmm.sentence
    n = len(sentence)
    fig = go.Figure()
    for t_idx, tag in enumerate(tags):
        y = trellis[:, t_idx]
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=y,
                mode="lines+markers",
                name=f"{tag} (part of speech)",
                line={"color": _tag_color(tag), "width": 2.5},
                marker={"size": 10, "color": _tag_color(tag)},
            )
        )
    _add_trellis_winning_path(fig, path, trellis, tags)
    _annotate_trellis_path(fig, sentence, path, trellis, tags)

    ticktext = [f"{w}<br>(word {i + 1})" for i, w in enumerate(sentence)]
    fig.update_layout(
        title="Viterbi trellis — best tag path",
        template="plotly_white",
        height=480,
        margin={"t": 90},
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(n)),
            "ticktext": ticktext,
            "title": "word (you can read the sentence)",
        },
        yaxis_title="log score (higher = more likely)",
        legend={
            "orientation": "h",
            "y": 1.08,
            "title": {"text": "Thick line + large dot = winning tag path"},
        },
    )
    return fig


def plot_crf_forward_columns(model: TripletCRF, sentence: list[str]) -> go.Figure:
    """Visualize forward DP columns (denominator)."""
    _, alpha = model._forward_log(sentence)
    rows = []
    for step in range(len(sentence)):
        for t_idx, tag in enumerate(model.tags):
            rows.append(
                {
                    "step": step,
                    "word": sentence[step],
                    "tag": tag,
                    "log alpha": float(alpha[step, t_idx]),
                }
            )
    frame = pd.DataFrame(rows)
    labels = [f"{tag} @ '{word}'" for tag, word in zip(frame["tag"], frame["word"])]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=frame["log alpha"],
            marker_color=[_tag_color(t) for t in frame["tag"]],
            text=[f"{v:.2f}" for v in frame["log alpha"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Forward DP columns (log α) — adds up all tag paths for Z(x)",
        template="plotly_white",
        height=400,
        xaxis_title="tag at each word position",
        yaxis_title="log α (partial sum of path scores)",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.18,
        text="Each bar is one (tag, word) cell in the forward table",
        showarrow=False,
        font={"size": 10, "color": "#666666"},
        xanchor="center",
    )
    return fig


_SHIRT_COLORS = {
    "green": "#5A8F6F",
    "blue": "#2F6F8F",
    "red": "#C45C26",
}


def _rounded_rect(
    fig: go.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    fill: str,
    line_color: str,
    dash: str | None = None,
) -> None:
    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fillcolor=fill,
        line={"color": line_color, "width": 2, "dash": dash or "solid"},
        layer="below",
    )


def plot_hmm_problem_intro(
    obs_seq: list[str],
    *,
    title: str = "Guess hidden mood from shirt colors you can see",
) -> go.Figure:
    """Layered day view plus Markov assumptions: mood←yesterday, shirt←today only."""
    fig = go.Figure()
    n = len(obs_seq)

    for i, color in enumerate(obs_seq):
        x = float(i)
        _rounded_rect(
            fig,
            x - 0.38,
            1.55,
            x + 0.38,
            2.05,
            fill="#f4f0fa",
            line_color=_PALETTE["Sad"],
        )
        fig.add_annotation(
            x=x,
            y=1.8,
            text="<b>?</b><br><span style='font-size:10px'>mood</span>",
            showarrow=False,
            font={"size": 16, "color": _PALETTE["Sad"]},
            yanchor="middle",
        )
        shirt_fill = _SHIRT_COLORS.get(color, "#dddddd")
        _rounded_rect(
            fig,
            x - 0.42,
            0.35,
            x + 0.42,
            0.95,
            fill=shirt_fill,
            line_color=shirt_fill,
        )
        fig.add_annotation(
            x=x,
            y=0.65,
            text=f"<b>{color}</b><br><span style='font-size:10px'>shirt</span>",
            showarrow=False,
            font={"size": 13, "color": "white"},
            yanchor="middle",
        )
        fig.add_annotation(
            x=x,
            y=-0.02,
            text=f"Day {i + 1}",
            showarrow=False,
            font={"size": 11, "color": "#666666"},
        )
        _add_arrow(
            fig,
            x,
            1.48,
            x,
            1.08,
            "emission" if i == 0 else "",
            color="#5A8F6F",
            label_offset=(0.62, 0.0) if i == 0 else (0.0, 0.0),
        )
        if i < n - 1:
            _add_arrow(
                fig,
                x + 0.42,
                1.8,
                x + 0.58,
                1.8,
                "transition" if i == 0 else "",
                color="#2F6F8F",
                label_offset=(0.0, 0.24) if i == 0 else (0.0, 0.0),
            )

    fig.add_annotation(
        x=-0.95,
        y=1.8,
        text="Hidden<br><b>you never see</b>",
        showarrow=False,
        font={"size": 11, "color": _PALETTE["Sad"]},
        xanchor="right",
    )
    fig.add_annotation(
        x=-0.95,
        y=0.65,
        text="Visible<br><b>all you get</b>",
        showarrow=False,
        font={"size": 11, "color": "#444444"},
        xanchor="right",
    )
    fig.add_annotation(
        x=(n - 1) / 2,
        y=2.38,
        text=f"<b>Observed shirts:</b> {' → '.join(obs_seq)}",
        showarrow=False,
        font={"size": 12, "color": "#333333"},
        xanchor="center",
    )
    fig.add_annotation(
        x=(n - 1) / 2,
        y=-0.42,
        text=(
            "<b>Assumption 1:</b> today's mood depends on <b>yesterday's mood only</b> &nbsp;·&nbsp; "
            "<b>Assumption 2:</b> today's shirt depends on <b>today's mood only</b>"
        ),
        showarrow=False,
        font={"size": 11, "color": "#333333"},
        xanchor="center",
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=480,
        xaxis={"visible": False, "range": [-1.2, max(n - 1, 1) + 0.8]},
        yaxis={"visible": False, "range": [-0.55, 2.55]},
        margin={"l": 70, "r": 20, "t": 70, "b": 55},
        showlegend=False,
    )
    return fig


def plot_pos_problem_intro(
    sentence: list[str],
    *,
    n_tags: int = 3,
    title: str = "Tag each word — tags are hidden, words are visible",
) -> go.Figure:
    """Hidden POS tags above sentence; inset bar chart for brute force vs Viterbi work."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.08,
        subplot_titles=(
            "Sentence you read (bottom) · tags you infer (top)",
            "Work per sentence length",
        ),
    )

    n = len(sentence)
    for i, word in enumerate(sentence):
        x = float(i)
        _rounded_rect(
            fig,
            x - 0.38,
            1.35,
            x + 0.38,
            1.85,
            fill="#eef4f8",
            line_color=_PALETTE["DET"],
            dash="dash",
        )
        fig.add_annotation(
            x=x,
            y=1.6,
            xref="x",
            yref="y",
            text="<b>?</b>",
            showarrow=False,
            font={"size": 15, "color": _PALETTE["DET"]},
        )
        _rounded_rect(
            fig,
            x - 0.42,
            0.35,
            x + 0.42,
            0.95,
            fill="#ffffff",
            line_color="#cccccc",
        )
        fig.add_annotation(
            x=x,
            y=0.65,
            xref="x",
            yref="y",
            text=f"<b>{word}</b>",
            showarrow=False,
            font={"size": 13, "color": "#222222"},
        )

    fig.add_annotation(
        x=(n - 1) / 2,
        y=2.05,
        xref="x",
        yref="y",
        text="Best tag path = one tag per word",
        showarrow=False,
        font={"size": 11, "color": "#555555"},
        xanchor="center",
    )

    length = n
    brute = float(n_tags**length)
    viterbi = float(length * n_tags * n_tags)
    methods = ["Brute force<br>all paths", "Viterbi<br>one best per tag"]
    counts = [brute, viterbi]
    colors = ["#C45C26", "#5A8F6F"]
    fig.add_trace(
        go.Bar(
            x=methods,
            y=counts,
            marker_color=colors,
            text=[f"{int(c):,}" for c in counts],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(type="log", title_text="paths to score (log scale)", row=1, col=2)

    fig.update_xaxes(visible=False, range=[-0.7, max(n - 1, 1) + 0.7], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0.1, 2.3], row=1, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 80, "b": 40},
        showlegend=False,
    )
    return fig


def _panel_node(
    fig: go.Figure,
    x: float,
    y: float,
    label: str,
    *,
    fill: str,
    xref: str,
    yref: str,
    text_color: str = "white",
) -> None:
    fig.add_annotation(
        x=x,
        y=y,
        text=f"<b>{label}</b>",
        xref=xref,
        yref=yref,
        showarrow=False,
        font={"size": 12, "color": text_color},
        bgcolor=fill,
        bordercolor=fill,
        borderwidth=2,
        borderpad=10,
        yanchor="middle",
        xanchor="center",
    )


def plot_hmm_vs_crf_contrast(
    *,
    title: str = "HMM generates · CRF classifies",
) -> go.Figure:
    """Side-by-side generative HMM vs discriminative CRF story."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("HMM — pick mood, then shirt", "CRF — see sentence, pick tags"),
        horizontal_spacing=0.12,
    )

    _panel_node(fig, 0.5, 2.2, "mood", fill=_PALETTE["Happy"], xref="x", yref="y")
    _panel_node(
        fig,
        0.5,
        1.0,
        "shirt",
        fill=_PALETTE["observation"],
        xref="x",
        yref="y",
        text_color="#222222",
    )
    fig.add_annotation(
        x=0.5,
        y=1.25,
        ax=0.5,
        ay=2.0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#666666",
        text="emission",
        font={"size": 10, "color": "#666666"},
    )
    fig.add_annotation(
        x=0.5,
        y=2.55,
        xref="x",
        yref="y",
        text="Factory recipe",
        showarrow=False,
        font={"size": 11, "color": "#666666"},
    )

    words = ["the", "dog", "barks"]
    tags = ["DET", "NOUN", "VERB"]
    for i, w in enumerate(words):
        xi = float(i)
        fig.add_shape(
            type="rect",
            x0=xi - 0.32,
            y0=0.25,
            x1=xi + 0.32,
            y1=0.75,
            xref="x2",
            yref="y2",
            fillcolor="#ffffff",
            line={"color": "#cccccc", "width": 2},
            layer="below",
        )
        fig.add_annotation(
            x=xi,
            y=0.5,
            xref="x2",
            yref="y2",
            text=f"<b>{w}</b>",
            showarrow=False,
            font={"size": 11, "color": "#222222"},
        )
        fig.add_annotation(
            x=xi,
            y=1.35,
            xref="x2",
            yref="y2",
            text=f"<b>{tags[i]}</b>",
            showarrow=False,
            font={"size": 11, "color": "white"},
            bgcolor=_tag_color(tags[i]),
            bordercolor=_tag_color(tags[i]),
            borderwidth=2,
            borderpad=6,
        )
        fig.add_annotation(
            x=xi,
            y=1.05,
            ax=xi,
            ay=0.82,
            xref="x2",
            yref="y2",
            axref="x2",
            ayref="y2",
            showarrow=True,
            arrowhead=0,
            arrowcolor="#aaaaaa",
            text="",
        )
    fig.add_annotation(
        x=1.0,
        y=1.75,
        xref="x2",
        yref="y2",
        text="Detective reads words → picks tags",
        showarrow=False,
        font={"size": 11, "color": "#666666"},
        xanchor="center",
    )

    fig.update_xaxes(visible=False, range=[-0.2, 1.2], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0.6, 2.8], row=1, col=1)
    fig.update_xaxes(visible=False, range=[-0.5, 2.5], row=1, col=2)
    fig.update_yaxes(visible=False, range=[0.0, 2.0], row=1, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        margin={"t": 90, "b": 30},
        showlegend=False,
    )
    return fig


def plot_crf_code_pipeline(*, title: str = "CRF code in five steps") -> go.Figure:
    """Horizontal pipeline: path score → Z(x) → probability → Viterbi."""
    steps = [
        ("1", "path score", "one tag path", _PALETTE["DET"]),
        ("2", "Z(x)", "all paths", _PALETTE["NOUN"]),
        ("3", "probability", "score ÷ Z", _PALETTE["VERB"]),
        ("4", "forward DP", "fast Z(x)", _PALETTE["Happy"]),
        ("5", "Viterbi", "best path", _PALETTE["Sad"]),
    ]
    fig = go.Figure()
    for i, (num, label, sub, fill) in enumerate(steps):
        x = float(i * 1.6)
        _add_node(fig, x, 0.5, f"{num}. {label}", fill=fill, size=11)
        fig.add_annotation(
            x=x,
            y=0.05,
            text=sub,
            showarrow=False,
            font={"size": 10, "color": "#666666"},
        )
        if i < len(steps) - 1:
            _add_arrow(fig, x + 0.55, 0.5, x + 1.05, 0.5, "", color="#aaaaaa")

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=220,
        xaxis={"visible": False, "range": [-0.5, (len(steps) - 1) * 1.6 + 0.5]},
        yaxis={"visible": False, "range": [-0.15, 0.85]},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        showlegend=False,
    )
    return fig


def plot_complexity_visual(*, title: str = "Why Viterbi / Forward DP is O(N · K²)") -> go.Figure:
    """Visual proof of O(N · K^2) dynamic programming vs O(K^N) brute force."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.48, 0.52],
        subplot_titles=(
            "K × K Local Transitions at Step t",
            "Brute Force (K^N) vs Dynamic Programming (N · K²)",
        ),
        horizontal_spacing=0.10,
    )

    tags = ["DET", "NOUN", "VERB"]
    k = len(tags)
    y_pos = [2.2, 1.2, 0.2]

    # Draw nodes for step t-1 and step t
    for i, tag in enumerate(tags):
        y = y_pos[i]
        # Step t-1 node
        fig.add_annotation(
            x=0.0,
            y=y,
            text=f"<b>{tag}</b> (t-1)",
            showarrow=False,
            font={"size": 11, "color": "white"},
            bgcolor=_tag_color(tag),
            bordercolor=_tag_color(tag),
            borderwidth=2,
            borderpad=6,
            xref="x1",
            yref="y1",
        )
        # Step t node
        fig.add_annotation(
            x=2.0,
            y=y,
            text=f"<b>{tag}</b> (t)",
            showarrow=False,
            font={"size": 11, "color": "white"},
            bgcolor=_tag_color(tag),
            bordercolor=_tag_color(tag),
            borderwidth=2,
            borderpad=6,
            xref="x1",
            yref="y1",
        )

    # Draw all K x K = 9 directed arrows connecting step t-1 to step t
    for i, tag_prev in enumerate(tags):
        y_prev = y_pos[i]
        for j, tag_curr in enumerate(tags):
            y_curr = y_pos[j]
            fig.add_annotation(
                x=1.70,
                y=y_curr,
                ax=0.30,
                ay=y_prev,
                xref="x1",
                yref="y1",
                axref="x1",
                ayref="y1",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5 if i == j else 1.0,
                arrowcolor=_tag_color(tag_curr) if i == j else "#bbbbbb",
                text="",
            )

    fig.add_annotation(
        x=1.0,
        y=2.7,
        text="<b>K states × K predecessors = K² operations per token</b>",
        showarrow=False,
        font={"size": 11, "color": "#2F6F8F"},
        xref="x1",
        yref="y1",
    )
    fig.add_annotation(
        x=1.0,
        y=-0.25,
        text="Across <b>N</b> tokens: Total Operations = <b>N · K²</b>",
        showarrow=False,
        font={"size": 11, "color": "#111111"},
        xref="x1",
        yref="y1",
    )

    # Right subplot: Bar comparison on logarithmic scale
    configs = [
        ("N=3, K=2<br>(Mood HMM)", 2**3, 3 * (2**2)),
        ("N=5, K=3<br>(Toy POS)", 3**5, 5 * (3**2)),
        ("N=10, K=10<br>(Small NLP)", 10**10, 10 * (10**2)),
        ("N=20, K=45<br>(Real POS)", 45**20, 20 * (45**2)),
    ]

    labels = [c[0] for c in configs]
    brute_ops = [c[1] for c in configs]
    dp_ops = [c[2] for c in configs]

    # Use log10 for visual bar scale
    log_brute = [np.log10(float(v)) for v in brute_ops]
    log_dp = [np.log10(float(v)) for v in dp_ops]

    fig.add_trace(
        go.Bar(
            name="Brute Force (K^N)",
            x=labels,
            y=log_brute,
            marker_color="#C45C26",
            text=[f"{v:.1e}" if v > 1e4 else str(v) for v in brute_ops],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Viterbi / DP (N · K²)",
            x=labels,
            y=log_dp,
            marker_color="#5A8F6F",
            text=[f"{v:,}" for v in dp_ops],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(visible=False, range=[-0.4, 2.4], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.5, 3.0], row=1, col=1)
    fig.update_xaxes(title_text="Sequence Length (N) & State Count (K)", row=1, col=2)
    fig.update_yaxes(
        title_text="Operations (log₁₀ scale)",
        row=1,
        col=2,
        range=[0, 36],
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin={"t": 80, "b": 40},
        legend={"orientation": "h", "y": 1.12, "x": 0.52},
    )
    return fig


def plot_crf_training_flow(*, title: str = "CRF Supervised Training Loop") -> go.Figure:
    """Visual flowchart showing the 5-step supervised CRF training cycle."""
    fig = go.Figure()

    boxes = [
        (0.0, 2.0, "1. Labeled Pair (x, y*)", "Empirical gold tags", "#2F6F8F"),
        (2.4, 2.0, "2. Forward Pass", "Computes Z(x) via DP", "#5A8F6F"),
        (4.8, 2.0, "3. Model Expectations", "E[f] over all paths", "#8F6FA8"),
        (4.8, 0.4, "4. Loss & Gradient", "∇L = E[f] − f*(gold)", "#C45C26"),
        (1.2, 0.4, "5. Weight Update", "w ← w − η · ∇L", "#2F6F8F"),
    ]

    for x, y, header, sub, fill in boxes:
        _add_node(fig, x, y, header, fill=fill, size=11)
        fig.add_annotation(
            x=x,
            y=y - 0.28,
            text=f"<i>{sub}</i>",
            showarrow=False,
            font={"size": 10, "color": "#555555"},
        )

    # Arrows between boxes
    arrows = [
        ((0.0, 1.7), (0.0, 0.7), "Extract f*"),
        ((0.0, 0.4), (0.6, 0.4), "Feed f*"),
        ((0.7, 2.0), (1.7, 2.0), "Input sentence x"),
        ((3.1, 2.0), (4.0, 2.0), "Z(x) & probabilities"),
        ((4.8, 1.6), (4.8, 0.8), "Compute gradient"),
        ((4.0, 0.4), (2.0, 0.4), "Update weights w"),
    ]

    for (x0, y0), (x1, y1), lbl in arrows:
        _add_arrow(fig, x0, y0, x1, y1, lbl, color="#444444")

    # Feedback arrow from step 5 back to step 2 (next iteration)
    fig.add_annotation(
        x=2.4,
        y=1.65,
        ax=1.2,
        ay=0.75,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#5A8F6F",
        text="",
    )
    fig.add_annotation(
        x=1.8,
        y=1.2,
        text="New weights w",
        showarrow=False,
        font={"size": 9, "color": "#5A8F6F"},
        bgcolor="#ffffff",
        bordercolor="#5A8F6F",
        borderwidth=1,
        borderpad=2,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=320,
        xaxis={"visible": False, "range": [-0.8, 5.6]},
        yaxis={"visible": False, "range": [-0.1, 2.5]},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        showlegend=False,
    )
    return fig


def plot_crf_forward_trellis(model: TripletCRF, sentence: list[str]) -> go.Figure:
    """Trellis showing Forward DP accumulation (summing incoming path masses for Z(x))."""
    log_z, alpha = model._forward_log(sentence)
    tags = model.tags
    n = len(sentence)

    fig = go.Figure()

    # Plot alpha trajectories
    for t_idx, tag in enumerate(tags):
        y_vals = [float(alpha[step, t_idx]) for step in range(n)]
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=y_vals,
                mode="lines+markers+text",
                name=f"{tag} (log α)",
                line={"color": _tag_color(tag), "width": 2.5},
                marker={"size": 12, "color": _tag_color(tag)},
                text=[f"α={np.exp(v):.1f}<br>(log {v:.2f})" for v in y_vals],
                textposition="top center",
            )
        )

    # Add final partition Z(x) point on right
    z_val = float(np.exp(log_z))
    fig.add_trace(
        go.Scatter(
            x=[n - 0.2],
            y=[log_z],
            mode="markers+text",
            name="Z(x) = ∑ α_final",
            marker={"size": 16, "color": "#C45C26", "symbol": "diamond"},
            text=[f"<b>Z(x) = {z_val:.1f}</b><br>(log Z={log_z:.2f})"],
            textposition="top right",
        )
    )

    ticktext = [f"<b>{w}</b><br>(Word {i+1})" for i, w in enumerate(sentence)]
    fig.update_layout(
        title="Forward DP Trellis — Computing Partition Function Z(x)",
        template="plotly_white",
        height=420,
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(n)),
            "ticktext": ticktext,
            "title": "Word in sentence",
            "range": [-0.3, n + 0.3],
        },
        yaxis={"title": "log α (accumulated mass entering tag)"},
        legend={"orientation": "h", "y": 1.15, "x": 0.05},
        margin={"t": 90, "b": 40},
    )
    return fig


def plot_crf_viterbi_trellis(model: TripletCRF, sentence: list[str]) -> go.Figure:
    """Trellis showing CRF Viterbi decoding with highlighted winning path."""
    path, scores = model.viterbi(sentence)
    tags = model.tags
    n = len(sentence)

    fig = go.Figure()

    # Plot lines for each tag
    for t_idx, tag in enumerate(tags):
        y_vals = [float(scores[step, t_idx]) for step in range(n)]
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=y_vals,
                mode="lines+markers",
                name=f"{tag}",
                line={"color": _tag_color(tag), "width": 1.5, "dash": "dot"},
                marker={"size": 8, "color": _tag_color(tag)},
            )
        )

    # Highlight winning path in bold
    win_x = list(range(n))
    win_y = [float(scores[i, model.tag2idx[path[i]]]) for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=win_x,
            y=win_y,
            mode="lines+markers+text",
            name=f"Winning Path: {' → '.join(path)}",
            line={"color": "#111111", "width": 4},
            marker={"size": 14, "color": [_tag_color(t) for t in path]},
            text=[f"<b>{path[i]}</b><br>score={win_y[i]:.2f}" for i in range(n)],
            textposition="top center",
        )
    )

    ticktext = [f"<b>{w}</b><br>(Word {i+1})" for i, w in enumerate(sentence)]
    fig.update_layout(
        title=f"CRF Viterbi Decode — Winning Path: {' → '.join(path)}",
        template="plotly_white",
        height=420,
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(n)),
            "ticktext": ticktext,
            "title": "Word in sentence",
            "range": [-0.3, n - 0.7],
        },
        yaxis={"title": "Viterbi log score (max over incoming paths)"},
        legend={"orientation": "h", "y": 1.15, "x": 0.05},
        margin={"t": 90, "b": 40},
    )
    return fig


def plot_forward_vs_viterbi_compare() -> go.Figure:
    """Side-by-side comparison of Forward DP (Sum) vs Viterbi DP (Max)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Forward DP: Sums ALL Paths (Denominator Z)",
            "Viterbi DP: Keeps BEST Path (Prediction y*)",
        ),
        horizontal_spacing=0.10,
    )

    # Panel 1: Forward (Sum)
    _panel_node(fig, 0.5, 2.0, "All Paths", fill="#2F6F8F", xref="x1", yref="y1")
    _panel_node(fig, 0.5, 1.0, "Operator: ∑ (sum)", fill="#5A8F6F", xref="x1", yref="y1")
    _panel_node(fig, 0.5, 0.0, "Output: Z(x) (Mass)", fill="#C45C26", xref="x1", yref="y1")

    fig.add_annotation(
        x=0.5, y=1.25, ax=0.5, ay=1.8,
        xref="x1", yref="y1", axref="x1", ayref="y1",
        showarrow=True, arrowhead=2, arrowcolor="#444444", text="",
    )
    fig.add_annotation(
        x=0.5, y=0.25, ax=0.5, ay=0.8,
        xref="x1", yref="y1", axref="x1", ayref="y1",
        showarrow=True, arrowhead=2, arrowcolor="#444444", text="",
    )

    # Panel 2: Viterbi (Max)
    _panel_node(fig, 0.5, 2.0, "All Paths", fill="#2F6F8F", xref="x2", yref="y2")
    _panel_node(fig, 0.5, 1.0, "Operator: max + argmax", fill="#8F6FA8", xref="x2", yref="y2")
    _panel_node(fig, 0.5, 0.0, "Output: Best Path y*", fill="#111111", xref="x2", yref="y2")

    fig.add_annotation(
        x=0.5, y=1.25, ax=0.5, ay=1.8,
        xref="x2", yref="y2", axref="x2", ayref="y2",
        showarrow=True, arrowhead=2, arrowcolor="#444444", text="",
    )
    fig.add_annotation(
        x=0.5, y=0.25, ax=0.5, ay=0.8,
        xref="x2", yref="y2", axref="x2", ayref="y2",
        showarrow=True, arrowhead=2, arrowcolor="#444444", text="",
    )

    fig.update_xaxes(visible=False, range=[-0.2, 1.2], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.4, 2.4], row=1, col=1)
    fig.update_xaxes(visible=False, range=[-0.2, 1.2], row=1, col=2)
    fig.update_yaxes(visible=False, range=[-0.4, 2.4], row=1, col=2)

    fig.update_layout(
        title="Same O(N · K²) Dynamic Programming — Different Operators",
        template="plotly_white",
        height=320,
        margin={"t": 70, "b": 20},
        showlegend=False,
    )
    return fig

