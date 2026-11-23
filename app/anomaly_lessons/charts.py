"""Vega/Altair chart builders for the Financial Anomaly Detection lab."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from st_compat import altair_chart


def _style(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    """Apply crisp spacing, legible typography, and clean axis/title margins."""
    return (
        chart.configure_title(fontSize=14, subtitleFontSize=12, subtitleColor="#52606d", anchor="start", offset=10)
        .configure_axis(labelFontSize=11, titleFontSize=12, titlePadding=8, labelPadding=5)
        .configure_legend(labelFontSize=11, titleFontSize=11, symbolSize=90, padding=6)
        .configure_view(strokeWidth=0)
    )


def amount_threshold_chart(
    df: pd.DataFrame,
    *,
    lower: float | None,
    upper: float,
    title: str,
    flag_col: str = "is_anomaly",
) -> None:
    """Bar chart of transaction amounts with anomaly threshold rule lines."""
    plot_df = df.copy()
    plot_df["Flag"] = plot_df[flag_col].map({True: "Anomaly", False: "Normal"})

    max_len = int(plot_df["tx_id"].astype(str).str.len().max()) if not plot_df.empty else 3
    label_angle = -25 if max_len > 5 else 0
    y_max = max(float(plot_df["amount"].max()), float(upper)) * 1.12

    bars = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("tx_id:N", title="Transaction", sort=None, axis=alt.Axis(labelAngle=label_angle)),
            y=alt.Y("amount:Q", title="Amount ($)", scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color(
                "Flag:N",
                title=None,
                scale=alt.Scale(domain=["Normal", "Anomaly"], range=["#1f6f8b", "#d9483b"]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["tx_id:N", "amount:Q", "Flag:N"],
        )
    )
    layers = [bars]
    upper_rule = (
        alt.Chart(pd.DataFrame({"y": [upper]}))
        .mark_rule(color="#d9483b", strokeDash=[6, 4], strokeWidth=2)
        .encode(y="y:Q")
    )
    layers.append(upper_rule)
    subtitle_parts = [f"Red dashed line = upper cutoff (${upper:,.1f})"]
    if lower is not None and lower > 0:
        lower_rule = (
            alt.Chart(pd.DataFrame({"y": [lower]}))
            .mark_rule(color="#e08e26", strokeDash=[4, 4], strokeWidth=1.5)
            .encode(y="y:Q")
        )
        layers.append(lower_rule)
        subtitle_parts.append(f"Amber line = lower cutoff (${lower:,.1f})")

    chart = alt.layer(*layers).properties(
        title=alt.TitleParams(title, subtitle=" · ".join(subtitle_parts)),
        height=320,
    )
    altair_chart(_style(chart))


def mahalanobis_2d_chart(points_df: pd.DataFrame, contours_df: pd.DataFrame) -> None:
    """2D scatter with Euclidean circle vs Mahalanobis ellipse contours."""
    all_x = pd.concat([points_df["amount"], contours_df["amount"]], ignore_index=True)
    all_y = pd.concat([points_df["distance_km"], contours_df["distance_km"]], ignore_index=True)
    x_pad = (float(all_x.max()) - float(all_x.min())) * 0.08
    y_pad = (float(all_y.max()) - float(all_y.min())) * 0.10
    x_dom = [float(all_x.min()) - x_pad, float(all_x.max()) + x_pad]
    y_dom = [float(all_y.min()) - y_pad, float(all_y.max()) + y_pad]

    pts = (
        alt.Chart(points_df)
        .mark_circle(opacity=0.9, stroke="#ffffff", strokeWidth=0.6)
        .encode(
            x=alt.X("amount:Q", title="Transaction amount ($)", scale=alt.Scale(domain=x_dom)),
            y=alt.Y("distance_km:Q", title="Distance from home (km)", scale=alt.Scale(domain=y_dom)),
            color=alt.Color(
                "point_role:N",
                title="Point",
                scale=alt.Scale(
                    domain=["Normal history", "Along-trend (Legit)", "Off-trend (Fraud)"],
                    range=["#4a7c9b", "#2a9d6f", "#d9483b"],
                ),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            size=alt.condition(
                alt.datum.point_role == "Normal history",
                alt.value(55),
                alt.value(175),
            ),
            tooltip=["tx_id:N", "amount:Q", "distance_km:Q", "euclidean_z_dist:Q", "mahalanobis_dist:Q"],
        )
    )
    lines = (
        alt.Chart(contours_df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x="amount:Q",
            y="distance_km:Q",
            order="order:Q",
            color=alt.Color(
                "boundary:N",
                title="Boundary",
                scale=alt.Scale(
                    domain=["Mahalanobis ellipse", "Euclidean (ignores correlation)"],
                    range=["#1b4965", "#e08e26"],
                ),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
            ),
            strokeDash=alt.StrokeDash(
                "boundary:N",
                scale=alt.Scale(
                    domain=["Mahalanobis ellipse", "Euclidean (ignores correlation)"],
                    range=[[1, 0], [6, 4]],
                ),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
            ),
        )
    )

    along_df = points_df[points_df["point_role"] == "Along-trend (Legit)"]
    off_df = points_df[points_df["point_role"] == "Off-trend (Fraud)"]
    along_lbl = (
        alt.Chart(along_df)
        .mark_text(align="right", dx=-12, dy=-12, fontSize=11, fontWeight="bold", color="#1b7a56")
        .encode(x="amount:Q", y="distance_km:Q", text=alt.value("Legit (along trend)"))
    )
    off_lbl = (
        alt.Chart(off_df)
        .mark_text(align="right", dx=-12, dy=16, fontSize=11, fontWeight="bold", color="#b83227")
        .encode(x="amount:Q", y="distance_km:Q", text=alt.value("Fraud (off trend)"))
    )

    chart = (
        alt.layer(lines, pts, along_lbl, off_lbl)
        .resolve_scale(color="independent")
        .properties(title="Euclidean circle vs Mahalanobis ellipse", height=390)
    )
    altair_chart(_style(chart))


def isolation_1d_cuts_chart(step: int) -> None:
    """Two-panel 1D visual: full scale ($0-$520) + zoomed normal crowd ($38-$63) with zero text overlap."""
    pts_df = pd.DataFrame(
        [
            {"tx": "$42", "amount": 42.0, "y": 0.0, "role": "Normal", "cuts": 3},
            {"tx": "$48", "amount": 48.0, "y": 0.0, "role": "Normal", "cuts": 3},
            {"tx": "$51", "amount": 51.0, "y": 0.0, "role": "Normal", "cuts": 3},
            {"tx": "$55", "amount": 55.0, "y": 0.0, "role": "Normal", "cuts": 4},
            {"tx": "$59", "amount": 59.0, "y": 0.0, "role": "Normal", "cuts": 4},
            {"tx": "$490", "amount": 490.0, "y": 0.0, "role": "Anomaly", "cuts": 1},
        ]
    )

    # Panel 1: Full scale shows WHY Cut 1 isolates $490 immediately (huge empty gap)
    base_full = (
        alt.Chart(pd.DataFrame({"x1": [0.0], "x2": [525.0], "y": [0.0]}))
        .mark_rule(color="#cbd5e1", strokeWidth=2)
        .encode(
            x=alt.X("x1:Q", title="Full scale — Transaction amount ($)", scale=alt.Scale(domain=[0, 535])),
            x2="x2:Q",
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.05, 1.3])),
        )
    )
    pts_full = (
        alt.Chart(pts_df)
        .mark_circle(size=135, stroke="#1d1d1d", strokeWidth=0.9)
        .encode(
            x="amount:Q",
            y="y:Q",
            color=alt.Color(
                "role:N",
                title=None,
                scale=alt.Scale(domain=["Normal", "Anomaly"], range=["#1f6f8b", "#d9483b"]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["tx:N", "role:N", "cuts:Q"],
        )
    )
    # Only two non-overlapping group labels positioned inside the plot area (above x-axis)
    cluster_lbl = (
        alt.Chart(pd.DataFrame([{"x": 16.0, "y": -0.58, "text": "Normal crowd ($42–$59)"}]))
        .mark_text(align="left", fontSize=11, fontWeight="bold", color="#1f6f8b")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    fraud_lbl = (
        alt.Chart(pd.DataFrame([{"x": 490.0, "y": -0.58, "text": "$490 (Fraud)"}]))
        .mark_text(align="center", fontSize=11, fontWeight="bold", color="#d9483b")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    cut1_rule = (
        alt.Chart(pd.DataFrame([{"x": 210.0, "label": "Cut 1 ($210) -> $490 isolated!"}]))
        .mark_rule(color="#e08e26", strokeDash=[6, 4], strokeWidth=2.2)
        .encode(x="x:Q", tooltip=["label:N"])
    )
    cut1_lbl = (
        alt.Chart(pd.DataFrame([{"x": 218.0, "y": 0.78, "text": "Cut 1 ($210) → $490 isolated in 1 cut!"}]))
        .mark_text(align="left", fontSize=11, fontWeight="bold", color="#b45309")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    full_layers = [base_full, cut1_rule, cut1_lbl, pts_full, cluster_lbl, fraud_lbl]
    if step >= 2:
        inner_cuts = pd.DataFrame([{"step": s, "x": x} for s, x in [(2, 50.0), (3, 53.0), (4, 57.0)] if s <= step])
        full_layers.insert(
            1,
            alt.Chart(inner_cuts)
            .mark_rule(color="#e08e26", strokeDash=[3, 3], strokeWidth=1.2, opacity=0.75)
            .encode(x="x:Q"),
        )
    full_chart = alt.layer(*full_layers).properties(
        title=alt.TitleParams(
            f"Step {step} of 4 — Full scale ($0–$520)",
            subtitle="Huge empty gap around $490 means Cut 1 isolates fraud immediately",
        ),
        height=215,
    )
    altair_chart(_style(full_chart))

    # Panel 2: Zoomed view of the tight normal crowd ($38-$63) so every point & cut label has room
    normal_pts = pts_df[pts_df["role"] == "Normal"].copy()
    normal_pts["y_lbl"] = -0.58
    zoom_cuts = pd.DataFrame(
        [
            {"step": 2, "x": 50.0, "y_lbl": 0.86, "text": "Cut 2 ($50)", "align": "right", "dx": -6},
            {"step": 3, "x": 53.0, "y_lbl": 0.44, "text": "Cut 3 ($53)", "align": "center", "dx": 0},
            {"step": 4, "x": 57.0, "y_lbl": 0.86, "text": "Cut 4 ($57)", "align": "left", "dx": 6},
        ]
    )
    shown_zoom_cuts = zoom_cuts[zoom_cuts["step"] <= step]

    base_zoom = (
        alt.Chart(pd.DataFrame({"x1": [39.0], "x2": [62.0], "y": [0.0]}))
        .mark_rule(color="#cbd5e1", strokeWidth=2)
        .encode(
            x=alt.X("x1:Q", title="Zoomed into normal crowd ($38–$63)", scale=alt.Scale(domain=[38, 63])),
            x2="x2:Q",
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.05, 1.3])),
        )
    )
    pts_zoom = (
        alt.Chart(normal_pts)
        .mark_circle(size=165, color="#1f6f8b", stroke="#1d1d1d", strokeWidth=0.9)
        .encode(x="amount:Q", y="y:Q", tooltip=["tx:N", "cuts:Q"])
    )
    pt_lbls_zoom = (
        alt.Chart(normal_pts)
        .mark_text(fontSize=11, fontWeight="bold", color="#1f2937")
        .encode(x="amount:Q", y="y_lbl:Q", text="tx:N")
    )

    zoom_layers = [base_zoom, pts_zoom, pt_lbls_zoom]
    if not shown_zoom_cuts.empty:
        zoom_rules = (
            alt.Chart(shown_zoom_cuts)
            .mark_rule(color="#e08e26", strokeDash=[5, 4], strokeWidth=2)
            .encode(x="x:Q", tooltip=["text:N"])
        )
        zoom_layers.insert(1, zoom_rules)
        for _, row in shown_zoom_cuts.iterrows():
            zoom_layers.append(
                alt.Chart(pd.DataFrame([row]))
                .mark_text(
                    align=str(row["align"]),
                    dx=int(row["dx"]),
                    fontSize=11,
                    fontWeight="bold",
                    color="#b45309",
                )
                .encode(x="x:Q", y="y_lbl:Q", text="text:N")
            )

    zoom_subtitle = (
        "Slide to steps 2–4 to watch random cuts slowly separate tightly packed normal points"
        if step == 1
        else "Packed normal spend needs 3–4 cuts to isolate each transaction"
    )
    zoom_chart = alt.layer(*zoom_layers).properties(
        title=alt.TitleParams("Zoomed view — Normal crowd ($42 to $59)", subtitle=zoom_subtitle),
        height=205,
    )
    altair_chart(_style(zoom_chart))


def isolation_2d_chart(points_df: pd.DataFrame, splits_df: pd.DataFrame, *, show_splits: bool = True) -> None:
    """2D Isolation Forest scatter overlaid with Tree #1 random axis-aligned cuts."""
    pts = (
        alt.Chart(points_df)
        .mark_circle(size=85, opacity=0.9, stroke="#ffffff", strokeWidth=0.5)
        .encode(
            x=alt.X("amount:Q", title="Amount ($)", scale=alt.Scale(zero=False, padding=12)),
            y=alt.Y("tx_velocity_24h:Q", title="Transactions in 24h", scale=alt.Scale(zero=False, padding=12)),
            color=alt.Color(
                "predicted:N",
                title=None,
                scale=alt.Scale(domain=["Normal", "Flagged anomaly"], range=["#1f6f8b", "#d9483b"]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["tx_id:N", "amount:Q", "tx_velocity_24h:Q", "avg_path_length:Q", "anomaly_score:Q"],
        )
    )
    if show_splits and not splits_df.empty:
        cuts = (
            alt.Chart(splits_df)
            .mark_line(color="#8c9aa6", strokeWidth=1.1, opacity=0.65)
            .encode(
                x="x:Q",
                y="y:Q",
                detail="split_id:N",
                order="order:Q",
            )
        )
        chart = alt.layer(cuts, pts).properties(title="Random splits in Tree #1 + ensemble anomaly flags", height=360)
    else:
        chart = pts.properties(title="Isolation Forest ensemble anomaly flags", height=360)
    altair_chart(_style(chart))


def isolation_score_curve_chart(curve_df: pd.DataFrame, *, highlight_h: float, highlight_s: float) -> None:
    """Curve mapping average path length E[h(x)] to anomaly score s(x, psi)."""
    x_max = float(curve_df["avg_path_length"].max())
    line = (
        alt.Chart(curve_df)
        .mark_line(color="#1f6f8b", strokeWidth=2.5)
        .encode(
            x=alt.X("avg_path_length:Q", title="Average path length E[h(x)]"),
            y=alt.Y("anomaly_score:Q", title="Anomaly score s(x, psi)", scale=alt.Scale(domain=[0, 1.06])),
            tooltip=["avg_path_length:Q", "anomaly_score:Q"],
        )
    )
    ref_half = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(color="#888888", strokeDash=[5, 4])
        .encode(y="y:Q")
    )
    ref_lbl = (
        alt.Chart(pd.DataFrame({"x": [x_max], "y": [0.5], "text": ["s = 0.5 (typical normal)"]}))
        .mark_text(align="right", dy=-7, fontSize=11, color="#52606d")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    pt_df = pd.DataFrame(
        {
            "avg_path_length": [highlight_h],
            "anomaly_score": [highlight_s],
            "text": [f"E[h]={highlight_h:.2f} → s={highlight_s:.2f}"],
        }
    )
    pt = (
        alt.Chart(pt_df)
        .mark_circle(color="#d9483b", size=140, stroke="#ffffff", strokeWidth=1.2)
        .encode(x="avg_path_length:Q", y="anomaly_score:Q")
    )
    on_left = highlight_h < x_max * 0.55
    pt_lbl = (
        alt.Chart(pt_df)
        .mark_text(
            align="left" if on_left else "right",
            dx=10 if on_left else -10,
            dy=14 if highlight_s > 0.88 else -10,
            fontSize=11,
            fontWeight="bold",
            color="#b83227",
        )
        .encode(x="avg_path_length:Q", y="anomaly_score:Q", text="text:N")
    )
    chart = alt.layer(line, ref_half, ref_lbl, pt, pt_lbl).properties(
        title="How path length maps to s(x, psi)",
        height=290,
    )
    altair_chart(_style(chart))


def ocsvm_1d_spotlight_chart(curve_df: pd.DataFrame, probes_df: pd.DataFrame, *, rho: float) -> None:
    """1D first-principles RBF spotlight curve with carefully placed non-overlapping probe labels."""
    y_max = max(1.48, float(curve_df["warmth"].max()) + 0.30, rho + 0.20)
    area = (
        alt.Chart(curve_df)
        .mark_area(color="#1f6f8b", opacity=0.16, line={"color": "#1f6f8b", "strokeWidth": 2.4})
        .encode(
            x=alt.X("amount:Q", title="Transaction amount ($)", scale=alt.Scale(domain=[0, 275])),
            y=alt.Y("warmth:Q", title="Warmth", scale=alt.Scale(domain=[0, y_max])),
            tooltip=["amount:Q", "warmth:Q", "decision:Q", "zone:N"],
        )
    )
    sv_posts = (
        alt.Chart(pd.DataFrame({"amount": [50.0, 150.0], "tag": ["Fence post ($50)", "Fence post ($150)"]}))
        .mark_rule(color="#e08e26", strokeDash=[3, 3], strokeWidth=1.5)
        .encode(x="amount:Q", tooltip=["tag:N"])
    )
    cutoff_rule = (
        alt.Chart(pd.DataFrame({"rho": [rho]}))
        .mark_rule(color="#d9483b", strokeDash=[6, 4], strokeWidth=2.2)
        .encode(y="rho:Q")
    )
    cutoff_label = (
        alt.Chart(pd.DataFrame({"x": [272.0], "y": [rho], "text": [f"ρ = {rho:.2f}"]}))
        .mark_text(align="right", dy=-7, color="#d9483b", fontWeight="bold", fontSize=11)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    pts = (
        alt.Chart(probes_df)
        .mark_circle(size=145, stroke="#1d1d1d", strokeWidth=0.9)
        .encode(
            x="amount:Q",
            y="warmth:Q",
            color=alt.Color(
                "verdict:N",
                title=None,
                scale=alt.Scale(
                    domain=["Normal (Inside)", "Anomaly (Outside)"],
                    range=["#2a9d6f", "#d9483b"],
                ),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["label:N", "k_50:Q", "k_150:Q", "warmth:Q", "decision:Q", "verdict:N"],
        )
    )

    # Explicit per-probe placement so $42, $52, $100, $152, and $240 NEVER overlap
    gap_warmth = float(probes_df[probes_df["amount"] == 100.0]["warmth"].iloc[0]) if not probes_df.empty else 0.0
    gap_dy = 17 if (gap_warmth < rho and gap_warmth > 0.12) else -14
    placement: dict[float, tuple[str, str, int, int]] = {
        42.0: ("$42", "right", -10, 4),
        52.0: ("$52", "left", 10, -12),
        100.0: ("$100 (Gap)", "center", 0, gap_dy),
        152.0: ("$152", "left", 10, -12),
        240.0: ("$240 (Fraud)", "right", -10, -12),
    }
    label_layers: list[alt.Chart] = []
    for _, row in probes_df.iterrows():
        amt = float(row["amount"])
        short_lbl, align, dx, dy = placement.get(amt, (str(row["label"]), "center", 0, -14))
        color = "#1b7a56" if row["verdict"] == "Normal (Inside)" else "#b83227"
        label_layers.append(
            alt.Chart(pd.DataFrame([{"amount": amt, "warmth": float(row["warmth"]), "short": short_lbl}]))
            .mark_text(align=align, dx=dx, dy=dy, fontSize=11, fontWeight="bold", color=color)
            .encode(x="amount:Q", y="warmth:Q", text="short:N")
        )

    chart = alt.layer(area, sv_posts, cutoff_rule, cutoff_label, pts, *label_layers).properties(
        title=alt.TitleParams(
            "RBF spotlights around normal spend ($50 & $150)",
            subtitle=f"Above red cutoff (ρ = {rho:.2f}) = Normal · Below = Anomaly · Orange dotted lines = fence posts",
        ),
        height=300,
    )
    altair_chart(_style(chart))


def ocsvm_boundary_chart(grid_df: pd.DataFrame, points_df: pd.DataFrame) -> None:
    """2D One-Class SVM learned normal region + highlighted support vectors & anomalies."""
    bg_num = (
        alt.Chart(grid_df)
        .mark_square(size=120, opacity=0.35)
        .encode(
            x=alt.X("amount:Q", title="Amount ($)", scale=alt.Scale(zero=False)),
            y=alt.Y("tx_velocity_24h:Q", title="Transactions in 24h", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "region:N",
                title="Learned fence",
                scale=alt.Scale(
                    domain=["Normal region (f >= 0)", "Anomaly region (f < 0)"],
                    range=["#b7e4c7", "#f8c8b4"],
                ),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
            ),
        )
    )
    pts = (
        alt.Chart(points_df)
        .mark_circle(stroke="#1d1d1d", strokeWidth=0.8)
        .encode(
            x="amount:Q",
            y="tx_velocity_24h:Q",
            color=alt.Color(
                "plot_role:N",
                title="Transaction",
                scale=alt.Scale(
                    domain=["Normal (non-SV)", "Support vector", "Injected anomaly"],
                    range=["#1f6f8b", "#f4a261", "#d9483b"],
                ),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            size=alt.condition(
                alt.datum.plot_role == "Normal (non-SV)",
                alt.value(55),
                alt.value(135),
            ),
            tooltip=["tx_id:N", "plot_role:N", "amount:Q", "tx_velocity_24h:Q", "decision_score:Q", "flagged:N"],
        )
    )
    chart = (
        alt.layer(bg_num, pts)
        .resolve_scale(color="independent")
        .properties(
            title=alt.TitleParams(
                "One-Class SVM learned fence",
                subtitle="Green = inside normal fence · Orange dots = fence posts (support vectors)",
            ),
            height=390,
        )
    )
    altair_chart(_style(chart))


def autoencoder_toy_rebuild_chart(toy_df: pd.DataFrame, *, title: str) -> None:
    """Grouped bar chart comparing Real vs Rebuilt values on a 3-feature toy transaction."""
    chart = (
        alt.Chart(toy_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("feature:N", title="Feature (z-scaled)", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="Value", scale=alt.Scale(domain=[0, 3.3])),
            xOffset="series:N",
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=["Real (x)", "Rebuilt (x_hat)"], range=["#1f6f8b", "#e08e26"]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["feature:N", "series:N", "value:Q"],
        )
        .properties(title=title, height=240)
    )
    altair_chart(_style(chart))


def autoencoder_loss_chart(loss_df: pd.DataFrame) -> None:
    chart = (
        alt.Chart(loss_df)
        .mark_line(color="#1f6f8b", strokeWidth=2.5)
        .encode(
            x=alt.X("epoch:Q", title="Training epoch"),
            y=alt.Y("train_mse:Q", title="Training reconstruction MSE"),
            tooltip=["epoch:Q", "train_mse:Q"],
        )
        .properties(title="Autoencoder training loss curve", height=260)
    )
    altair_chart(_style(chart))


def autoencoder_error_dist_chart(eval_df: pd.DataFrame, threshold: float) -> None:
    """Histogram of reconstruction MSE separated by normal vs true anomaly."""
    hist = (
        alt.Chart(eval_df)
        .mark_bar(opacity=0.72)
        .encode(
            x=alt.X("recon_mse:Q", bin=alt.Bin(maxbins=30), title="Reconstruction error (MSE)"),
            y=alt.Y("count():Q", title="Transactions"),
            color=alt.Color(
                "status:N",
                title=None,
                scale=alt.Scale(domain=["Normal", "True anomaly"], range=["#1f6f8b", "#d9483b"]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="#d9483b", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="t:Q")
    )
    chart = alt.layer(hist, rule).properties(
        title=alt.TitleParams(
            "Reconstruction MSE distribution",
            subtitle=f"Red dashed line = anomaly cutoff ({threshold:.3f})",
        ),
        height=260,
    )
    altair_chart(_style(chart))


def autoencoder_feature_error_chart(feat_df: pd.DataFrame) -> None:
    plot_df = feat_df.copy()
    pretty = {
        "amount": "Amount",
        "tx_velocity_24h": "Velocity (24h)",
        "distance_km": "Distance (km)",
        "hour_of_day": "Hour of day",
        "merchant_risk": "Merchant risk",
    }
    plot_df["feature_label"] = plot_df["feature"].map(lambda s: pretty.get(str(s), str(s)))
    chart = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("feature_label:N", title="Feature", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean_sq_error:Q", title="Mean squared error"),
            xOffset="group:N",
            color=alt.Color(
                "group:N",
                title=None,
                scale=alt.Scale(domain=["Normal", "True anomaly"], range=["#1f6f8b", "#d9483b"]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["feature_label:N", "group:N", "mean_sq_error:Q"],
        )
        .properties(title="Per-feature reconstruction error (why anomalies triggered)", height=280)
    )
    altair_chart(_style(chart))


def lof_cluster_chart(points_df: pd.DataFrame) -> None:
    x_max = float(points_df["amount"].max()) + 30.0
    y_max = float(points_df["tx_velocity_24h"].max()) + 1.2
    pts = (
        alt.Chart(points_df)
        .mark_circle(opacity=0.88, stroke="#222222", strokeWidth=0.6)
        .encode(
            x=alt.X("amount:Q", title="Amount ($)", scale=alt.Scale(domain=[0, x_max])),
            y=alt.Y("tx_velocity_24h:Q", title="Transactions in 24h", scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color(
                "group:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "Tight cluster (Everyday)",
                        "Wide cluster (Affluent)",
                        "Local outlier",
                        "Global outlier",
                    ],
                    range=["#1f6f8b", "#64a6bd", "#e08e26", "#d9483b"],
                ),
                legend=alt.Legend(orient="top", columns=2),
            ),
            size=alt.Size(
                "lof_score:Q",
                scale=alt.Scale(range=[45, 340]),
                title="LOF score",
                legend=alt.Legend(orient="right"),
            ),
            tooltip=["tx_id:N", "group:N", "lof_score:Q", "lrd:Q", "global_z_dist:Q"],
        )
    )
    local_df = points_df[points_df["tx_id"] == "LOCAL_OUT"]
    global_df = points_df[points_df["tx_id"] == "GLOBAL_OUT"]
    local_lbl = (
        alt.Chart(local_df)
        .mark_text(align="left", dx=12, dy=-10, fontSize=11, fontWeight="bold", color="#b45309")
        .encode(x="amount:Q", y="tx_velocity_24h:Q", text="tx_id:N")
    )
    global_lbl = (
        alt.Chart(global_df)
        .mark_text(align="right", dx=-14, dy=4, fontSize=11, fontWeight="bold", color="#b83227")
        .encode(x="amount:Q", y="tx_velocity_24h:Q", text="tx_id:N")
    )
    chart = alt.layer(pts, local_lbl, global_lbl).properties(
        title="Dense vs sparse clusters: circle size = LOF score",
        height=370,
    )
    altair_chart(_style(chart))


def graph_network_chart(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Entity graph with custom per-node label offsets so no text overlaps circles, edges, or neighbors."""
    plot_nodes = nodes_df.copy()
    plot_nodes["node_label"] = plot_nodes.apply(
        lambda r: f"{r['id']} ({r['final_risk']:.2f})", axis=1
    )

    edges = (
        alt.Chart(edges_df)
        .mark_line(color="#9aa5b1", strokeWidth=1.8, opacity=0.78)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.4, 10.2])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0.2, 4.45])),
            detail="edge_id:N",
            order="order:Q",
        )
    )
    pts = (
        alt.Chart(plot_nodes)
        .mark_circle(stroke="#222222", strokeWidth=1.0)
        .encode(
            x="x:Q",
            y="y:Q",
            size=alt.Size(
                "final_risk:Q",
                scale=alt.Scale(domain=[0.0, 1.0], range=[160, 560]),
                title="Risk score",
                legend=alt.Legend(orient="bottom", direction="horizontal"),
            ),
            color=alt.Color(
                "entity:N",
                title=None,
                scale=alt.Scale(
                    domain=["Customer", "Device", "Merchant"],
                    range=["#1f6f8b", "#e08e26", "#d9483b"],
                ),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["id:N", "entity:N", "degree:Q", "hop_0_base_risk:Q", "final_risk:Q"],
        )
    )

    # Region header captions inside open top margin
    headers = (
        alt.Chart(
            pd.DataFrame(
                [
                    {"x": 2.2, "y": 4.22, "text": "Normal activity (1-to-1)"},
                    {"x": 7.0, "y": 4.22, "text": "Shared-device mule ring"},
                ]
            )
        )
        .mark_text(align="center", fontSize=11, fontWeight="bold", color="#616e7c")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    # Per-node label placement tailored to graph layout for zero overlap
    offsets: dict[str, tuple[str, int, int]] = {
        "Cust_A": ("center", 0, -22),
        "Dev_1": ("center", 0, 26),
        "Merch_Grocer": ("center", 0, -22),
        "Cust_B": ("right", -14, 4),
        "Dev_2": ("center", 0, 26),
        "Cust_X": ("center", 0, -22),
        "Cust_Y": ("right", -16, 4),
        "Cust_Z": ("center", 0, 26),
        "Dev_Shared": ("center", 0, 46),
        "Merch_Crypto": ("center", 0, -28),
    }
    label_layers: list[alt.Chart] = []
    for _, row in plot_nodes.iterrows():
        nid = str(row["id"])
        align, dx, dy = offsets.get(nid, ("center", 0, -20))
        label_layers.append(
            alt.Chart(pd.DataFrame([row]))
            .mark_text(align=align, dx=dx, dy=dy, fontSize=11, fontWeight="bold", color="#1f2937")
            .encode(x="x:Q", y="y:Q", text="node_label:N")
        )

    chart = alt.layer(edges, pts, headers, *label_layers).properties(
        title="Entity risk graph: normal (left) vs shared-device ring (right)",
        height=380,
    )
    altair_chart(_style(chart))


def temporal_sequence_chart(seq_df: pd.DataFrame) -> None:
    """Two full-width vertical charts: raw dollar amount (looks normal) vs sequence anomaly score."""
    burst_band = (
        alt.Chart(pd.DataFrame([{"x1": 21.5, "x2": 26.5}]))
        .mark_rect(color="#fee2e2", opacity=0.55)
        .encode(x="x1:Q", x2="x2:Q")
    )
    amt_line = (
        alt.Chart(seq_df)
        .mark_line(point=True, strokeWidth=2.0)
        .encode(
            x=alt.X("step:Q", title="Transaction sequence step", scale=alt.Scale(domain=[1, 32])),
            y=alt.Y("amount:Q", title="Amount ($)"),
            color=alt.Color(
                "status:N",
                title=None,
                scale=alt.Scale(domain=["Normal rhythm", "Anomalous sequence"], range=["#1f6f8b", "#d9483b"]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=["step:Q", "amount:Q", "hour_of_day:Q", "gap_minutes:Q", "point_amount_z:Q"],
        )
    )
    amt_chart = alt.layer(burst_band, amt_line).properties(
        title="1. Raw dollar amount looks ordinary during the 2 AM burst (shaded steps 22–26)",
        height=210,
    )
    altair_chart(_style(amt_chart))

    score_area = (
        alt.Chart(seq_df)
        .mark_area(color="#d9483b", opacity=0.32, line={"color": "#d9483b", "strokeWidth": 2.2})
        .encode(
            x=alt.X("step:Q", title="Transaction sequence step", scale=alt.Scale(domain=[1, 32])),
            y=alt.Y("sequence_score:Q", title="Sequence score"),
            tooltip=["step:Q", "sequence_score:Q", "hour_of_day:Q", "gap_minutes:Q"],
        )
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({"y": [1.05]}))
        .mark_rule(color="#b83227", strokeDash=[5, 4], strokeWidth=1.8)
        .encode(y="y:Q")
    )
    threshold_lbl = (
        alt.Chart(pd.DataFrame({"x": [2.0], "y": [1.05], "text": ["Alert threshold (1.05)"]}))
        .mark_text(align="left", dy=-7, fontSize=11, fontWeight="bold", color="#b83227")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    score_chart = alt.layer(burst_band, score_area, threshold_rule, threshold_lbl).properties(
        title="2. Recurrent sequence model catches the sudden 2 AM + 3-min gap burst",
        height=200,
    )
    altair_chart(_style(score_chart))


def evaluation_curves_chart(pr_df: pd.DataFrame, roc_df: pd.DataFrame) -> None:
    """Side-by-side responsive PR and ROC curves rendered in Streamlit columns for clean sizing."""
    col1, col2 = st.columns(2)
    with col1:
        pr = (
            alt.Chart(pr_df)
            .mark_line(strokeWidth=2.2)
            .encode(
                x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1.04])),
                color=alt.Color(
                    "Algorithm:N",
                    title=None,
                    legend=alt.Legend(orient="bottom", columns=2),
                ),
            )
            .properties(title="Precision-Recall curves (primary for rare fraud)", height=310)
        )
        altair_chart(_style(pr))
    with col2:
        roc = (
            alt.Chart(roc_df)
            .mark_line(strokeWidth=2.2)
            .encode(
                x=alt.X("FPR:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("TPR:Q", title="True positive rate (Recall)", scale=alt.Scale(domain=[0, 1.04])),
                color=alt.Color(
                    "Algorithm:N",
                    title=None,
                    legend=alt.Legend(orient="bottom", columns=2),
                ),
            )
            .properties(title="ROC curves", height=310)
        )
        altair_chart(_style(roc))


def cost_tradeoff_chart(sweep_df: pd.DataFrame) -> None:
    opt = sweep_df.loc[sweep_df["Total cost ($)"].idxmin()]
    opt_df = pd.DataFrame(
        [
            {
                "alert_rate_pct": float(opt["alert_rate_pct"]),
                "Total cost ($)": float(opt["Total cost ($)"]),
                "label": f"Min cost: ${float(opt['Total cost ($)']):,.0f} @ {float(opt['alert_rate_pct']):.0f}%",
            }
        ]
    )
    y_max = float(sweep_df["Total cost ($)"].max()) * 1.12
    x_mid = float(sweep_df["alert_rate_pct"].median())

    cost_line = (
        alt.Chart(sweep_df)
        .mark_line(point=True, color="#d9483b", strokeWidth=2.4)
        .encode(
            x=alt.X("alert_rate_pct:Q", title="Alert rate / contamination (%)"),
            y=alt.Y("Total cost ($):Q", title="Total financial cost ($)", scale=alt.Scale(domain=[0, y_max])),
            tooltip=["alert_rate_pct:Q", "Precision:Q", "Recall:Q", "FP:Q", "FN:Q", "Total cost ($):Q"],
        )
    )
    opt_pt = (
        alt.Chart(opt_df)
        .mark_circle(size=160, color="#2a9d6f", stroke="#1d1d1d", strokeWidth=1.0)
        .encode(x="alert_rate_pct:Q", y="Total cost ($):Q")
    )
    on_left = float(opt["alert_rate_pct"]) <= x_mid
    opt_lbl = (
        alt.Chart(opt_df)
        .mark_text(
            align="left" if on_left else "right",
            dx=10 if on_left else -10,
            dy=-14,
            fontSize=11,
            fontWeight="bold",
            color="#1b7a56",
        )
        .encode(x="alert_rate_pct:Q", y="Total cost ($):Q", text="label:N")
    )
    chart = alt.layer(cost_line, opt_pt, opt_lbl).properties(
        title="Financial cost (FP friction + FN fraud loss) vs alert rate",
        height=280,
    )
    altair_chart(_style(chart))
