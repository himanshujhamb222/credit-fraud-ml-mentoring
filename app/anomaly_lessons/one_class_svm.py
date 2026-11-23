"""Lesson 6: One-Class SVM — explain-like-I'm-5 intuition, then build the math step by step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC
from anomaly_lab.one_class_svm import build_1d_spotlight_demo, rbf_similarity, run_2d_ocsvm_demo
from anomaly_lessons.charts import ocsvm_1d_spotlight_chart, ocsvm_boundary_chart
from anomaly_lessons.ui import lesson_header, render_interview_qa, working
from st_compat import dataframe, metric, mermaid_chart


@st.cache_data(max_entries=32)
def _cached_ocsvm_demo(kernel: str, nu: float, gamma: float):
    return run_2d_ocsvm_demo(kernel=kernel, nu=nu, gamma=gamma)


def render() -> None:
    lesson_header(
        "6. One-Class SVM",
        "Draw the tightest chalk circle around normal spending — anything standing outside the circle is flagged.",
        priority_badge=":red-badge[Highest priority — Machine Learning]",
        interview_focus=[
            "**Only normal examples:** no fraud labels needed — wrap a boundary around normal spend.",
            "**Campfires on the edge:** only border points (support vectors) shine a closeness glow (RBF).",
            r"**Two knobs:** $\nu$ controls fence tightness; $\gamma$ controls how wide each glow reaches.",
            r"**Limits:** slow $O(n^2)$–$O(n^3)$ training and **must** scale features first.",
        ],
    )

    tab_base, tab_demo, tab_code, tab_qa = st.tabs(
        [
            "1. ELI5 → Math",
            "2. Two knobs & 2D fence",
            "3. Code & limits",
            "4. Interview prep",
        ]
    )

    with tab_base:
        with st.container(border=True):
            st.markdown(":material/lightbulb: **Explain like I'm 5**")
            st.markdown(
                "Imagine normal transactions are **kids playing in a park**. You have **no photos of strangers** — only normal kids. "
                "Draw a tight chalk circle around the group; anyone standing **outside** is an outsider."
            )
            st.markdown(
                """
- **Only edge kids matter:** kids in the middle can move around; only the few kids on the **edge** hold up the chalk line (**support vectors**).
- **Campfire test:** put a warm campfire at each edge kid. Close to a fire = **warm (Normal)**; far out in the dark = **cold (Anomaly)**.
"""
            )

        st.markdown("#### Build the math in 3 tiny steps")
        s_a, s_b, s_c = st.columns(3)
        with s_a:
            with st.container(border=True):
                st.markdown("**Step 1: Closeness to 1 fire**")
                st.caption("Right on the fire (`d = 0`) → `1.0`. Far away → `0.0`.")
                st.latex(r"K(x, s) = e^{-\gamma d^2}")
                st.caption("`d = 0` → `1.0` · `d = 2` → `0.02` (cold!)")
        with s_b:
            with st.container(border=True):
                st.markdown("**Step 2: Add all edge fires**")
                st.caption("Sum the warmth from every edge post (weight = fire size).")
                st.latex(r"W(x) = \sum_{i} \alpha_i K(x, s_i)")
                st.caption("Interior points have weight `0` (ignored).")
        with s_c:
            with st.container(border=True):
                st.markdown("**Step 3: Warm enough?**")
                st.caption(r"Compare total warmth against one cutoff line $\rho$.")
                st.latex(r"f(x) = W(x) - \rho")
                st.caption(r"$f(x) \ge 0$ → **Normal** · $f(x) < 0$ → **Anomaly**")

        st.subheader(r"Try it: 2 normal habits (\$50 and \$150)")
        s1, s2 = st.columns(2)
        with s1:
            g1d = st.slider(
                "Campfire narrowness (gamma)",
                0.15,
                6.0,
                1.8,
                0.15,
                key="ocsvm_1d_gamma",
                help="Small gamma = wide warm blanket; large gamma = tiny narrow bubbles.",
            )
        with s2:
            rho1d = st.slider(
                "Warmth cutoff line (rho)",
                0.10,
                1.10,
                0.45,
                0.05,
                key="ocsvm_1d_rho",
                help="Transactions below this red line are too cold -> flagged as anomalies.",
            )

        spot = build_1d_spotlight_demo(gamma=g1d, rho=rho1d)
        ocsvm_1d_spotlight_chart(spot.curve, spot.probes, rho=rho1d)

        gap_row = spot.probes[spot.probes["amount"] == 100.0].iloc[0]
        norm_edge = spot.probes[spot.probes["amount"] == 42.0].iloc[0]
        if norm_edge["verdict"] == "Anomaly (Outside)":
            note = r"**Too strict (overfitting):** Even normal **\$42** is too cold — lower `gamma` or `rho`."
        elif gap_row["verdict"] == "Normal (Inside)":
            note = r"**Too loose (underfitting):** The **\$50** and **\$150** glows merged, so empty-gap **\$100** looks warm."
        else:
            note = r"**Just right:** Warm bubbles hug **\$50** and **\$150**, catching both the **\$100** gap and **\$240** fraud!"
        st.info(note, icon=":material/fence:")

        col_flow, col_nums = st.columns([5, 6])
        with col_flow:
            st.markdown("**How a new transaction gets scored:**")
            mermaid_chart(
                """
flowchart TD
    A["1. Scale features (StandardScaler)"] --> B["2. Measure distance to edge posts (SVs)"]
    B --> C["3. Add campfire glows: warmth(x)"]
    C --> D{"warmth(x) >= rho?"}
    D -->|Yes| N["Warm enough -> Normal (+1)"]
    D -->|No| F["Too cold -> Anomaly (-1)"]
"""
            )
        with col_nums:
            st.markdown("**Live arithmetic for 3 transactions:**")
            p52 = spot.probes[spot.probes["amount"] == 52.0].iloc[0]
            p100 = spot.probes[spot.probes["amount"] == 100.0].iloc[0]
            p240 = spot.probes[spot.probes["amount"] == 240.0].iloc[0]
            working(
                [
                    f"glow(x, post) = exp(-{g1d:.2f} * ((x - post) / 50)^2)",
                    f"$52  (Normal): {p52['k_50']:.2f} + {p52['k_150']:.2f} = {p52['warmth']:.2f} vs {rho1d:.2f} -> {p52['verdict']}",
                    f"$100 (Gap):    {p100['k_50']:.2f} + {p100['k_150']:.2f} = {p100['warmth']:.2f} vs {rho1d:.2f} -> {p100['verdict']}",
                    f"$240 (Fraud):  {p240['k_50']:.3f} + {p240['k_150']:.3f} = {p240['warmth']:.3f} vs {rho1d:.2f} -> {p240['verdict']}",
                ]
            )
            st.caption(r"Notice \$240 is far from both campfires, so its warmth drops to ~0.")

    with tab_demo:
        st.markdown("**In practice, only 2 knobs matter** — both have plain-English meanings:")
        with st.container(border=True):
            st.markdown(
                r"""
- **`nu` ($\nu$, fence tightness):** max % of training points allowed outside the fence (and min % of edge posts: $\text{outliers} \le \nu \le \text{SV fraction}$).
- **`gamma` ($\gamma$, glow width):** small $\gamma$ = one smooth loose oval; large $\gamma$ = spiky islands hugging every point.
- **Decision rule:** $f(x) \ge 0$ = inside fence (**Normal `+1`**) · $f(x) < 0$ = outside fence (**Anomaly `-1`**).
"""
            )

        st.subheader("Interactive 2D fence (amount + velocity)")
        c1, c2, c3 = st.columns(3)
        with c1:
            kernel = st.selectbox("Fence shape (kernel)", ["rbf", "linear", "poly"], index=0, key="ocsvm_kernel")
        with c2:
            nu = st.slider("nu (fence tightness)", 0.02, 0.35, 0.08, 0.02, key="ocsvm_nu")
        with c3:
            gamma = st.slider("gamma (glow narrowness)", 0.05, 3.0, 0.50, 0.05, key="ocsvm_gamma")

        res = _cached_ocsvm_demo(kernel=kernel, nu=nu, gamma=gamma)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric("Edge posts (SVs)", res.n_support_vectors)
        with m2:
            metric("SV frac (>= nu)", f"{res.sv_fraction:.1%}")
        with m3:
            metric("Train outside (<= nu)", f"{res.flagged_fraction:.1%}")
        with m4:
            anom_caught = int(res.points[res.points["true_role"] == "Anomaly"]["flagged"].sum())
            metric("Anomalies caught", f"{anom_caught} / 5")

        ocsvm_boundary_chart(res.grid, res.points)

        sv_pts = np.array([[0.0, 0.0], [1.5, 1.2]])
        k_close = rbf_similarity(np.array([0.2, 0.1]), sv_pts, gamma=gamma)
        k_far = rbf_similarity(np.array([3.2, -2.5]), sv_pts, gamma=gamma)
        working(
            [
                f"2D campfire glow K(x, s) = exp(-{gamma:.2f} * ||x - s||^2):",
                f"  Near edge post (normal): K = [{k_close[0]:.3f}, {k_close[1]:.3f}]  -> warmth >= rho -> Normal",
                f"  Far away (anomaly):      K = [{k_far[0]:.5f}, {k_far[1]:.5f}]  -> warmth ~ 0 < rho -> Flagged",
            ]
        )

        with st.expander("Formal optimization math (interview detail — built from the same idea)"):
            st.markdown(
                r"""
1. **Lift onto a ball:** RBF lifts every point onto the surface of a sphere (distance 1 from zero).
2. **Slice away from zero:** push a flat knife as far from the origin `(0, 0, ...)` as possible while keeping normal points on the right side.
3. **Allow a tiny mistake budget:** let a few noisy normal points peek across by slack $\xi_i$ so one weird point doesn't stretch the whole fence.
"""
            )
            st.latex(
                r"\min_{w,\,\xi,\,\rho}\; \underbrace{\frac{1}{2}\|w\|^2}_{\text{keep fence smooth}} + "
                r"\underbrace{\frac{1}{\nu n}\sum_{i=1}^{n}\xi_i}_{\text{mistake penalty}} - "
                r"\underbrace{\rho}_{\text{push far from zero}}"
                r"\qquad \text{s.t.}\quad w^T\phi(x_i) \ge \rho - \xi_i,\;\; \xi_i \ge 0"
            )
            st.markdown(
                r"""
- **Why $\frac{1}{2}\|w\|^2$?** Keeps the fence smooth instead of jagged.
- **Why $\xi_i \ge 0$?** Soft forgiveness for noisy training points (controlled by $\nu$).
- **Why dual weights $\alpha_i$?** Solver sets $\alpha_i > 0$ *only* for edge points (support vectors).
"""
            )

    with tab_code:
        st.markdown(
            r"**Mandatory first step — scale features!** "
            r"Without `StandardScaler`, \$200 vs 3 tx/day turns squared distance into 40,000 vs 9, making velocity invisible."
        )
        st.code(
            """from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ocsvm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"))
ocsvm.fit(X_train_normal)  # train on clean normal transactions

scores = -ocsvm.decision_function(X_eval)  # negate: larger = more anomalous
flags = ocsvm.predict(X_eval) == -1
""",
            language="python",
        )
        lim = pd.DataFrame(
            [
                {"Limit": "Slow on big data: O(n^2)-O(n^3)", "Fix": "Nystroem + SGDOneClassSVM, or Isolation Forest"},
                {"Limit": "Distance breaks if unscaled", "Fix": "Always wrap with StandardScaler / RobustScaler"},
                {"Limit": "Sensitive to gamma and nu", "Fix": "Big gamma = spiky islands; small = loose oval"},
                {"Limit": "Dirty training data stretches fence", "Fix": "Train on verified-normal transactions only"},
            ]
        )
        dataframe(lim, hide_index=True)

    with tab_qa:
        render_interview_qa(CORE_QA_BY_TOPIC["One-Class SVM"], key_prefix="qa_ocsvm")
