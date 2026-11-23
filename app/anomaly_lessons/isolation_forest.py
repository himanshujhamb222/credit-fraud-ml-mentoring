"""Lesson 5: Isolation Forest — built up gently from first principles, crisp & visual."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC
from anomaly_lab.isolation_forest import (
    anomaly_score,
    expected_path_length,
    harmonic_number,
    run_2d_isolation_demo,
    score_curve_table,
)
from anomaly_lessons.charts import (
    isolation_1d_cuts_chart,
    isolation_2d_chart,
    isolation_score_curve_chart,
)
from anomaly_lessons.ui import lesson_header, render_interview_qa, working
from st_compat import dataframe, metric, mermaid_chart


@st.cache_data(max_entries=32)
def _cached_iforest_demo(n_trees: int, psi: int, contamination: float, seed: int):
    return run_2d_isolation_demo(n_trees=n_trees, psi=psi, contamination=contamination, tree_seed=seed)


def render() -> None:
    lesson_header(
        "5. Isolation Forest",
        "Start simple: anomalies sit alone, so random cuts separate them in 1-2 steps while normal points need many.",
        priority_badge=":red-badge[Highest priority] :orange-badge[Extra deep]",
        interview_focus=[
            "First principle: isolate anomalies directly instead of measuring distances or density.",
            "Short path = anomaly; long path = normal (averaged across many random trees).",
            "How path length converts to a 0-to-1 score + why small subsamples (256) help.",
            "`contamination` only moves the cutoff; axis-aligned cuts struggle on diagonal trends.",
        ],
    )

    tab_base, tab_score, tab_code, tab_qa = st.tabs(
        [
            "1. First principles",
            "2. Score math & 2D splits",
            "3. Code & limits",
            "4. Interview prep",
        ]
    )

    with tab_base:
        st.markdown(
            "**Big idea:** Don't model normal behavior first. Just ask: "
            "*how many random cuts does it take to put a transaction in a box by itself?*"
        )
        st.markdown(
            r"""
- **Normal spend (\$42, \$48, \$51, \$55, \$59):** packed tightly together → needs **many cuts**.
- **Fraud (\$490):** sits far away with a huge empty gap → almost any random cut separates it in **1 cut**.
"""
        )

        cut_step = st.slider("Step through random cuts (1 → 4)", 1, 4, 1, 1, key="if_1d_step")
        isolation_1d_cuts_chart(cut_step)

        step_notes = {
            1: r"Cut 1 at **\$210** lands in the huge empty gap → **\$490 is isolated immediately (path = 1)**!",
            2: r"Cut 2 at **\$50** splits the normal crowd into `[$42, $48]` and `[$51, $55, $59]`.",
            3: r"Cut 3 at **\$53** isolates **\$51** (path = 3); **\$55** and **\$59** are still packed together.",
            4: r"Cut 4 at **\$57** finally separates **\$55** and **\$59** (**path = 4** — 4× deeper than fraud!).",
        }
        st.info(step_notes[cut_step], icon=":material/content_cut:")

        col_tree, col_forest = st.columns([3, 2])
        with col_tree:
            st.markdown("**Those 4 cuts form one Isolation Tree:**")
            mermaid_chart(
                """
flowchart TD
    R["Cut 1: amount < $210?"] -->|No| F["$490 alone! (path = 1)"]
    R -->|Yes| C2["Cut 2: amount < $50?"]
    C2 -->|Yes| L["$42, $48 (path = 3)"]
    C2 -->|No| C3["Cut 3: amount < $53?"]
    C3 -->|Yes| M["$51 alone (path = 3)"]
    C3 -->|No| C4["Cut 4: amount < $57?"]
    C4 -->|Yes| N1["$55 alone (path = 4)"]
    C4 -->|No| N2["$59 alone (path = 4)"]
"""
            )
        with col_forest:
            with st.container(border=True):
                st.markdown("**From 1 Tree to a Forest**")
                st.markdown(
                    """
1. One random tree can get unlucky.
2. Build **100 random trees**, each on a small random sample (**256 rows**).
3. Record how many cuts point $x$ needs in each tree: $h(x)$.
4. Take the average across all trees: **$E[h(x)]$**.
   - **Small $E[h(x)]$** → **Anomaly**
   - **Large $E[h(x)]$** → **Normal**
"""
                )

    with tab_score:
        st.markdown(
            "**Why convert path length to a 0-to-1 score?** "
            "A path of `4` cuts is *short* in a 256-row tree, but *long* in a 16-row tree. "
            "So we compare $E[h(x)]$ against the **typical normal path $c(n)$** for a tree of size $n$:"
        )
        st.latex(r"s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}} \in (0, 1]")
        with st.container(border=True):
            st.markdown(
                r"""
- **$E[h(x)] \ll c(n)$** (much fewer cuts than typical) $\Rightarrow$ **$s \to 1.0$ (Anomaly)**
- **$E[h(x)] \approx c(n)$** (typical number of cuts) $\Rightarrow$ **$s = 2^{-1} = 0.5$ (Normal)**
- **$E[h(x)] \gg c(n)$** (deep inside dense crowd) $\Rightarrow$ **$s \to 0.0$ (Very normal)**
"""
            )

        with st.expander("Formula for typical path c(n) & interactive score curve (interview detail)"):
            st.latex(r"c(n) = 2H(n-1) - \frac{2(n-1)}{n}, \qquad H(i) \approx \ln(i) + 0.5772")
            cc1, cc2 = st.columns(2)
            with cc1:
                calc_psi = st.select_slider("Subsample size (psi)", options=[32, 64, 128, 256, 512], value=256, key="if_calc_psi")
            c_val = expected_path_length(calc_psi)
            with cc2:
                calc_h = st.slider("Average path E[h(x)]", 1.0, round(c_val * 2.0, 1), 3.5, 0.25, key="if_calc_h")
            s_val = float(anomaly_score(calc_h, calc_psi))
            working(
                [
                    f"Typical path c({calc_psi}) = {c_val:.2f}  (using H({calc_psi - 1}) = {harmonic_number(calc_psi - 1):.2f})",
                    f"Score s = 2^(-{calc_h:.2f} / {c_val:.2f}) = {s_val:.3f}",
                ]
            )
            isolation_score_curve_chart(score_curve_table(calc_psi), highlight_h=calc_h, highlight_s=s_val)

        st.subheader("Interactive 2D splits (amount + velocity)")
        st.markdown("Now extend the exact same random cuts to 2 features: pick a random feature, then a random split.")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            n_trees = st.slider("Trees (T)", 5, 100, 40, 5, key="if_trees")
        with p2:
            psi = st.slider("Subsample (psi)", 12, 55, 32, 4, key="if_psi")
        with p3:
            contamination = st.slider("Contamination", 0.02, 0.20, 0.07, 0.01, key="if_contam")
        with p4:
            seed = st.number_input("Seed", min_value=1, max_value=99, value=7, step=1, key="if_seed")

        show_splits = st.toggle("Show Tree #1 random cuts", value=True, key="if_show_splits")
        demo = _cached_iforest_demo(n_trees, psi, contamination, int(seed))

        norm_path = float(demo.points[demo.points["true_role"] == "Normal"]["avg_path_length"].mean())
        anom_path = float(demo.points[demo.points["true_role"] == "Anomaly"]["avg_path_length"].mean())

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric("Typical path c(psi)", f"{demo.c_psi:.2f}")
        with m2:
            metric("Normal avg cuts", f"{norm_path:.2f}")
        with m3:
            metric("Anomaly avg cuts", f"{anom_path:.2f}", delta=f"{anom_path - norm_path:.2f} fewer")
        with m4:
            metric("Flagged", int(demo.points["flagged"].sum()))

        isolation_2d_chart(demo.points, demo.splits, show_splits=show_splits)
        top8 = demo.points.sort_values("anomaly_score", ascending=False).head(8)
        dataframe(top8[["tx_id", "amount", "tx_velocity_24h", "true_role", "avg_path_length", "anomaly_score", "predicted"]], hide_index=True)

    with tab_code:
        st.markdown(
            r"""
**Why small subsamples ($\psi = 256$) win:**
- **Fast:** $O(T\psi\log\psi)$ train, $O(T\log\psi)$ score — independent of total dataset size $N$.
- **Cleaner splits:** prevents **swamping** (normal points crowding outliers) and **masking** (clustered fraud hiding itself).
- **No feature scaling** needed.
"""
        )
        st.code(
            """from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=200, max_samples=256, contamination=0.02, random_state=42)
iso.fit(X_tx)  # no scaling required

scores = -iso.score_samples(X_tx)   # negate: sklearn returns lower = more anomalous
flags = iso.predict(X_tx) == -1
""",
            language="python",
        )
        lim = pd.DataFrame(
            [
                {"Limit": "Axis-aligned splits", "Fix": "Extended Isolation Forest (random angled cuts)"},
                {"Limit": "Local density blind spot", "Fix": "Use LOF for multi-density clusters"},
                {"Limit": "Noise features dilute splits", "Fix": "Drop dead/irrelevant columns"},
                {"Limit": "Contamination = cutoff only", "Fix": "Rank continuous `-score_samples` for Top-K review"},
            ]
        )
        dataframe(lim, hide_index=True)

    with tab_qa:
        render_interview_qa(CORE_QA_BY_TOPIC["Isolation Forest"], key_prefix="qa_iforest")
