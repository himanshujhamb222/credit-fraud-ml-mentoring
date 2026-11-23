"""Lesson 7: Autoencoders — ELI5 first, then build the math on top."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.autoencoder import customer_vs_global_demo, run_autoencoder_demo
from anomaly_lab.data import BENCHMARK_FEATURES, make_financial_benchmark
from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC
from anomaly_lessons.charts import (
    autoencoder_error_dist_chart,
    autoencoder_feature_error_chart,
    autoencoder_loss_chart,
    autoencoder_toy_rebuild_chart,
)
from anomaly_lessons.ui import lesson_header, render_interview_qa, working
from st_compat import choice, dataframe, metric, mermaid_chart


@st.cache_data(max_entries=16)
def _cached_benchmark():
    return make_financial_benchmark()


@st.cache_data(max_entries=32)
def _cached_ae_run(bottleneck_dim: int, epochs: int, train_normal_only: bool, thresh_q: float):
    df = _cached_benchmark()
    return run_autoencoder_demo(
        df,
        BENCHMARK_FEATURES,
        bottleneck_dim=bottleneck_dim,
        epochs=epochs,
        train_normal_only=train_normal_only,
        threshold_quantile=thresh_q,
    )


@st.cache_data(max_entries=4)
def _cached_customer_vs_global():
    return customer_vs_global_demo(_cached_benchmark(), BENCHMARK_FEATURES)


def _toy_tx_frame(example: str) -> tuple[pd.DataFrame, list[str], str, str, str]:
    """Return a 3-clue Real vs Redrawn comparison with simple whole-number math."""
    if example.startswith("Normal"):
        real = [2.0, 1.0, 1.0]
        rebuilt = [2.0, 1.2, 1.0]
        lines = [
            'Tiny sticky note (1 word): "Local lunch"',
            "Real x:     [Amount=2.0, Distance=1.0, Speed=1.0]",
            "Redrawn x^: [Amount=2.0, Distance=1.2, Speed=1.0]",
            "Squared gaps: [0.00, 0.04, 0.00]  ->  Avg mistake (MSE) = 0.01  -> NORMAL",
        ]
        title = "Normal receipt: Friend 2 redraws it almost perfectly"
        note = "Friend 2 knows normal lunches by heart, so the redraw error is tiny (0.01)."
        kind = "info"
    elif example.startswith("Fraud"):
        real = [2.0, 9.0, 1.0]
        rebuilt = [2.0, 1.5, 1.0]
        lines = [
            'Tiny sticky note (1 word): "Local lunch"  (no word for 9 km jump!)',
            "Real x:     [Amount=2.0, Distance=9.0, Speed=1.0]",
            "Redrawn x^: [Amount=2.0, Distance=1.5, Speed=1.0]",
            "Squared gaps: [0.00, 56.25, 0.00] ->  Avg mistake (MSE) = 18.75 -> FRAUD!",
        ]
        title = "Fraud receipt: Distance jumps to 9.0, Friend 2 guesses 1.5 -> HUGE mistake!"
        note = "Instant explanation: Distance caused 100% of the error (56.25 out of 56.25)."
        kind = "error"
    else:
        real = [2.0, 9.0, 1.0]
        rebuilt = [2.0, 9.0, 1.0]
        lines = [
            'HUGE sticky note (3 words): "Amt=2, Dist=9, Spd=1"  (copied word-for-word!)',
            "Real x:     [Amount=2.0, Distance=9.0, Speed=1.0]",
            "Redrawn x^: [Amount=2.0, Distance=9.0, Speed=1.0]",
            "Squared gaps: [0.00, 0.00, 0.00]  ->  Avg mistake (MSE) = 0.00  -> FRAUD ESCAPED!",
        ]
        title = "Cheat mode (huge sticky note): copies fraud verbatim -> error = 0!"
        note = "That's why the bottleneck MUST be smaller than the input (k < d) — otherwise it cheats!"
        kind = "warning"

    feats = ["Amount", "Distance", "Speed"]
    rows = []
    for f, r_val, g_val in zip(feats, real, rebuilt):
        rows.append({"feature": f, "series": "Real (x)", "value": r_val})
        rows.append({"feature": f, "series": "Redrawn (x_hat)", "value": g_val})
    return pd.DataFrame(rows), lines, title, note, kind


def render() -> None:
    lesson_header(
        "7. Autoencoders",
        "Play a copy game through a tiny sticky note: normal habits redraw cleanly, weird fraud comes out wrong.",
        priority_badge=":red-badge[Highest priority] :orange-badge[Extra deep]",
        interview_focus=[
            r"Tiny sticky note (bottleneck $k < d$) stops verbatim copying.",
            "Train on normal rows only; big redraw mistake (MSE) flags fraud.",
            r"Per-column mistake $(x_j - \hat{x}_j)^2$ explains every alert.",
            "Pass customer-relative ratios so one shared model scales.",
        ],
    )

    tab_eli5, tab_demo, tab_code, tab_qa = st.tabs(
        ["1. ELI5 -> Math", "2. Live playground", "3. Code & limits", "4. Interview prep"]
    )

    with tab_eli5:
        col_story, col_diag = st.columns([5, 6])
        with col_story:
            with st.container(border=True):
                st.markdown(":material/lightbulb: **Explain like I'm 5: the tiny sticky-note game**")
                st.markdown(
                    """
Imagine two friends playing a copy game using **only normal receipts**:

1. **Squeeze:** Friend 1 shrinks a 5-clue receipt onto a tiny **2-word sticky note**.
2. **Redraw:** Friend 2 rebuilds all 5 clues using *only* that tiny note.
3. **Catch fraud:** Normal receipts redraw easily. Weird fraud doesn't fit the note, so the redraw comes out **way wrong**!
"""
                )
        with col_diag:
            mermaid_chart(
                """
flowchart LR
    In["Real receipt x<br/>(5 clues)"] --> Enc["Friend 1: Squeeze<br/>(Encoder)"]
    Enc --> Bot["Tiny sticky note z<br/>(only 2 clues!)"]
    Bot --> Dec["Friend 2: Redraw<br/>(Decoder)"]
    Dec --> Out["Redrawn x_hat<br/>(5 clues)"]
    In -.-> Diff["Big mistake?<br/>-> Anomaly!"]
    Out -.-> Diff
"""
            )

        sample = choice(
            "Try the game on a 3-clue receipt",
            [
                "Normal (local lunch)",
                "Fraud (9 km jump)",
                "Cheat mode (huge sticky note)",
            ],
            default="Fraud (9 km jump)",
            key="ae_toy_choice",
            horizontal=True,
        )
        toy_df, toy_lines, toy_title, toy_note, toy_kind = _toy_tx_frame(sample)
        autoencoder_toy_rebuild_chart(toy_df, title=toy_title)
        if toy_kind == "error":
            st.error(toy_note, icon=":material/warning:")
        elif toy_kind == "warning":
            st.warning(toy_note, icon=":material/report:")
        else:
            st.info(toy_note, icon=":material/check_circle:")
        working(toy_lines)

        st.subheader("Now put math on top of the game (3 tiny steps)")
        st.caption("Every math symbol is just a short nickname for a piece of the sticky-note game above.")

        m1, m2, m3 = st.columns(3)
        with m1:
            with st.container(border=True):
                st.markdown("**Step 1: Squeeze (Encoder)**")
                st.latex(r"z = f(x)")
                st.caption(r"$x$ = 5 real clues $\to$ $z$ = tiny 2-number note")
        with m2:
            with st.container(border=True):
                st.markdown("**Step 2: Redraw (Decoder)**")
                st.latex(r"\hat{x} = g(z)")
                st.caption(r"$\hat{x}$ ('x-hat') = Friend 2's redrawn guess")
        with m3:
            with st.container(border=True):
                st.markdown("**Step 3: Score the mistake (MSE)**")
                st.latex(r"\text{MSE}(x) = \frac{1}{d}\sum_{j=1}^{d}(x_j - \hat{x}_j)^2")
                st.caption(r"Average of $(\text{real} - \text{guess})^2$ across $d$ clues")

        with st.container(border=True):
            st.markdown(
                r"""
- **Why square $(x_j - \hat{x}_j)^2$?** So `+5` and `-5` don't cancel out, and huge misses jump out.
- **Decision rule:** Pick a cutoff $\tau$ (e.g. 95th percentile of normal mistakes). $\text{MSE}(x) > \tau \Rightarrow$ **Anomaly**.
- **Why normal-only practice?** If fraud is in training, the friends invent a sticky-note code for fraud too!
"""
            )

        with st.expander("Formal math & interview details (weights, gradient descent, PCA link)"):
            st.latex(
                r"z = f_\theta(x) \in \mathbb{R}^k \;(k < d), \qquad "
                r"\hat{x} = g_\phi(z) \in \mathbb{R}^d, \qquad "
                r"W \leftarrow W - \eta\,\nabla_W \mathcal{L}"
            )
            st.markdown(
                r"""
- **$\theta, \phi$:** neural network weights tuned on normal data to minimize average MSE ($\mathcal{L}$).
- **$W \leftarrow W - \eta\,\nabla_W \mathcal{L}$:** nudge weights downhill by learning rate $\eta$ each epoch.
- **PCA link:** A purely linear autoencoder (no ReLU/tanh) with MSE loss spans the exact same subspace as **PCA**.
"""
            )

    with tab_demo:
        st.markdown(
            """
- **Experiment 1:** Slide **sticky-note size (`k`)** up to `5` to watch Cheat Mode ruin separation!
- **Experiment 2:** Turn off **Normal-only train** to see what happens when fraud leaks into practice data.
"""
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bottleneck = st.slider("Sticky-note size (k of 5)", 1, 5, 2, 1, key="ae_bottleneck")
        with c2:
            epochs = st.slider("Practice rounds (epochs)", 40, 320, 200, 40, key="ae_epochs")
        with c3:
            thresh_q = st.slider("Alert cutoff (quantile)", 0.85, 0.99, 0.95, 0.01, key="ae_q")
        with c4:
            normal_only = st.toggle("Normal-only train", value=True, key="ae_normal_only")

        run = _cached_ae_run(bottleneck, epochs, normal_only, thresh_q)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric("Normal mistake (MSE)", f"{run.normal_mean_mse:.3f}")
        with m2:
            metric("Anomaly mistake (MSE)", f"{run.anomaly_mean_mse:.3f}")
        with m3:
            metric("Separation", f"{run.separation_ratio:.1f}x")
        with m4:
            metric("Alert cutoff", f"{run.threshold:.3f}")

        if bottleneck >= 4:
            st.warning(
                r"Cheat mode! Sticky note too wide ($k \ge 4$) $\Rightarrow$ copies inputs $\Rightarrow$ separation drops.",
                icon=":material/warning:",
            )

        anom_feat = run.feature_errors[run.feature_errors["group"] == "True anomaly"]
        top_feat = anom_feat.sort_values("mean_sq_error", ascending=False).iloc[0]
        top_anom = run.eval_table[run.eval_table["is_anomaly"] == 1].sort_values("recon_mse", ascending=False).iloc[0]
        working(
            [
                f"Settings: sticky note k = {bottleneck}/5, epochs = {epochs}, normal_only = {normal_only}",
                f"Avg MSE:  Normal = {run.normal_mean_mse:.3f}, Anomaly = {run.anomaly_mean_mse:.3f} ({run.separation_ratio:.1f}x higher)",
                f"Top alert ({top_anom['tx_id']}): MSE = {top_anom['recon_mse']:.3f} vs cutoff {run.threshold:.3f}",
                f"Biggest broken clue: {top_feat['feature']} (avg sq error = {top_feat['mean_sq_error']:.3f})",
            ]
        )

        ch1, ch2 = st.columns(2)
        with ch1:
            autoencoder_loss_chart(run.loss_history)
        with ch2:
            autoencoder_error_dist_chart(run.eval_table, threshold=run.threshold)
        autoencoder_feature_error_chart(run.feature_errors)

        st.markdown(r"**Scope:** Global models miss a **\$380** Everyday spend; customer-level models flag it.")
        dataframe(_cached_customer_vs_global(), hide_index=True)

    with tab_code:
        st.code(
            """import torch.nn as nn

class TxAutoencoder(nn.Module):
    def __init__(self, d=5, k=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, k))
        self.decoder = nn.Sequential(nn.Linear(k, 16), nn.ReLU(), nn.Linear(16, d))

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Train on standardized NORMAL rows only:
# loss = ((X_norm - model(X_norm)) ** 2).mean()
# Score = per-row MSE:
# recon_mse = ((X_eval - model(X_eval)) ** 2).mean(dim=1)
""",
            language="python",
        )
        lim = pd.DataFrame(
            [
                {"Limit": "Wide bottleneck copies fraud", "Fix": "Tight k < d, L2 decay, dropout, Denoising/VAE"},
                {"Limit": "Fraud in training set", "Fix": "Drop chargebacks; trim top-1% MSE and retrain"},
                {"Limit": "Large scales dominate MSE", "Fix": "Standardize and log1p-transform skewed columns"},
                {"Limit": "Per-user models don't scale", "Fix": "Pass customer-relative ratios to a shared model"},
            ]
        )
        dataframe(lim, hide_index=True)

    with tab_qa:
        render_interview_qa(CORE_QA_BY_TOPIC["Autoencoders"], key_prefix="qa_ae")
