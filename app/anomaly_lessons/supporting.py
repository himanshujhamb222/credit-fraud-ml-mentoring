"""Lesson 8: Supporting algorithms — LOF, Graph/GNN, Temporal (first principles, then compact math)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_lab.data import make_temporal_sequence, make_transaction_graph
from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC
from anomaly_lab.supporting import (
    run_graph_message_passing_demo,
    run_lof_dense_vs_sparse_demo,
    run_temporal_sequence_demo,
)
from anomaly_lessons.charts import graph_network_chart, lof_cluster_chart, temporal_sequence_chart
from anomaly_lessons.ui import lesson_header, render_interview_qa, working
from st_compat import dataframe, metric, mermaid_chart


@st.cache_data(max_entries=16)
def _cached_lof(k: int):
    return run_lof_dense_vs_sparse_demo(k=k)


@st.cache_data(max_entries=8)
def _cached_graph(steps: int):
    nodes, edges = make_transaction_graph()
    return run_graph_message_passing_demo(nodes, edges, steps=steps)


@st.cache_data(max_entries=8)
def _cached_temporal(window: int):
    return run_temporal_sequence_demo(make_temporal_sequence(), window=window)


def _render_lof_first_principles() -> None:
    st.markdown(
        "**Failure mode:** Global distance mixes dense everyday spend with wide affluent spread."
    )
    st.markdown(
        "**LOCAL_OUT** looks globally fine but is lonely next to its own $k$ neighbors."
    )
    with st.container(border=True):
        st.markdown(
            "Country house **100 m** apart = normal. Same gap in a crowded city = weird."
        )
        st.markdown(
            r"""
- **Wrong question:** *How far from everyone?* (flags sparse normals.)
- **Right question:** *Lonely vs your own neighbors?*
"""
        )
    compare = pd.DataFrame(
        [
            {
                "Detector": "Global Z / distance",
                "LOCAL_OUT": "Miss",
                "Sparse normal A01": "OK",
            },
            {
                "Detector": "LOF (local)",
                "LOCAL_OUT": "Flag (> 1.5)",
                "Sparse normal A01": "OK (~1.0)",
            },
        ]
    )
    dataframe(compare, hide_index=True)


def _render_lof_math() -> None:
    st.markdown("**Intuition:** Neighbors' average density divided by mine → ratio near 1 is normal.")
    st.latex(
        r"\text{LOF}_k(p) = \frac{\frac{1}{|N_k(p)|}\sum_{o \in N_k(p)} \text{lrd}_k(o)}{\text{lrd}_k(p)}"
    )
    st.latex(
        r"\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p,o)}"
    )
    st.markdown(
        r"""
- **$N_k(p)$:** $k$ nearest neighbors of point $p$.
- **$\text{lrd}_k(p)$:** inverse mean reachability distance (local density).
- **Flag:** $\text{LOF} > 1.5$ → much lonelier than neighbors.
"""
    )
    working(
        [
            "Interview check: mean lrd(neighbors) = 0.80, lrd(p) = 0.20",
            "LOF(p) = 0.80 / 0.20 = 4.0  ->  strong local outlier",
        ]
    )
    with st.expander("Reachability distance (k-dist floor)"):
        st.markdown(
            "If two points overlap, raw $1/d$ explodes. LOF floors distance at each neighbor's $k$-distance."
        )
        st.latex(
            r"\text{reach-dist}_k(p, o) = \max\bigl(\text{k-dist}(o),\, d(p, o)\bigr)"
        )
        st.markdown(
            r"""
- **$\text{k-dist}(o)$:** distance from $o$ to its $k$-th neighbor.
- **$\text{reach-dist}$:** never smaller than that floor.
- **$\text{lrd}$:** $k$ divided by sum of reach-distances.
"""
        )


def _render_lof_demo() -> None:
    st.markdown(
        r"""
- **Everyday cluster:** tight (\~\$60). **Affluent:** wide (\~\$260).
- Slide **k**; watch `LOCAL_OUT` vs global Z on the chart.
"""
    )
    k_nbrs = st.slider("Neighbors (k)", 3, 15, 6, 1, key="lof_k")
    lof_res = _cached_lof(k_nbrs)
    local_row = lof_res.points[lof_res.points["tx_id"] == "LOCAL_OUT"].iloc[0]
    global_row = lof_res.points[lof_res.points["tx_id"] == "GLOBAL_OUT"].iloc[0]
    dense_row = lof_res.points[lof_res.points["tx_id"] == "E01"].iloc[0]
    sparse_row = lof_res.points[lof_res.points["tx_id"] == "A01"].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("LOCAL_OUT LOF", f"{local_row['lof_score']:.2f}", delta="Flagged (> 1.5)")
    with c2:
        metric("LOCAL_OUT Global Z", f"{local_row['global_z_dist']:.2f}", delta="Missed globally")
    with c3:
        metric("GLOBAL_OUT LOF", f"{global_row['lof_score']:.2f}", delta="Flagged")

    working(
        [
            f"k = {k_nbrs} (LOF = mean lrd(neighbors) / lrd(self)):",
            f"E01 (dense normal):    lrd = {dense_row['lrd']:.2f}  -> LOF = {dense_row['lof_score']:.2f}",
            f"A01 (sparse normal):   lrd = {sparse_row['lrd']:.2f}  -> LOF = {sparse_row['lof_score']:.2f}",
            f"LOCAL_OUT:             lrd = {local_row['lrd']:.2f}  -> LOF = {local_row['lof_score']:.2f}",
        ]
    )
    lof_cluster_chart(lof_res.points)

    st.code(
        """from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lof = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=10, novelty=True))
lof.fit(X_train_normal)
scores = -lof.decision_function(X_eval)  # higher = more anomalous
""",
        language="python",
    )
    lof_lim = pd.DataFrame(
        [
            {"Limit": "Slow O(n^2) neighbor search", "Fix": "Use KD-tree / FAISS or subsample"},
            {"Limit": "Distances blur in high dims", "Fix": "Reduce dims (PCA) before LOF"},
            {"Limit": "Sensitive to k and scaling", "Fix": "Standardize; tune k to cluster size"},
        ]
    )
    dataframe(lof_lim, hide_index=True)


def render_lof_section() -> None:
    tab_fp, tab_math, tab_demo = st.tabs(
        ["First principles", "Math", "Demo & code"]
    )
    with tab_fp:
        _render_lof_first_principles()
    with tab_math:
        _render_lof_math()
    with tab_demo:
        _render_lof_demo()


def _render_graph_first_principles() -> None:
    st.markdown(
        "**First principle:** Tabular models score one row; graphs score who you share links with."
    )
    with st.container(border=True):
        st.markdown(
            "**Pass the paint:** customers, phones, merchants start with base risk; "
            "each hop blends risk along edges."
        )
        st.markdown(
            r"""
- **Hop 0:** row only — blind to links.
- **Hop 1:** phone and merchant neighbors.
- **Hop 2:** other customers on the same device.
"""
        )
    mermaid_chart(
        """
flowchart LR
    subgraph Normal ["Normal (clean 1-to-1 links)"]
        CA["Cust_A (0.10)"] --> D1["Dev_1 (0.08)"] --> MG["Grocer (0.12)"]
    end
    subgraph Ring ["Mule ring (shared shady hub)"]
        CX["Cust_X (0.35)"] --> DS["Dev_Shared (0.85)"]
        CY["Cust_Y (0.32)"] --> DS
        CZ["Cust_Z (0.38)"] --> DS
        DS --> MC["Crypto (0.90)"]
    end
"""
    )
    st.markdown(
        r"""
- **Nodes:** Customer, Device, Merchant.
- **Ring:** three mules share **Dev_Shared** + shady crypto shop.
- **Tabular miss:** each mule row looks calm (`~0.35`) alone.
"""
    )


def _render_graph_math() -> None:
    st.markdown("**Intuition:** Repeat a fair neighbor blend matrix on the risk vector.")
    st.latex(
        r"h^{(0)} = h_{\text{base}}, \qquad h^{(\ell+1)} = \hat{A}\, h^{(\ell)}"
    )
    st.markdown(
        r"""
- **$\hat{A}$:** degree-normalized adjacency with self-loops ($\tilde{D}^{-1/2}(\tilde{A})\tilde{D}^{-1/2}$).
- **$h^{(\ell)}$:** risk score on every node after $\ell$ hops.
- **Stop at $\ell \in [1,3]$:** rings visible before over-smoothing.
"""
    )
    out_nodes, _ = _cached_graph(2)
    cust_rows = out_nodes[out_nodes["entity"] == "Customer"]
    cust_a = cust_rows[cust_rows["id"] == "Cust_A"].iloc[0]
    cust_x = cust_rows[cust_rows["id"] == "Cust_X"].iloc[0]
    working(
        [
            "After l = 2 (cached demo, Â @ h twice):",
            f"Cust_A (Normal): h^(0) = {cust_a['hop_0_base_risk']:.2f} -> h^(2) = {cust_a['final_risk']:.2f}",
            f"Cust_X (Ring):   h^(0) = {cust_x['hop_0_base_risk']:.2f} -> h^(2) = {cust_x['final_risk']:.2f}",
        ]
    )
    with st.expander("Full GNN / GCN matrix form (interviews)"):
        st.latex(
            r"H^{(l+1)} = \sigma\!\Bigl(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}\, H^{(l)}\, W^{(l)}\Bigr), \qquad \tilde{A} = A + I"
        )
        st.markdown(
            r"""
- **$\tilde{A} = A + I$:** links plus self-loop.
- **$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$:** same fair blend as $\hat{A}$.
- **$W^{(l)}$, $\sigma$:** learned weights in trained GNNs.
"""
        )


def _render_graph_demo() -> None:
    hops = st.slider("Pass-the-paint rounds / hops (l)", 0, 3, 2, 1, key="graph_hops")
    out_nodes, edge_lines = _cached_graph(hops)
    cust_rows = out_nodes[out_nodes["entity"] == "Customer"]
    norm_avg = float(cust_rows[cust_rows["group"] == "Normal"]["final_risk"].mean())
    ring_avg = float(cust_rows[cust_rows["group"] == "Ring"]["final_risk"].mean())

    g1, g2, g3 = st.columns(3)
    with g1:
        metric("Normal avg risk", f"{norm_avg:.2f}")
    with g2:
        metric("Ring avg risk", f"{ring_avg:.2f}", delta=f"+{ring_avg - norm_avg:.2f} vs normal")
    with g3:
        metric("Ring / Normal ratio", f"{ring_avg / max(norm_avg, 1e-6):.1f}x")

    cust_a = cust_rows[cust_rows["id"] == "Cust_A"].iloc[0]
    cust_x = cust_rows[cust_rows["id"] == "Cust_X"].iloc[0]
    dev_s = out_nodes[out_nodes["id"] == "Dev_Shared"].iloc[0]
    working(
        [
            f"After l = {hops} (A_norm @ h):",
            f"Cust_A:   {cust_a['hop_0_base_risk']:.2f} -> {cust_a['final_risk']:.2f}",
            f"Cust_X:   {cust_x['hop_0_base_risk']:.2f} -> {cust_x['final_risk']:.2f}",
            f"Dev_Shared hub: {dev_s['hop_0_base_risk']:.2f} -> {dev_s['final_risk']:.2f} ({dev_s['degree']} links)",
        ]
    )
    graph_network_chart(out_nodes, edge_lines)

    summary = cust_rows[
        ["id", "group", "degree", "shared_entity_links", "hop_0_base_risk", "final_risk"]
    ].rename(
        columns={
            "id": "Customer",
            "group": "Group",
            "degree": "Links",
            "shared_entity_links": "Shared-device degree",
            "hop_0_base_risk": "Base risk (Hop 0)",
            "final_risk": f"Final risk (Hop {hops})",
        }
    )
    dataframe(summary, hide_index=True)

    st.code(
        """import numpy as np

A_tilde = A + np.eye(len(A))
d_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(1)))
A_norm = d_inv_sqrt @ A_tilde @ d_inv_sqrt

h = base_risk.copy()
for _ in range(hops):
    h = A_norm @ h
""",
        language="python",
    )
    graph_lim = pd.DataFrame(
        [
            {"Limit": "Over-smoothing (> 3 hops)", "Fix": "Keep 1-3 hops so colors don't blur together"},
            {"Limit": "Giant hubs dilute risk", "Fix": "Degree-normalize or use attention (GAT)"},
            {"Limit": "Real-time scoring latency", "Fix": "Precompute embeddings; query 1-hop ego"},
        ]
    )
    dataframe(graph_lim, hide_index=True)


def render_graph_section() -> None:
    tab_fp, tab_math, tab_demo = st.tabs(
        ["First principles", "Math", "Demo & code"]
    )
    with tab_fp:
        _render_graph_first_principles()
    with tab_math:
        _render_graph_math()
    with tab_demo:
        _render_graph_demo()


def _render_temporal_first_principles() -> None:
    st.markdown(
        r"**Failure mode:** Each $90 buy alone passes amount checks; the burst still looks like fraud."
    )
    with st.container(border=True):
        st.markdown("One cookie at noon = fine. Five cookies in 10 minutes at 2 AM = flag.")
        st.markdown(
            r"""
- **Point model:** one transaction at a time.
- **Sequence model:** remembers order, hour, and time gaps.
- **Each step:** feeds $(\text{amount}, \text{hour}, \text{gap})$ into memory.
"""
        )


def _render_temporal_math() -> None:
    st.markdown("**Intuition:** Normalize features, update hidden state, score rolling memory drift.")
    st.latex(
        r"x_t = [z_{\text{amt}}, z_{\text{hour}}, z_{\text{gap}}], \quad "
        r"h_t = \tanh(0.65\, h_{t-1} + 0.45\, x_t)"
    )
    st.latex(
        r"\text{SeqScore}_t = \text{mean}_{j \in \text{window}} \|h_j\|_2"
    )
    st.markdown(
        r"""
- **$z$ features:** baseline z-scores from first 15 normal steps.
- **Window:** slider length for mean $\|h_j\|$.
- **Flag:** $\text{SeqScore} > 1.05$ vs amount $|z| > 2.5$.
"""
    )
    arch_df = pd.DataFrame(
        [
            {"Model": "RNN", "Production use": "Simple rolling memory", "Trade-off": "Forgets long gaps"},
            {"Model": "LSTM / GRU", "Production use": "Gated bursts", "Trade-off": "Sequential training"},
            {"Model": "Transformer", "Production use": "Full-history attention", "Trade-off": "GPU + latency"},
        ]
    )
    st.caption("Heavier architectures replace the demo tanh cell in production.")
    dataframe(arch_df, hide_index=True)
    with st.expander("LSTM gate sketch (interviews only)"):
        st.latex(
            r"f_t = \sigma(W_f [h_{t-1}, x_t]), \quad i_t = \sigma(W_i [h_{t-1}, x_t]), \quad "
            r"c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t])"
        )


def _render_temporal_demo() -> None:
    win = st.slider("Memory window (recent transactions)", 2, 8, 4, 1, key="temp_win")
    seq_res = _cached_temporal(win)
    burst_rows = seq_res[seq_res["is_seq_anomaly"] == 1]
    norm_sample = seq_res[seq_res["step"] == 10].iloc[0]
    burst_sample = seq_res[seq_res["step"] == 24].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("Amount Z caught", f"{int(burst_rows['point_flag'].sum())} / {len(burst_rows)}", delta="Missed $90 buys")
    with c2:
        metric("Sequence caught", f"{int(burst_rows['sequence_flag'].sum())} / {len(burst_rows)}", delta="Rhythm break")
    with c3:
        metric("Burst gap", f"{burst_rows['gap_minutes'].mean():.1f} min", delta="vs ~420 min normal")

    working(
        [
            f"window = {win} (SeqScore = mean ||h_j|| over recent steps):",
            f"Step 10: Amt Z = {norm_sample['point_amount_z']:.2f}, ||h|| = {norm_sample['hidden_norm']:.2f}, Seq = {norm_sample['sequence_score']:.2f}",
            f"Step 24: Amt Z = {burst_sample['point_amount_z']:.2f} (miss), ||h|| = {burst_sample['hidden_norm']:.2f}, Seq = {burst_sample['sequence_score']:.2f} (flag)",
        ]
    )
    temporal_sequence_chart(seq_res)

    st.code(
        """# Demo recurrence (see run_temporal_sequence_demo in src/anomaly_lab/supporting.py)
for t, x_t in enumerate(x_seq):
    h[t] = np.tanh(0.65 * h[t-1] + 0.45 * x_t)
seq_score[t] = np.mean(np.linalg.norm(h[lo:t+1], axis=1))
""",
        language="python",
    )
    temp_lim = pd.DataFrame(
        [
            {"Limit": "Cold-start new users (no history)", "Fix": "Fall back to global tabular / peer features"},
            {"Limit": "Heavy training & serving cost", "Fix": "Use rolling velocity features + GBDT/IForest first"},
            {"Limit": "Irregular time gaps", "Fix": "Feed log(gap_minutes) and hour sin/cos explicitly"},
        ]
    )
    dataframe(temp_lim, hide_index=True)


def render_temporal_section() -> None:
    tab_fp, tab_math, tab_demo = st.tabs(
        ["First principles", "Math", "Demo & code"]
    )
    with tab_fp:
        _render_temporal_first_principles()
    with tab_math:
        _render_temporal_math()
    with tab_demo:
        _render_temporal_demo()


def render() -> None:
    lesson_header(
        "8. Supporting: LOF, Graph & Temporal",
        "Three simple tricks for fraud that single-row global models miss: local crowds, shared links, and rapid bursts.",
        priority_badge=":blue-badge[Medium priority]",
        interview_focus=[
            r"**LOF:** compare how lonely you are vs your own $k$ neighbors ($\text{LOF} \gg 1$ = outlier).",
            "**Graph / GNN:** pass risk across shared phones and merchants like 'pass the paint'.",
            "**Temporal:** keep a running memory of order, hour, and time gaps to spot bursts.",
        ],
    )

    tab_lof, tab_graph, tab_temp, tab_qa = st.tabs(
        ["1. LOF (local crowds)", "2. Graph / GNN (links)", "3. Temporal (bursts)", "4. Interview prep"]
    )

    with tab_lof:
        render_lof_section()
    with tab_graph:
        render_graph_section()
    with tab_temp:
        render_temporal_section()
    with tab_qa:
        render_interview_qa(CORE_QA_BY_TOPIC["Supporting (LOF, Graph, Temporal)"], key_prefix="qa_supporting")