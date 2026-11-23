"""Supporting anomaly detection paradigms: LOF, Graph-based (GNN), and Temporal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class LOFDemoResult:
    points: pd.DataFrame
    k: int


def compute_lof_first_principles(X: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute k-distance, Local Reachability Density (LRD), and LOF score directly."""
    arr = np.asarray(X, dtype=float)
    n = len(arr)
    k_eff = max(2, min(k, n - 1))

    dist = cdist(arr, arr, metric="euclidean")
    np.fill_diagonal(dist, np.inf)

    order = np.argsort(dist, axis=1)
    k_dist = dist[np.arange(n), order[:, k_eff - 1]]

    lrd = np.zeros(n, dtype=float)
    neighbors_list: list[np.ndarray] = []
    for i in range(n):
        nbrs = order[i, :k_eff]
        neighbors_list.append(nbrs)
        reach = np.maximum(k_dist[nbrs], dist[i, nbrs])
        mean_reach = float(np.mean(reach))
        lrd[i] = 1.0 / max(mean_reach, 1e-9)

    lof = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = neighbors_list[i]
        lof[i] = float(np.mean(lrd[nbrs]) / max(lrd[i], 1e-9))

    return k_dist, lrd, lof


def run_lof_dense_vs_sparse_demo(k: int = 6, seed: int = 42) -> LOFDemoResult:
    """Demonstrate why LOF beats global methods when clusters have different densities.

    Cluster A (Everyday local spenders): very tight density.
    Cluster B (Affluent travelers): naturally wide spread.
    Point L_Out sits just slightly outside the tight Everyday cluster — close in global
    distance compared to the Affluent spread, yet clearly isolated relative to its local neighbors!
    """
    rng = np.random.default_rng(seed)
    dense_cluster = np.column_stack([rng.normal(60.0, 5.5, 35), rng.normal(2.0, 0.25, 35)])
    sparse_cluster = np.column_stack([rng.normal(260.0, 42.0, 35), rng.normal(6.0, 1.35, 35)])
    local_outlier = np.array([[104.0, 3.05]])  # slightly outside tight cluster, closer than sparse spread
    global_outlier = np.array([[370.0, 9.6]])

    X = np.vstack([dense_cluster, sparse_cluster, local_outlier, global_outlier])
    # Standardize for fair Euclidean neighborhood distances
    X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)

    k_dist, lrd, lof = compute_lof_first_principles(X_std, k=k)
    # Global distance from nearest cluster centroid or overall mean
    global_z = np.linalg.norm(X_std, axis=1)

    groups = (
        ["Tight cluster (Everyday)"] * len(dense_cluster)
        + ["Wide cluster (Affluent)"] * len(sparse_cluster)
        + ["Local outlier"]
        + ["Global outlier"]
    )
    ids = (
        [f"E{i + 1:02d}" for i in range(len(dense_cluster))]
        + [f"A{i + 1:02d}" for i in range(len(sparse_cluster))]
        + ["LOCAL_OUT", "GLOBAL_OUT"]
    )

    df = pd.DataFrame(
        {
            "tx_id": ids,
            "amount": np.round(X[:, 0], 1),
            "tx_velocity_24h": np.round(X[:, 1], 2),
            "group": groups,
            "k_distance": np.round(k_dist, 3),
            "lrd": np.round(lrd, 3),
            "lof_score": np.round(lof, 3),
            "global_z_dist": np.round(global_z, 3),
            "lof_flag": lof > 1.5,
        }
    )
    return LOFDemoResult(points=df, k=k)


def run_graph_message_passing_demo(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    steps: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute structural graph features + normalized GNN message-passing risk propagation."""
    node_ids = nodes["id"].tolist()
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    A = np.zeros((n, n), dtype=float)
    for _, row in edges.iterrows():
        u, v = idx_map[row["source"]], idx_map[row["target"]]
        A[u, v] = 1.0
        A[v, u] = 1.0

    degrees = A.sum(axis=1)
    # Shared-device indicator: how many customers connect to the same device node(s)
    device_indices = {i for i, ent in enumerate(nodes["entity"]) if ent == "Device"}
    shared_neighbors = np.zeros(n, dtype=int)
    for i in range(n):
        if nodes.iloc[i]["entity"] == "Customer":
            dev_nbrs = [j for j in device_indices if A[i, j] > 0]
            max_shared = max((int(A[j].sum()) for j in dev_nbrs), default=0)
            shared_neighbors[i] = max_shared
        else:
            shared_neighbors[i] = int(degrees[i])

    # Normalized adjacency with self-loops: A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}
    A_tilde = A + np.eye(n)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A_tilde.sum(axis=1), 1e-9)))
    A_norm = d_inv_sqrt @ A_tilde @ d_inv_sqrt

    h = nodes["base_risk"].to_numpy(dtype=float).copy()
    history = {"hop_0_base_risk": np.round(h, 3)}
    for hop in range(1, steps + 1):
        h = A_norm @ h
        history[f"hop_{hop}_gnn_risk"] = np.round(h, 3)

    out_nodes = nodes.copy()
    out_nodes["degree"] = degrees.astype(int)
    out_nodes["shared_entity_links"] = shared_neighbors
    for col, vals in history.items():
        out_nodes[col] = vals
    out_nodes["final_risk"] = np.round(h, 3)

    # Expand edge coordinates for Altair line plotting
    coord_map = {row["id"]: (row["x"], row["y"]) for _, row in nodes.iterrows()}
    edge_rows = []
    for idx_e, row in edges.iterrows():
        x1, y1 = coord_map[row["source"]]
        x2, y2 = coord_map[row["target"]]
        edge_rows.append({"edge_id": idx_e, "relation": row["relation"], "x": x1, "y": y1, "order": 0})
        edge_rows.append({"edge_id": idx_e, "relation": row["relation"], "x": x2, "y": y2, "order": 1})
    edge_lines = pd.DataFrame(edge_rows)
    return out_nodes, edge_lines


def run_temporal_sequence_demo(seq_df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Compare single-transaction amount Z-score vs sliding-window temporal sequence anomaly score.

    Demonstrates why point-wise amount checks miss a burst of ordinary-sized 2 AM transactions
    while a sequence model (tracking rolling gap & hour shifts + recurrent hidden state) catches them.
    """
    out = seq_df.copy()
    amt = out["amount"].to_numpy(dtype=float)
    hour = out["hour_of_day"].to_numpy(dtype=float)
    gap = out["gap_minutes"].to_numpy(dtype=float)

    # Point-wise amount Z-score (misses the sequence anomaly!)
    out["point_amount_z"] = np.round(np.abs((amt - amt.mean()) / max(amt.std(ddof=1), 1e-9)), 3)

    # Normalize behavioral sequence features using the first 15 normal transactions as baseline
    base = slice(0, 15)
    hour_z = (hour - hour[base].mean()) / max(hour[base].std(ddof=1), 1e-9)
    gap_z = (gap - gap[base].mean()) / max(gap[base].std(ddof=1), 1e-9)
    amt_z = (amt - amt[base].mean()) / max(amt[base].std(ddof=1), 1e-9)

    # Lightweight recurrent hidden-state update h_t = tanh(0.65 * h_{t-1} + 0.5 * x_t)
    # captures accumulated sequence drift across consecutive abnormal steps
    h_state = np.zeros((len(out), 3), dtype=float)
    x_seq = np.column_stack([amt_z, hour_z, gap_z])
    for t in range(len(out)):
        prev = h_state[t - 1] if t > 0 else np.zeros(3)
        h_state[t] = np.tanh(0.65 * prev + 0.45 * x_seq[t])

    # Rolling window behavioral anomaly score
    seq_scores = np.zeros(len(out), dtype=float)
    for t in range(len(out)):
        lo = max(0, t - window + 1)
        win_norm = float(np.mean(np.linalg.norm(h_state[lo : t + 1], axis=1)))
        seq_scores[t] = win_norm

    hidden_norm = np.linalg.norm(h_state, axis=1)
    out["hidden_norm"] = np.round(hidden_norm, 3)
    out["sequence_score"] = np.round(seq_scores, 3)
    out["point_flag"] = out["point_amount_z"] > 2.5
    out["sequence_flag"] = out["sequence_score"] > 1.05
    return out
