"""Unified 6-algorithm evaluation, score normalization, cost analysis, and scenario selector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from anomaly_lab.autoencoder import NumpyAutoencoder
from anomaly_lab.data import BENCHMARK_FEATURES
from anomaly_lab.isolation_forest import fit_sklearn_iforest
from anomaly_lab.one_class_svm import fit_sklearn_ocsvm
from anomaly_lab.statistical import iqr_analysis, mahalanobis_analysis, zscore_analysis

CORE_MODEL_NAMES = [
    "Z-score (amount)",
    "IQR (amount)",
    "Mahalanobis",
    "Isolation Forest",
    "One-Class SVM",
    "Autoencoder",
]


@dataclass(frozen=True)
class BenchmarkComparison:
    scores_df: pd.DataFrame
    metrics_df: pd.DataFrame
    pr_curves: pd.DataFrame
    roc_curves: pd.DataFrame


def normalize_scores(scores: np.ndarray, method: str = "percentile") -> np.ndarray:
    """Place heterogeneous anomaly scores onto a comparable scale."""
    arr = np.asarray(scores, dtype=float)
    if method == "percentile":
        ranks = pd.Series(arr).rank(method="average").to_numpy(dtype=float)
        return (ranks - 1.0) / max(len(arr) - 1, 1)
    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / max(hi - lo, 1e-9)
    # z-standardized
    std = float(arr.std(ddof=1))
    return (arr - float(arr.mean())) / max(std, 1e-9)


def run_six_model_benchmark(
    df: pd.DataFrame,
    *,
    contamination: float = 0.05,
    seed: int = 42,
) -> BenchmarkComparison:
    """Score the shared financial benchmark with all six core algorithms."""
    X = df[BENCHMARK_FEATURES].to_numpy(dtype=float)
    y = df["is_anomaly"].to_numpy(dtype=int)
    amounts = df["amount"].to_numpy(dtype=float)
    normal_mask = y == 0

    # 1. Z-score on amount
    z_scores = zscore_analysis(amounts).table["abs_z"].to_numpy(dtype=float)

    # 2. IQR continuous excess on amount
    iqr_scores = iqr_analysis(amounts).table["iqr_excess"].to_numpy(dtype=float)

    # 3. Mahalanobis on all 5 features (fit on normal reference)
    mahal_scores = mahalanobis_analysis(X, fit_mask=normal_mask).table["mahalanobis_dist"].to_numpy(dtype=float)

    # 4. Isolation Forest (unsupervised on full dataset)
    iforest_scores, _ = fit_sklearn_iforest(X, n_estimators=140, contamination=contamination, random_state=seed)

    # 5. One-Class SVM (trained on normal transactions)
    ocsvm_scores, _ = fit_sklearn_ocsvm(X[normal_mask], X, nu=contamination, gamma="scale", kernel="rbf")

    # 6. Pure-NumPy Autoencoder (trained on normal transactions)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[normal_mask])
    X_all_s = scaler.transform(X)
    ae = NumpyAutoencoder(input_dim=X.shape[1], hidden_dim=10, bottleneck_dim=2, lr=0.08, seed=seed).fit(X_tr_s, epochs=200)
    ae_scores = ae.reconstruction_error(X_all_s)

    raw_map = {
        "Z-score (amount)": z_scores,
        "IQR (amount)": iqr_scores,
        "Mahalanobis": mahal_scores,
        "Isolation Forest": iforest_scores,
        "One-Class SVM": ocsvm_scores,
        "Autoencoder": ae_scores,
    }

    scores_df = df[["tx_id", "customer_id", "amount", "anomaly_type", "is_anomaly"]].copy()
    metric_rows = []
    pr_frames = []
    roc_frames = []

    for name, raw in raw_map.items():
        scores_df[name] = np.round(raw, 4)
        scores_df[f"{name}__pct"] = np.round(normalize_scores(raw, "percentile"), 4)

        cutoff = float(np.quantile(raw, 1.0 - contamination))
        preds = (raw >= cutoff).astype(int)

        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))
        roc_auc = float(roc_auc_score(y, raw))
        pr_auc = float(average_precision_score(y, raw))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))

        metric_rows.append(
            {
                "Algorithm": name,
                "PR-AUC": round(pr_auc, 3),
                "ROC-AUC": round(roc_auc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1": round(f1, 3),
                "False positives": fp,
                "False negatives": fn,
            }
        )

        p_arr, r_arr, _ = precision_recall_curve(y, raw)
        pr_frames.append(pd.DataFrame({"Recall": r_arr, "Precision": p_arr, "Algorithm": name}))

        fpr, tpr, _ = roc_curve(y, raw)
        roc_frames.append(pd.DataFrame({"FPR": fpr, "TPR": tpr, "Algorithm": name}))

    metrics_df = pd.DataFrame(metric_rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return BenchmarkComparison(
        scores_df=scores_df,
        metrics_df=metrics_df,
        pr_curves=pd.concat(pr_frames, ignore_index=True),
        roc_curves=pd.concat(roc_frames, ignore_index=True),
    )


def threshold_cost_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fp_cost: float = 15.0,
    fn_cost: float = 450.0,
) -> pd.DataFrame:
    """Evaluate Precision, Recall, F1, and total financial cost across alert rates."""
    rows = []
    for alert_pct in np.linspace(1.0, 20.0, 20):
        cutoff = float(np.quantile(scores, 1.0 - alert_pct / 100.0))
        preds = (scores >= cutoff).astype(int)
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append(
            {
                "alert_rate_pct": round(float(alert_pct), 1),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1": round(f1, 3),
                "FP": fp,
                "FN": fn,
                "Total cost ($)": round(fp * fp_cost + fn * fn_cost, 1),
            }
        )
    return pd.DataFrame(rows)


def precision_at_k_table(y_true: np.ndarray, scores_map: dict[str, np.ndarray], k: int = 20) -> pd.DataFrame:
    """Simulate scarce-label evaluation: human analysts review only the top-K highest alerts."""
    rows = []
    total_pos = int(np.sum(y_true == 1))
    k_eff = max(1, min(k, len(y_true)))
    for name, scores in scores_map.items():
        top_idx = np.argsort(scores)[::-1][:k_eff]
        hits = int(np.sum(y_true[top_idx] == 1))
        rows.append(
            {
                "Algorithm": name,
                f"Fraud found in top {k_eff}": hits,
                f"Precision@{k_eff}": round(hits / k_eff, 3),
                f"Recall@{k_eff}": round(hits / max(total_pos, 1), 3),
            }
        )
    return pd.DataFrame(rows).sort_values(f"Precision@{k_eff}", ascending=False).reset_index(drop=True)


SCENARIO_SELECTOR_ROWS = [
    {
        "Financial scenario": "Single-metric real-time alert (e.g., per-customer spend spike)",
        "Recommended algorithm": "IQR or Modified Z-score (after log1p)",
        "Why": "Millisecond latency, zero training cost, robust to skewed amounts and extreme past spikes.",
    },
    {
        "Financial scenario": "Correlated tabular signals (amount + distance + credit limit)",
        "Recommended algorithm": "Mahalanobis Distance",
        "Why": "Explicitly uses covariance to catch points that look fine per feature but break joint correlation.",
    },
    {
        "Financial scenario": "Large high-dimensional transaction tables with unknown fraud patterns",
        "Recommended algorithm": "Isolation Forest",
        "Why": "O(t * psi * log(psi)) speed, handles mixed scales without distance collapse, needs no clean labels.",
    },
    {
        "Financial scenario": "Tight nonlinear boundary around clean verified-normal behavior (medium N)",
        "Recommended algorithm": "One-Class SVM (RBF kernel)",
        "Why": "Learns flexible curved support around normal transactions when dataset size is moderate (<20k).",
    },
    {
        "Financial scenario": "Rich multi-feature customer behavior + feature-level explainability",
        "Recommended algorithm": "Autoencoder",
        "Why": "Captures complex nonlinear interactions and attributes anomaly score directly to per-feature MSE.",
    },
    {
        "Financial scenario": "Multi-density customer segments (Everyday vs Affluent clusters)",
        "Recommended algorithm": "Local Outlier Factor (LOF)",
        "Why": "Compares a point's density to its local k-neighbors instead of a single global threshold.",
    },
    {
        "Financial scenario": "Mule rings, synthetic identities, shared devices/IPs/merchants",
        "Recommended algorithm": "Graph-based / GNN",
        "Why": "Individual transactions look benign; fraud only appears through shared entity links.",
    },
    {
        "Financial scenario": "Rapid card-testing bursts or gradual account-takeover drift",
        "Recommended algorithm": "Temporal models (LSTM / Transformer)",
        "Why": "Detects abnormal order, timing gaps, and trajectory shifts even when single amounts look normal.",
    },
]
