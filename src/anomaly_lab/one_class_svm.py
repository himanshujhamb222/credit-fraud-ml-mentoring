"""One-Class SVM interactive boundary, support-vector, and RBF kernel explorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass(frozen=True)
class OneClassSVMResult:
    points: pd.DataFrame
    grid: pd.DataFrame
    n_support_vectors: int
    sv_fraction: float
    flagged_fraction: float
    nu: float
    gamma: float | str
    kernel: str


@dataclass(frozen=True)
class Spotlight1DResult:
    curve: pd.DataFrame
    probes: pd.DataFrame
    sv_centers: tuple[float, float]
    scale: float
    gamma: float
    rho: float


def build_1d_spotlight_demo(gamma: float = 1.5, rho: float = 0.45) -> Spotlight1DResult:
    """First-principles 1D RBF spotlight demo with two normal spending modes ($50 and $150)."""
    sv_centers = (50.0, 150.0)
    scale = 50.0  # standardize dollars by $50 so gamma behaves cleanly in [0.1, 5.0]

    xs = np.linspace(0.0, 270.0, 136)
    d1 = (xs - sv_centers[0]) / scale
    d2 = (xs - sv_centers[1]) / scale
    k1 = np.exp(-gamma * (d1**2))
    k2 = np.exp(-gamma * (d2**2))
    warmth = k1 + k2
    score = warmth - rho

    curve = pd.DataFrame(
        {
            "amount": np.round(xs, 1),
            "spotlight_50": np.round(k1, 4),
            "spotlight_150": np.round(k2, 4),
            "warmth": np.round(warmth, 4),
            "decision": np.round(score, 4),
            "zone": np.where(score >= 0.0, "Inside fence (Normal)", "Outside fence (Anomaly)"),
        }
    )

    probe_specs = [
        ("$42 (Normal)", 42.0, "Normal"),
        ("$52 (Normal)", 52.0, "Normal"),
        ("$100 (Gap)", 100.0, "Gap test"),
        ("$152 (Normal)", 152.0, "Normal"),
        ("$240 (Fraud)", 240.0, "Fraud"),
    ]
    rows: list[dict[str, object]] = []
    for label, amt, kind in probe_specs:
        p1 = float(np.exp(-gamma * (((amt - sv_centers[0]) / scale) ** 2)))
        p2 = float(np.exp(-gamma * (((amt - sv_centers[1]) / scale) ** 2)))
        w = p1 + p2
        f = w - rho
        rows.append(
            {
                "label": label,
                "amount": amt,
                "kind": kind,
                "k_50": round(p1, 3),
                "k_150": round(p2, 3),
                "warmth": round(w, 3),
                "decision": round(f, 3),
                "verdict": "Normal (Inside)" if f >= 0.0 else "Anomaly (Outside)",
            }
        )

    return Spotlight1DResult(
        curve=curve,
        probes=pd.DataFrame(rows),
        sv_centers=sv_centers,
        scale=scale,
        gamma=gamma,
        rho=rho,
    )


def rbf_similarity(x: np.ndarray, support_vectors: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel K(x, x_i) = exp(-gamma * ||x - x_i||^2)."""
    diff = support_vectors - x[None, :]
    sq_dist = np.sum(diff * diff, axis=1)
    return np.exp(-gamma * sq_dist)


def run_2d_ocsvm_demo(
    *,
    kernel: str = "rbf",
    nu: float = 0.08,
    gamma: float = 0.5,
    n_normal: int = 80,
    seed: int = 42,
) -> OneClassSVMResult:
    """Fit One-Class SVM on a curved 2D normal transaction manifold + injected anomalies."""
    rng = np.random.default_rng(seed)

    # Normal customers form two gentle spending modes (daytime local + evening dining)
    n1 = n_normal // 2
    n2 = n_normal - n1
    c1 = np.column_stack([rng.normal(65.0, 14.0, n1), rng.normal(2.2, 0.55, n1)])
    c2 = np.column_stack([rng.normal(150.0, 20.0, n2), rng.normal(4.0, 0.65, n2)])
    normal_X = np.vstack([c1, c2])

    anomalies_X = np.array(
        [
            [245.0, 1.4],
            [260.0, 6.8],
            [45.0, 7.2],
            [115.0, 6.9],
            [18.0, 5.4],
        ]
    )

    all_X = np.vstack([normal_X, anomalies_X])
    true_roles = ["Normal"] * len(normal_X) + ["Anomaly"] * len(anomalies_X)

    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_X)
    all_scaled = scaler.transform(all_X)

    gamma_arg: float | str = gamma if kernel in ("rbf", "poly", "sigmoid") else "scale"
    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma_arg)
    model.fit(normal_scaled)

    decision = model.decision_function(all_scaled)
    preds = model.predict(all_scaled)
    sv_indices = set(model.support_.tolist())

    # Build a crisp 2D grid for Altair heatmap visualization
    x_min, x_max = float(all_X[:, 0].min() - 15), float(all_X[:, 0].max() + 15)
    y_min, y_max = float(all_X[:, 1].min() - 0.8), float(all_X[:, 1].max() + 0.8)
    gx = np.linspace(x_min, x_max, 36)
    gy = np.linspace(y_min, y_max, 36)
    xx, yy = np.meshgrid(gx, gy)
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    mesh_decision = model.decision_function(scaler.transform(mesh))

    grid = pd.DataFrame(
        {
            "amount": np.round(mesh[:, 0], 1),
            "tx_velocity_24h": np.round(mesh[:, 1], 2),
            "decision": np.round(mesh_decision, 4),
            "region": np.where(mesh_decision >= 0, "Normal region (f >= 0)", "Anomaly region (f < 0)"),
        }
    )

    role_tags: list[str] = []
    for i, base_role in enumerate(true_roles):
        if i < len(normal_X) and i in sv_indices:
            role_tags.append("Support vector")
        elif base_role == "Anomaly":
            role_tags.append("Injected anomaly")
        else:
            role_tags.append("Normal (non-SV)")

    points = pd.DataFrame(
        {
            "tx_id": [f"N{i + 1:02d}" for i in range(len(normal_X))] + [f"A{i + 1:02d}" for i in range(len(anomalies_X))],
            "amount": np.round(all_X[:, 0], 1),
            "tx_velocity_24h": np.round(all_X[:, 1], 2),
            "true_role": true_roles,
            "plot_role": role_tags,
            "is_support_vector": [i < len(normal_X) and i in sv_indices for i in range(len(all_X))],
            "decision_score": np.round(decision, 4),
            "anomaly_score": np.round(-decision, 4),
            "flagged": preds == -1,
        }
    )

    n_sv = len(sv_indices)
    train_flagged = float(np.mean(model.predict(normal_scaled) == -1))

    return OneClassSVMResult(
        points=points,
        grid=grid,
        n_support_vectors=n_sv,
        sv_fraction=n_sv / len(normal_X),
        flagged_fraction=train_flagged,
        nu=nu,
        gamma=gamma_arg,
        kernel=kernel,
    )


def fit_sklearn_ocsvm(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    nu: float = 0.05,
    gamma: str | float = "scale",
    kernel: str = "rbf",
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize, fit One-Class SVM on training rows, and score evaluation rows."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_ev_s = scaler.transform(X_eval)
    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    model.fit(X_tr_s)
    scores = -model.decision_function(X_ev_s)
    flags = model.predict(X_ev_s) == -1
    return scores, flags
