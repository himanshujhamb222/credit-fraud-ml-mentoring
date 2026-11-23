"""Transparent pure-NumPy Autoencoder for financial anomaly detection teaching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AutoencoderRun:
    loss_history: pd.DataFrame
    eval_table: pd.DataFrame
    feature_errors: pd.DataFrame
    latent_points: pd.DataFrame
    threshold: float
    normal_mean_mse: float
    anomaly_mean_mse: float
    separation_ratio: float


class NumpyAutoencoder:
    """Compact 3-layer MLP Autoencoder (input -> hidden -> bottleneck -> hidden -> output).

    Uses tanh activations for smooth, stable analytic backpropagation in pure NumPy
    with zero external deep-learning dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        bottleneck_dim: int = 2,
        lr: float = 0.05,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr

        # Xavier/Glorot initialization
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / (input_dim + hidden_dim)), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / (hidden_dim + bottleneck_dim)), (hidden_dim, bottleneck_dim))
        self.b2 = np.zeros(bottleneck_dim)
        self.W3 = rng.normal(0.0, np.sqrt(2.0 / (bottleneck_dim + hidden_dim)), (bottleneck_dim, hidden_dim))
        self.b3 = np.zeros(hidden_dim)
        self.W4 = rng.normal(0.0, np.sqrt(2.0 / (hidden_dim + input_dim)), (hidden_dim, input_dim))
        self.b4 = np.zeros(input_dim)

        self.loss_history_: list[float] = []

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h1 = np.tanh(X @ self.W1 + self.b1)
        z = np.tanh(h1 @ self.W2 + self.b2)
        h3 = np.tanh(z @ self.W3 + self.b3)
        x_hat = h3 @ self.W4 + self.b4  # Linear output for standardized continuous features
        return h1, z, h3, x_hat

    def fit(self, X: np.ndarray, epochs: int = 200) -> NumpyAutoencoder:
        arr = np.asarray(X, dtype=float)
        n, d = arr.shape
        self.loss_history_ = []
        for _ in range(epochs):
            h1, z, h3, x_hat = self._forward(arr)
            err = x_hat - arr
            loss = float(np.mean(err * err))
            self.loss_history_.append(loss)

            # Analytic gradients of MSE = (1 / (n * d)) * sum((x_hat - x)^2)
            d_xhat = (2.0 / (n * d)) * err
            dW4 = h3.T @ d_xhat
            db4 = d_xhat.sum(axis=0)

            d_h3 = (d_xhat @ self.W4.T) * (1.0 - h3 * h3)
            dW3 = z.T @ d_h3
            db3 = d_h3.sum(axis=0)

            d_z = (d_h3 @ self.W3.T) * (1.0 - z * z)
            dW2 = h1.T @ d_z
            db2 = d_z.sum(axis=0)

            d_h1 = (d_z @ self.W2.T) * (1.0 - h1 * h1)
            dW1 = arr.T @ d_h1
            db1 = d_h1.sum(axis=0)

            # Simple gradient clipping for smooth teaching runs across all slider values
            for param, grad in (
                (self.W4, dW4),
                (self.b4, db4),
                (self.W3, dW3),
                (self.b3, db3),
                (self.W2, dW2),
                (self.b2, db2),
                (self.W1, dW1),
                (self.b1, db1),
            ):
                param -= self.lr * np.clip(grad, -2.0, 2.0)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        _, z, _, _ = self._forward(np.asarray(X, dtype=float))
        return z

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        _, _, _, x_hat = self._forward(np.asarray(X, dtype=float))
        return x_hat

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        x_hat = self.reconstruct(arr)
        return np.mean((arr - x_hat) ** 2, axis=1)


def run_autoencoder_demo(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    bottleneck_dim: int = 2,
    epochs: int = 220,
    train_normal_only: bool = True,
    threshold_quantile: float = 0.95,
    seed: int = 42,
) -> AutoencoderRun:
    """Train the NumPy autoencoder on financial transactions and collect diagnostics."""
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["is_anomaly"].to_numpy(dtype=int)

    train_mask = (y_all == 0) if train_normal_only else np.ones(len(df), dtype=bool)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_all[train_mask])
    X_all_s = scaler.transform(X_all)

    hidden_dim = max(8, bottleneck_dim * 2)
    ae = NumpyAutoencoder(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        lr=0.08,
        seed=seed,
    ).fit(X_train_s, epochs=epochs)

    x_hat_s = ae.reconstruct(X_all_s)
    sq_err = (X_all_s - x_hat_s) ** 2
    mse_scores = sq_err.mean(axis=1)
    latent = ae.encode(X_all_s)

    train_scores = mse_scores[train_mask]
    threshold = float(np.quantile(train_scores, threshold_quantile))
    flagged = mse_scores > threshold

    normal_mse = float(mse_scores[y_all == 0].mean())
    anom_mse = float(mse_scores[y_all == 1].mean()) if np.any(y_all == 1) else normal_mse
    ratio = anom_mse / max(normal_mse, 1e-6)

    loss_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(ae.loss_history_) + 1),
            "train_mse": np.round(ae.loss_history_, 5),
        }
    )

    eval_table = df[["tx_id", "customer_id", "customer_tier", *feature_cols, "is_anomaly", "anomaly_type"]].copy()
    eval_table["recon_mse"] = np.round(mse_scores, 4)
    eval_table["flagged"] = flagged
    eval_table["status"] = np.where(y_all == 1, "True anomaly", "Normal")

    # Per-feature average squared error for normal vs anomalies (great for explainability!)
    feat_rows = []
    for j, col in enumerate(feature_cols):
        feat_rows.append(
            {
                "feature": col,
                "group": "Normal",
                "mean_sq_error": round(float(sq_err[y_all == 0, j].mean()), 4),
            }
        )
        if np.any(y_all == 1):
            feat_rows.append(
                {
                    "feature": col,
                    "group": "True anomaly",
                    "mean_sq_error": round(float(sq_err[y_all == 1, j].mean()), 4),
                }
            )
    feature_errors = pd.DataFrame(feat_rows)

    z1 = latent[:, 0]
    z2 = latent[:, 1] if latent.shape[1] >= 2 else np.zeros(len(latent))
    latent_points = pd.DataFrame(
        {
            "tx_id": df["tx_id"],
            "z1": np.round(z1, 4),
            "z2": np.round(z2, 4),
            "recon_mse": np.round(mse_scores, 4),
            "status": np.where(y_all == 1, "True anomaly", "Normal"),
        }
    )

    return AutoencoderRun(
        loss_history=loss_df,
        eval_table=eval_table,
        feature_errors=feature_errors,
        latent_points=latent_points,
        threshold=threshold,
        normal_mean_mse=normal_mse,
        anomaly_mean_mse=anom_mse,
        separation_ratio=ratio,
    )


def customer_vs_global_demo(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compare a global autoencoder vs a customer-tier autoencoder on an Everyday customer jump.

    Shows why a $380 daytime transaction looks normal to a global model (because Affluent
    customers routinely spend $380) yet triggers a customer-level model for an Everyday spender.
    """
    normal_df = df[df["is_anomaly"] == 0].copy()
    everyday_df = normal_df[normal_df["customer_tier"] == "Everyday"].copy()

    # Probe transaction: $380 spent by an Everyday customer (looks like an Affluent customer!)
    probe = pd.DataFrame(
        [
            {
                "amount": 380.0,
                "tx_velocity_24h": 4.2,
                "distance_km": 32.0,
                "hour_of_day": 15.0,
                "merchant_risk": 0.28,
            }
        ]
    )

    results = []
    for scope, train_df in [("Global model (all customers)", normal_df), ("Customer-level model (Everyday tier)", everyday_df)]:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
        X_pr = scaler.transform(probe[feature_cols].to_numpy(dtype=float))
        ae = NumpyAutoencoder(input_dim=len(feature_cols), hidden_dim=8, bottleneck_dim=2, lr=0.08, seed=42).fit(X_tr, epochs=220)
        tr_err = ae.reconstruction_error(X_tr)
        pr_err = float(ae.reconstruction_error(X_pr)[0])
        thresh = float(np.quantile(tr_err, 0.95))
        results.append(
            {
                "Model scope": scope,
                "95th-pct normal MSE": round(thresh, 3),
                "Probe tx ($380) MSE": round(pr_err, 3),
                "Ratio (probe / threshold)": round(pr_err / max(thresh, 1e-6), 2),
                "Decision": "Flagged anomaly" if pr_err > thresh else "Missed (looks normal)",
            }
        )
    return pd.DataFrame(results)
