"""First-principles statistical anomaly detection: Z-score, IQR, and Mahalanobis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass(frozen=True)
class ZScoreResult:
    mean: float
    std: float
    threshold: float
    table: pd.DataFrame


@dataclass(frozen=True)
class IQRResult:
    q1: float
    median: float
    q3: float
    iqr: float
    k: float
    lower_fence: float
    upper_fence: float
    table: pd.DataFrame


@dataclass(frozen=True)
class MahalanobisResult:
    mean: np.ndarray
    cov: np.ndarray
    corr: np.ndarray
    inv_cov: np.ndarray
    chi2_threshold: float
    alpha: float
    table: pd.DataFrame


def zscore_analysis(
    values: np.ndarray | list[float],
    *,
    labels: list[str] | None = None,
    threshold: float = 2.5,
    ddof: int = 1,
) -> ZScoreResult:
    """Compute standard Z-score, Modified (MAD) Z-score, and log1p Z-score."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=ddof))
    z = (arr - mean) / std if std > 1e-12 else np.zeros_like(arr)

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    mod_z = 0.6745 * (arr - median) / mad if mad > 1e-12 else np.zeros_like(arr)

    log_arr = np.log1p(np.maximum(arr, 0.0))
    log_std = float(np.std(log_arr, ddof=ddof))
    log_z = (log_arr - float(np.mean(log_arr))) / log_std if log_std > 1e-12 else np.zeros_like(arr)

    ids = labels if labels is not None else [f"T{i + 1:02d}" for i in range(len(arr))]
    table = pd.DataFrame(
        {
            "tx_id": ids,
            "amount": np.round(arr, 2),
            "deviation": np.round(arr - mean, 2),
            "z_score": np.round(z, 3),
            "abs_z": np.round(np.abs(z), 3),
            "is_anomaly": np.abs(z) > threshold,
            "modified_z": np.round(mod_z, 3),
            "log1p_z": np.round(log_z, 3),
        }
    )
    return ZScoreResult(mean=mean, std=std, threshold=threshold, table=table)


def outlier_masking_demo(
    base_amounts: list[float],
    extreme_amount: float = 5000.0,
    threshold: float = 2.5,
) -> tuple[ZScoreResult, ZScoreResult]:
    """Show how a single extreme fraud inflates mean & std and masks a moderate outlier."""
    before = zscore_analysis(base_amounts, threshold=threshold)
    after = zscore_analysis([*base_amounts, extreme_amount], threshold=threshold)
    return before, after


def iqr_analysis(
    values: np.ndarray | list[float],
    *,
    labels: list[str] | None = None,
    k: float = 1.5,
) -> IQRResult:
    """Compute quartiles, IQR fences, and anomaly flags."""
    arr = np.asarray(values, dtype=float)
    q1, median, q3 = [float(v) for v in np.percentile(arr, [25, 50, 75])]
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    flags = (arr < lower) | (arr > upper)
    # Continuous score: how many IQRs beyond the nearest fence (0 if inside)
    excess = np.where(arr > upper, (arr - upper) / max(iqr, 1e-9), np.where(arr < lower, (lower - arr) / max(iqr, 1e-9), 0.0))

    ids = labels if labels is not None else [f"T{i + 1:02d}" for i in range(len(arr))]
    table = pd.DataFrame(
        {
            "tx_id": ids,
            "amount": np.round(arr, 2),
            "iqr_excess": np.round(excess, 3),
            "is_anomaly": flags,
            "status": np.where(flags, "Anomaly", "Normal"),
        }
    )
    return IQRResult(
        q1=q1,
        median=median,
        q3=q3,
        iqr=iqr,
        k=k,
        lower_fence=lower,
        upper_fence=upper,
        table=table,
    )


def compare_zscore_vs_iqr(
    values: np.ndarray | list[float],
    *,
    labels: list[str] | None = None,
    z_threshold: float = 2.5,
    k: float = 1.5,
) -> pd.DataFrame:
    """Side-by-side comparison of Z-score vs IQR on the same transactions."""
    z_res = zscore_analysis(values, labels=labels, threshold=z_threshold)
    i_res = iqr_analysis(values, labels=labels, k=k)
    return pd.DataFrame(
        {
            "tx_id": z_res.table["tx_id"],
            "amount": z_res.table["amount"],
            "z_score": z_res.table["z_score"],
            "z_flag": z_res.table["is_anomaly"],
            "iqr_excess": i_res.table["iqr_excess"],
            "iqr_flag": i_res.table["is_anomaly"],
        }
    )


def mahalanobis_analysis(
    X: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    labels: list[str] | None = None,
    alpha: float = 0.975,
    reg: float = 1e-6,
    fit_mask: np.ndarray | None = None,
) -> MahalanobisResult:
    """Compute Mahalanobis vs standardized Euclidean distance and chi-squared threshold."""
    arr = np.asarray(X, dtype=float)
    ref = arr[fit_mask] if fit_mask is not None else arr
    n_features = arr.shape[1]

    mean = np.mean(ref, axis=0)
    cov = np.cov(ref, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov_reg = cov + np.eye(n_features) * reg
    inv_cov = np.linalg.inv(cov_reg)
    corr = np.corrcoef(ref, rowvar=False)
    if corr.ndim == 0:
        corr = np.array([[1.0]])

    diff = arr - mean
    d_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    d_m = np.sqrt(np.maximum(d_sq, 0.0))

    std = np.maximum(np.std(ref, axis=0, ddof=1), 1e-9)
    d_euclid = np.linalg.norm(diff / std, axis=1)

    chi2_thresh = float(np.sqrt(chi2.ppf(alpha, df=n_features)))
    ids = labels if labels is not None else [f"M{i + 1:02d}" for i in range(len(arr))]
    cols = feature_names if feature_names is not None else [f"f{j + 1}" for j in range(n_features)]

    data: dict[str, object] = {"tx_id": ids}
    for j, name in enumerate(cols):
        data[name] = np.round(arr[:, j], 2)
    data["euclidean_z_dist"] = np.round(d_euclid, 3)
    data["mahalanobis_dist"] = np.round(d_m, 3)
    data["is_anomaly"] = d_m > chi2_thresh
    data["status"] = np.where(d_m > chi2_thresh, "Anomaly", "Normal")

    return MahalanobisResult(
        mean=mean,
        cov=cov,
        corr=corr,
        inv_cov=inv_cov,
        chi2_threshold=chi2_thresh,
        alpha=alpha,
        table=pd.DataFrame(data),
    )


def correlated_2d_demo(
    rho: float = 0.88,
    n_normal: int = 70,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, MahalanobisResult]:
    """Generate correlated 2D transactions + Euclidean circle & Mahalanobis ellipse contours."""
    rng = np.random.default_rng(seed)
    mean = np.array([380.0, 12.5])
    stds = np.array([140.0, 4.5])
    cov = np.array(
        [
            [stds[0] ** 2, rho * stds[0] * stds[1]],
            [rho * stds[0] * stds[1], stds[1] ** 2],
        ]
    )
    normal_pts = rng.multivariate_normal(mean, cov, size=n_normal)
    # Candidate A: along correlation axis (looks far in Euclidean, normal in Mahalanobis)
    # Candidate B: breaks correlation (looks close in Euclidean, anomalous in Mahalanobis)
    cand_along = np.array([[mean[0] + 1.8 * stds[0], mean[1] + 1.8 * stds[1]]])
    cand_break = np.array([[mean[0] + 1.45 * stds[0], mean[1] - 1.35 * stds[1]]])

    all_pts = np.vstack([normal_pts, cand_along, cand_break])
    fit_mask = np.zeros(len(all_pts), dtype=bool)
    fit_mask[:n_normal] = True
    labels = [f"N{i + 1:02d}" for i in range(n_normal)] + ["Along-trend (Legit)", "Off-trend (Fraud)"]

    res = mahalanobis_analysis(
        all_pts,
        feature_names=["amount", "distance_km"],
        labels=labels,
        alpha=0.975,
        fit_mask=fit_mask,
    )
    res.table["point_role"] = ["Normal history"] * n_normal + ["Along-trend (Legit)", "Off-trend (Fraud)"]

    # Compute 2D ellipse (Mahalanobis) and circle-in-standardized-space (Euclidean) contours
    theta = np.linspace(0, 2 * np.pi, 140)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])

    eigvals, eigvecs = np.linalg.eigh(res.cov)
    eigvals = np.maximum(eigvals, 1e-9)
    mahal_xy = res.mean[:, None] + eigvecs @ np.diag(np.sqrt(eigvals) * res.chi2_threshold) @ unit_circle

    emp_std = np.std(normal_pts, axis=0, ddof=1)
    euclid_xy = res.mean[:, None] + (emp_std[:, None] * res.chi2_threshold) * unit_circle

    contours = pd.concat(
        [
            pd.DataFrame(
                {
                    "amount": mahal_xy[0],
                    "distance_km": mahal_xy[1],
                    "order": np.arange(len(theta)),
                    "boundary": "Mahalanobis ellipse",
                }
            ),
            pd.DataFrame(
                {
                    "amount": euclid_xy[0],
                    "distance_km": euclid_xy[1],
                    "order": np.arange(len(theta)),
                    "boundary": "Euclidean (ignores correlation)",
                }
            ),
        ],
        ignore_index=True,
    )
    return res.table, contours, res
