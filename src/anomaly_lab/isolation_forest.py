"""First-principles Isolation Forest with explicit split tracing and path-length scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

EULER_GAMMA = 0.5772156649


def harmonic_number(n: int) -> float:
    """Harmonic number H(n) approximated cleanly via ln(n) + Euler-Mascheroni constant."""
    if n <= 0:
        return 0.0
    if n == 1:
        return 1.0
    return float(np.log(n) + EULER_GAMMA)


def expected_path_length(n: int) -> float:
    """Average path length c(n) of an unsuccessful search in a Binary Search Tree."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return float(2.0 * harmonic_number(n - 1) - (2.0 * (n - 1) / n))


def anomaly_score(avg_path_length: float | np.ndarray, n: int) -> np.ndarray:
    """Isolation Forest score s(x, n) = 2^(-E[h(x)] / c(n)) in (0, 1]."""
    c_n = expected_path_length(n)
    arr = np.asarray(avg_path_length, dtype=float)
    if c_n <= 0.0:
        return np.full_like(arr, 0.5)
    return np.power(2.0, -arr / c_n)


@dataclass
class SplitSegment:
    """A single axis-aligned cut recorded during 2D tree construction for visualization."""

    depth: int
    feature: int
    threshold: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class _Node:
    size: int
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None


class SimpleIsolationTree:
    """Transparent single Isolation Tree recording exact splits and path lengths."""

    def __init__(self, height_limit: int, seed: int = 0) -> None:
        self.height_limit = height_limit
        self.rng = np.random.default_rng(seed)
        self.root: _Node | None = None
        self.splits: list[SplitSegment] = []

    def fit(self, X: np.ndarray) -> SimpleIsolationTree:
        arr = np.asarray(X, dtype=float)
        bounds = np.column_stack([arr.min(axis=0), arr.max(axis=0)]) if arr.shape[1] == 2 else None
        self.splits = []
        self.root = self._build(arr, depth=0, bounds=bounds)
        return self

    def _build(self, X: np.ndarray, depth: int, bounds: np.ndarray | None) -> _Node:
        n_samples, n_features = X.shape
        if depth >= self.height_limit or n_samples <= 1:
            return _Node(size=n_samples)

        valid_features = [j for j in range(n_features) if float(X[:, j].max() - X[:, j].min()) > 1e-12]
        if not valid_features:
            return _Node(size=n_samples)

        feat = int(self.rng.choice(valid_features))
        lo, hi = float(X[:, feat].min()), float(X[:, feat].max())
        threshold = float(self.rng.uniform(lo, hi))

        left_mask = X[:, feat] < threshold
        left_bounds = bounds.copy() if bounds is not None else None
        right_bounds = bounds.copy() if bounds is not None else None
        if bounds is not None:
            if feat == 0:
                self.splits.append(
                    SplitSegment(
                        depth=depth,
                        feature=feat,
                        threshold=threshold,
                        x1=threshold,
                        y1=float(bounds[1, 0]),
                        x2=threshold,
                        y2=float(bounds[1, 1]),
                    )
                )
                left_bounds[0, 1] = threshold
                right_bounds[0, 0] = threshold
            else:
                self.splits.append(
                    SplitSegment(
                        depth=depth,
                        feature=feat,
                        threshold=threshold,
                        x1=float(bounds[0, 0]),
                        y1=threshold,
                        x2=float(bounds[0, 1]),
                        y2=threshold,
                    )
                )
                left_bounds[1, 1] = threshold
                right_bounds[1, 0] = threshold

        return _Node(
            size=n_samples,
            feature=feat,
            threshold=threshold,
            left=self._build(X[left_mask], depth + 1, left_bounds),
            right=self._build(X[~left_mask], depth + 1, right_bounds),
        )

    def path_length(self, x: np.ndarray) -> float:
        """Compute path length h(x) = edges traversed + c(leaf.size)."""
        node = self.root
        edges = 0
        while node is not None and node.feature is not None and node.threshold is not None:
            edges += 1
            node = node.left if x[node.feature] < node.threshold else node.right
        leaf_size = node.size if node is not None else 1
        return float(edges + expected_path_length(leaf_size))


@dataclass(frozen=True)
class IsolationForestDemoResult:
    points: pd.DataFrame
    splits: pd.DataFrame
    c_psi: float
    psi: int
    n_trees: int


def run_2d_isolation_demo(
    *,
    n_normal: int = 55,
    n_trees: int = 40,
    psi: int = 32,
    contamination: float = 0.08,
    tree_seed: int = 7,
    data_seed: int = 42,
) -> IsolationForestDemoResult:
    """Run transparent pure-Python Isolation Forest on 2D financial transactions."""
    rng = np.random.default_rng(data_seed)
    normal_amt = rng.normal(95.0, 22.0, size=n_normal).clip(25.0, 175.0)
    normal_vel = rng.normal(2.5, 0.75, size=n_normal).clip(0.5, 5.2)

    # 4 clear financial anomalies: high spend, velocity burst, and combined takeover
    anom_amt = np.array([340.0, 395.0, 75.0, 310.0])
    anom_vel = np.array([3.2, 8.8, 10.5, 9.4])

    X = np.vstack(
        [
            np.column_stack([normal_amt, normal_vel]),
            np.column_stack([anom_amt, anom_vel]),
        ]
    )
    roles = ["Normal"] * n_normal + ["Anomaly"] * len(anom_amt)
    ids = [f"N{i + 1:02d}" for i in range(n_normal)] + [f"A{i + 1:02d}" for i in range(len(anom_amt))]

    psi_eff = min(psi, len(X))
    height_limit = int(np.ceil(np.log2(max(psi_eff, 2))))
    sample_rng = np.random.default_rng(tree_seed)

    path_matrix = np.zeros((n_trees, len(X)), dtype=float)
    first_tree_splits: list[SplitSegment] = []

    for t in range(n_trees):
        idx = sample_rng.choice(len(X), size=psi_eff, replace=False)
        tree = SimpleIsolationTree(height_limit=height_limit, seed=tree_seed * 100 + t).fit(X[idx])
        if t == 0:
            first_tree_splits = tree.splits
        for i in range(len(X)):
            path_matrix[t, i] = tree.path_length(X[i])

    avg_paths = path_matrix.mean(axis=0)
    scores = anomaly_score(avg_paths, psi_eff)
    cutoff = float(np.quantile(scores, 1.0 - contamination))
    flags = scores >= cutoff

    points = pd.DataFrame(
        {
            "tx_id": ids,
            "amount": np.round(X[:, 0], 1),
            "tx_velocity_24h": np.round(X[:, 1], 2),
            "true_role": roles,
            "avg_path_length": np.round(avg_paths, 3),
            "anomaly_score": np.round(scores, 4),
            "flagged": flags,
            "predicted": np.where(flags, "Flagged anomaly", "Normal"),
        }
    )

    split_rows = []
    for idx_s, seg in enumerate(first_tree_splits):
        split_rows.append(
            {
                "split_id": idx_s,
                "depth": seg.depth,
                "x": seg.x1,
                "y": seg.y1,
                "order": 0,
            }
        )
        split_rows.append(
            {
                "split_id": idx_s,
                "depth": seg.depth,
                "x": seg.x2,
                "y": seg.y2,
                "order": 1,
            }
        )
    splits_df = pd.DataFrame(split_rows)

    return IsolationForestDemoResult(
        points=points,
        splits=splits_df,
        c_psi=expected_path_length(psi_eff),
        psi=psi_eff,
        n_trees=n_trees,
    )


def score_curve_table(psi: int = 256) -> pd.DataFrame:
    """Build E[h(x)] vs s(x, psi) curve so learners can read the 0.5 / 1.0 regimes visually."""
    c_n = expected_path_length(psi)
    hs = np.linspace(0.5, max(c_n * 2.2, 6.0), 80)
    scores = anomaly_score(hs, psi)
    return pd.DataFrame(
        {
            "avg_path_length": np.round(hs, 2),
            "anomaly_score": np.round(scores, 4),
            "c_psi": round(c_n, 3),
        }
    )


def fit_sklearn_iforest(
    X: np.ndarray,
    *,
    n_estimators: int = 150,
    max_samples: int | str = "auto",
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return higher-is-more-anomalous scores and binary anomaly flags from sklearn."""
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
    )
    preds = model.fit_predict(X)
    # sklearn score_samples returns negative anomaly score; negate so larger = more anomalous
    scores = -model.score_samples(X)
    return scores, (preds == -1)
