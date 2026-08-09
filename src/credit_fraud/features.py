"""Leakage-safe feature engineering with fit / transform."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class FeatureBuilder:
    """
    Construct a few transparent features for mentoring.

    - log1p(Amount) when Amount exists
    - Amount quantile bins (fit on train only)
    - Time → hour-of-day style cyclic features when Time exists (seconds)
    - |V1| * log1p(Amount) interaction when both exist
    """

    amount_bins: int = 5
    _binner: KBinsDiscretizer | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> FeatureBuilder:
        _ = y  # unused; kept for sklearn-style API in the session
        if "Amount" in x.columns:
            self._binner = KBinsDiscretizer(
                n_bins=self.amount_bins,
                encode="ordinal",
                strategy="quantile",
            )
            amount = x[["Amount"]].to_numpy(dtype=float)
            # guard constant / tiny samples
            if np.nanstd(amount) > 0 and len(x) >= self.amount_bins:
                self._binner.fit(amount)
            else:
                self._binner = None
        self._fitted = True
        # dry-run to freeze names
        self.feature_names_ = list(self.transform(x).columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out = x.copy()

        if "Amount" in out.columns:
            out["Amount_log1p"] = np.log1p(out["Amount"].clip(lower=0))
            if self._binner is not None:
                out["Amount_bin"] = self._binner.transform(out[["Amount"]].to_numpy(dtype=float)).ravel()
            if "V1" in out.columns:
                out["V1_abs_x_Amount_log1p"] = out["V1"].abs() * out["Amount_log1p"]

        if "Time" in out.columns:
            # Classic creditcard.csv: Time = seconds elapsed
            hours = (out["Time"] % 86_400) / 3_600.0
            out["Time_hour"] = hours
            out["Time_hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            out["Time_hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        # Fill residual NaNs with column medians computed on this frame (post-fit path
        # should already be clean; this keeps transform robust for API rows)
        num_cols = out.select_dtypes(include="number").columns
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())
        return out

    def fit_transform(self, x: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(x, y).transform(x)


def build_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureBuilder]:
    """Fit FeatureBuilder on train, transform train/test."""
    builder = FeatureBuilder()
    x_train_fe = builder.fit_transform(x_train, y_train)
    x_test_fe = builder.transform(x_test)
    return x_train_fe, x_test_fe, builder
