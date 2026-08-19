"""Leakage-safe feature engineering with fit / transform."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class FeatureBuilder:
    """
    Construct a few transparent, explainable features for mentoring.

    Works with the UCI credit-default columns (LIMIT_BAL, BILL_AMT*, PAY_*, …)
    and still supports the older fraud CSV (Amount / Time) if swapped in.
    """

    amount_bins: int = 5
    _binner: KBinsDiscretizer | None = field(default=None, init=False, repr=False)
    _bin_col: str | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> FeatureBuilder:
        _ = y  # unused; kept for sklearn-style API in the session
        bin_col = "LIMIT_BAL" if "LIMIT_BAL" in x.columns else ("Amount" if "Amount" in x.columns else None)
        self._bin_col = bin_col
        self._binner = None
        if bin_col is not None:
            self._binner = KBinsDiscretizer(
                n_bins=self.amount_bins,
                encode="ordinal",
                strategy="quantile",
            )
            values = x[[bin_col]].to_numpy(dtype=float)
            if np.nanstd(values) > 0 and len(x) >= self.amount_bins:
                self._binner.fit(values)
            else:
                self._binner = None
        self._fitted = True
        self.feature_names_ = list(self.transform(x).columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out = x.copy()

        # --- UCI credit-default features (interpretable) ---
        if "LIMIT_BAL" in out.columns:
            out["LIMIT_BAL_log1p"] = np.log1p(out["LIMIT_BAL"].clip(lower=0))
            if self._binner is not None and self._bin_col == "LIMIT_BAL":
                out["LIMIT_BAL_bin"] = self._binner.transform(
                    out[["LIMIT_BAL"]].to_numpy(dtype=float)
                ).ravel()

        if "BILL_AMT1" in out.columns and "LIMIT_BAL" in out.columns:
            denom = out["LIMIT_BAL"].replace(0, np.nan)
            out["UTIL_RATIO_1"] = (out["BILL_AMT1"] / denom).replace([np.inf, -np.inf], np.nan)

        if "PAY_AMT1" in out.columns and "BILL_AMT1" in out.columns:
            out["PAY_RATIO_1"] = out["PAY_AMT1"] / (out["BILL_AMT1"].abs() + 1.0)

        pay_cols = [c for c in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6") if c in out.columns]
        if pay_cols:
            out["AVG_PAY_STATUS"] = out[pay_cols].mean(axis=1)
            out["MAX_PAY_STATUS"] = out[pay_cols].max(axis=1)

        # --- Legacy fraud CSV support (Amount / Time) ---
        if "Amount" in out.columns:
            out["Amount_log1p"] = np.log1p(out["Amount"].clip(lower=0))
            if self._binner is not None and self._bin_col == "Amount":
                out["Amount_bin"] = self._binner.transform(
                    out[["Amount"]].to_numpy(dtype=float)
                ).ravel()

        if "Time" in out.columns:
            hours = (out["Time"] % 86_400) / 3_600.0
            out["Time_hour"] = hours
            out["Time_hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            out["Time_hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

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
