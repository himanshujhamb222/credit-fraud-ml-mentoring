"""EDA for independent (X) and dependent (y) variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from credit_fraud import config
from credit_fraud.io_utils import ensure_dirs, save_figure, save_json


@dataclass
class EDAResult:
    target_summary: pd.DataFrame
    describe_x: pd.DataFrame
    missingness: pd.DataFrame
    by_class_means: pd.DataFrame
    figures: dict[str, Any]
    imbalance_ratio: float


def target_summary(y: pd.Series) -> tuple[pd.DataFrame, float]:
    """Dependent-variable summary: counts, rates, imbalance ratio (neg/pos)."""
    counts = y.value_counts().sort_index()
    rates = y.value_counts(normalize=True).sort_index()
    summary = pd.DataFrame({"count": counts, "rate": rates})
    summary.index.name = config.TARGET_COL
    pos = int((y == config.POSITIVE_LABEL).sum())
    neg = int((y != config.POSITIVE_LABEL).sum())
    imbalance = (neg / pos) if pos else float("inf")
    return summary.reset_index(), float(imbalance)


def describe_features(x: pd.DataFrame) -> pd.DataFrame:
    """Independent-variable numeric describe."""
    return x.describe().T.reset_index().rename(columns={"index": "feature"})


def missingness_table(x: pd.DataFrame) -> pd.DataFrame:
    miss = x.isna().sum()
    return (
        pd.DataFrame({"feature": miss.index, "n_missing": miss.values, "pct_missing": miss.values / len(x)})
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )


def means_by_class(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Feature means (and std) split by fraud / non-fraud — teaching table."""
    frame = x.copy()
    frame["_target"] = y.values
    grouped = frame.groupby("_target")
    means = grouped.mean().T
    stds = grouped.std().T
    means.columns = [f"mean_class_{c}" for c in means.columns]
    stds.columns = [f"std_class_{c}" for c in stds.columns]
    out = means.join(stds)
    out = out.reset_index().rename(columns={"index": "feature"})
    return out


def plot_target_distribution(y: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    order = sorted(y.unique())
    sns.countplot(x=y.astype(str), order=[str(o) for o in order], ax=ax, color="#2F6F8F")
    ax.set_title("Target distribution (class imbalance)")
    ax.set_xlabel(config.TARGET_COL)
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_feature_hist_by_class(
    x: pd.DataFrame,
    y: pd.Series,
    feature: str,
    bins: int = 40,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, color in [(0, "#5B8FA8"), (1, "#C45C26")]:
        sns.histplot(
            x.loc[y == label, feature],
            bins=bins,
            stat="density",
            element="step",
            fill=True,
            alpha=0.35,
            color=color,
            label=f"class={label}",
            ax=ax,
        )
    ax.set_title(f"{feature} by class")
    ax.legend()
    fig.tight_layout()
    return fig


def run_eda(
    x: pd.DataFrame,
    y: pd.Series,
    save: bool = True,
    example_feature: str | None = None,
) -> EDAResult:
    """Run dependent + independent EDA; optionally persist summary + figures."""
    ensure_dirs()
    tgt, imbalance = target_summary(y)
    desc = describe_features(x)
    miss = missingness_table(x)
    by_class = means_by_class(x, y)

    figures: dict[str, Any] = {}
    fig_target = plot_target_distribution(y)
    figures["target_distribution"] = fig_target

    feat = example_feature or ("Amount" if "Amount" in x.columns else x.columns[0])
    fig_feat = plot_feature_hist_by_class(x, y, feat)
    figures[f"hist_{feat}"] = fig_feat

    if save:
        save_figure(fig_target, "target_distribution.png")
        save_figure(fig_feat, f"hist_{feat}.png")
        save_json(
            {
                "n_rows": int(len(y)),
                "n_features": int(x.shape[1]),
                "imbalance_ratio_neg_over_pos": imbalance,
                "positive_rate": float((y == config.POSITIVE_LABEL).mean()),
                "target_counts": tgt.set_index(config.TARGET_COL)["count"].to_dict(),
            },
            config.EDA_SUMMARY_PATH,
        )

    return EDAResult(
        target_summary=tgt,
        describe_x=desc,
        missingness=miss,
        by_class_means=by_class,
        figures=figures,
        imbalance_ratio=imbalance,
    )
