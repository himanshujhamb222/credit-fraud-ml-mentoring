"""Univariate feature ranking and plots (separation vs target)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

from credit_fraud import config
from credit_fraud.io_utils import ensure_dirs, save_dataframe, save_figure


@dataclass
class RankPlotResult:
    ranking: pd.DataFrame
    figures: dict[str, Any]


def univariate_auc(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """ROC-AUC of each feature alone (flip if AUC < 0.5)."""
    scores = {}
    for col in x.columns:
        values = x[col].to_numpy(dtype=float)
        if np.isnan(values).all() or np.nanstd(values) == 0:
            scores[col] = 0.5
            continue
        # simple median impute for ranking only
        filled = np.where(np.isnan(values), np.nanmedian(values), values)
        try:
            auc = float(roc_auc_score(y, filled))
            scores[col] = max(auc, 1.0 - auc)
        except ValueError:
            scores[col] = 0.5
    return pd.Series(scores, name="univariate_auc").sort_values(ascending=False)


def mutual_info_scores(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    filled = x.fillna(x.median(numeric_only=True))
    mi = mutual_info_classif(filled, y, random_state=config.RANDOM_STATE)
    return pd.Series(mi, index=x.columns, name="mutual_info").sort_values(ascending=False)


def rank_features(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    auc = univariate_auc(x, y)
    mi = mutual_info_scores(x, y)
    ranking = (
        pd.DataFrame({"feature": auc.index, "univariate_auc": auc.values})
        .merge(mi.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")
    )
    ranking = ranking.sort_values("univariate_auc", ascending=False).reset_index(drop=True)
    ranking["rank_auc"] = np.arange(1, len(ranking) + 1)
    return ranking


def plot_feature_ranks(ranking: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    top = ranking.head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=top, y="feature", x="univariate_auc", color="#2F6F8F", ax=ax)
    ax.set_xlim(0.5, 1.0)
    ax.set_title(f"Top {top_n} features by univariate AUC")
    ax.set_xlabel("univariate ROC-AUC (flipped if needed)")
    fig.tight_layout()
    return fig


def plot_top_feature_by_class(
    x: pd.DataFrame,
    y: pd.Series,
    ranking: pd.DataFrame,
    top_n: int = 3,
) -> dict[str, plt.Figure]:
    figs: dict[str, plt.Figure] = {}
    for feature in ranking["feature"].head(top_n):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        plot_df = pd.DataFrame({feature: x[feature], config.TARGET_COL: y.astype(str)})
        sns.boxplot(data=plot_df, x=config.TARGET_COL, y=feature, ax=ax, color="#5B8FA8")
        ax.set_title(f"{feature} by class")
        fig.tight_layout()
        figs[feature] = fig
    return figs


def rank_and_plot(
    x: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    save: bool = True,
) -> RankPlotResult:
    ensure_dirs()
    ranking = rank_features(x, y)
    figures: dict[str, Any] = {"rank_bar": plot_feature_ranks(ranking, top_n=top_n)}
    figures.update(plot_top_feature_by_class(x, y, ranking, top_n=min(3, top_n)))

    if save:
        save_dataframe(ranking, config.RANKING_PATH)
        save_figure(figures["rank_bar"], "feature_rank_auc.png")
        for name, fig in figures.items():
            if name == "rank_bar":
                continue
            save_figure(fig, f"box_{name}.png")

    return RankPlotResult(ranking=ranking, figures=figures)
