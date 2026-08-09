"""Compare candidates and persist the winning model + reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from credit_fraud import config
from credit_fraud.io_utils import ensure_dirs, save_dataframe, save_json, save_model
from credit_fraud.modeling import feature_importance_frame


@dataclass
class SelectionResult:
    comparison: pd.DataFrame
    best_name: str
    best_model: Any
    best_metrics: dict[str, float]
    importances: pd.DataFrame


def metrics_table(
    benchmark_metrics: dict[str, dict[str, float]],
    model_metrics: dict[str, dict[str, float]],
    cv_scores: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for name, metrics in {**benchmark_metrics, **model_metrics}.items():
        row = {"model": name, **metrics}
        if cv_scores and name in cv_scores:
            row["cv_pr_auc"] = cv_scores[name]
        rows.append(row)
    frame = pd.DataFrame(rows)
    primary = config.PRIMARY_METRIC
    if primary in frame.columns:
        frame = frame.sort_values(primary, ascending=False).reset_index(drop=True)
    return frame


def select_best(
    model_metrics: dict[str, dict[str, float]],
    models: dict[str, Any],
    feature_names: list[str],
    benchmark_metrics: dict[str, dict[str, float]] | None = None,
    cv_scores: dict[str, float] | None = None,
    best_params: dict[str, dict[str, Any]] | None = None,
    save: bool = True,
) -> SelectionResult:
    """
    Pick the best tree model by ``config.PRIMARY_METRIC`` (default PR-AUC).

    Benchmarks are included in the comparison table but are not eligible winners.
    """
    ensure_dirs()
    benchmark_metrics = benchmark_metrics or {}
    comparison = metrics_table(benchmark_metrics, model_metrics, cv_scores=cv_scores)

    primary = config.PRIMARY_METRIC
    eligible = comparison[comparison["model"].isin(models.keys())]
    if eligible.empty:
        raise ValueError("No eligible models to select from.")
    best_row = eligible.sort_values(primary, ascending=False).iloc[0]
    best_name = str(best_row["model"])
    best_model = models[best_name]
    best_metrics = model_metrics[best_name]
    importances = feature_importance_frame(best_model, feature_names)

    if save:
        save_model(best_model, config.MODEL_PATH)
        save_dataframe(importances, config.IMPORTANCES_PATH)
        payload = {
            "best_model": best_name,
            "primary_metric": primary,
            "best_metrics": best_metrics,
            "cv_scores": cv_scores or {},
            "best_params": (best_params or {}).get(best_name, {}),
            "comparison": comparison.to_dict(orient="records"),
        }
        save_json(payload, config.METRICS_PATH)
        save_dataframe(comparison, config.ARTIFACTS_DIR / "model_comparison.csv")

    return SelectionResult(
        comparison=comparison,
        best_name=best_name,
        best_model=best_model,
        best_metrics=best_metrics,
        importances=importances,
    )
