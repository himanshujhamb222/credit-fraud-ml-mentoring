"""End-to-end mentoring pipeline — call ``run_training()`` with no CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from credit_fraud import config
from credit_fraud.benchmark import run_benchmark
from credit_fraud.data import load_and_split
from credit_fraud.eda import run_eda
from credit_fraud.features import FeatureBuilder, build_features
from credit_fraud.io_utils import ensure_dirs, save_json
from credit_fraud.optimize import optimize_models
from credit_fraud.rank_plot import rank_and_plot
from credit_fraud.selection import select_best


@dataclass
class TrainingResult:
    comparison: pd.DataFrame
    best_name: str
    best_metrics: dict[str, float]
    importances: pd.DataFrame
    ranking: pd.DataFrame
    feature_builder: FeatureBuilder
    extras: dict[str, Any]


def run_training(n_trials: int | None = None) -> TrainingResult:
    """
    Full path used in the mentoring session:

    load → EDA → rank & plot → features → benchmark → Optuna RF/XGB → select & persist
    """
    ensure_dirs()
    print("1) Loading data + stratified split...")
    x_train, x_test, y_train, y_test, raw = load_and_split()
    print(f"   train={len(x_train):,} test={len(x_test):,} features={x_train.shape[1]}")

    print("2) EDA (independent + dependent)...")
    eda = run_eda(x_train, y_train, save=True)
    print(f"   imbalance ratio (neg/pos) = {eda.imbalance_ratio:.2f}")

    print("3) Rank & plot...")
    ranks = rank_and_plot(x_train, y_train, save=True)

    print("4) Feature engineering...")
    x_train_fe, x_test_fe, builder = build_features(x_train, x_test, y_train)
    print(f"   features after FE: {x_train_fe.shape[1]}")

    print("5) Benchmark baselines...")
    bench = run_benchmark(x_train_fe, x_test_fe, y_train, y_test)
    for name, metrics in bench.metrics.items():
        print(f"   {name}: pr_auc={metrics['pr_auc']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    print("6) Modelling + Optuna HPO...")
    opt = optimize_models(x_train_fe, x_test_fe, y_train, y_test, n_trials=n_trials)
    for name, metrics in opt.test_metrics.items():
        print(
            f"   {name}: pr_auc={metrics['pr_auc']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} cv_pr_auc={opt.cv_scores[name]:.4f}"
        )

    print("7) Model selection + persist artifacts...")
    selection = select_best(
        model_metrics=opt.test_metrics,
        models=opt.best_models,
        feature_names=list(x_train_fe.columns),
        benchmark_metrics=bench.metrics,
        cv_scores=opt.cv_scores,
        best_params=opt.best_params,
        save=True,
    )
    # Persist feature builder + column contract for API / Streamlit
    save_json(
        {
            "feature_names": list(x_train_fe.columns),
            "raw_feature_names": list(x_train.columns),
            "target_col": config.TARGET_COL,
            "best_model": selection.best_name,
        },
        config.ARTIFACTS_DIR / "feature_contract.json",
    )
    # joblib the builder next to the model for transform at serve time
    from credit_fraud.io_utils import save_model

    save_model(builder, config.ARTIFACTS_DIR / "feature_builder.joblib")

    print(f"Winner: {selection.best_name} | {config.PRIMARY_METRIC}={selection.best_metrics[config.PRIMARY_METRIC]:.4f}")
    print(f"Artifacts -> {config.ARTIFACTS_DIR}")

    return TrainingResult(
        comparison=selection.comparison,
        best_name=selection.best_name,
        best_metrics=selection.best_metrics,
        importances=selection.importances,
        ranking=ranks.ranking,
        feature_builder=builder,
        extras={
            "eda": eda,
            "benchmark": bench,
            "optimize": opt,
            "raw_shape": raw.shape,
            "x_train": x_train,
            "y_train": y_train,
            "x_train_fe": x_train_fe,
        },
    )


if __name__ == "__main__":
    run_training()
