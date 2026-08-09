"""Bayesian hyperparameter optimization with Optuna (TPE)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score

from credit_fraud import config
from credit_fraud.modeling import (
    classification_metrics,
    make_random_forest,
    make_xgboost,
    predict_proba_positive,
    scale_pos_weight,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _pr_auc_scorer(estimator: Any, x: pd.DataFrame, y: pd.Series) -> float:
    return float(average_precision_score(y, predict_proba_positive(estimator, x)))


@dataclass
class OptimizeResult:
    best_models: dict[str, Any]
    best_params: dict[str, dict[str, Any]]
    cv_scores: dict[str, float]
    test_metrics: dict[str, dict[str, float]]
    studies: dict[str, optuna.Study] = field(default_factory=dict)


def _cv_pr_auc(model: Any, x: pd.DataFrame, y: pd.Series) -> float:
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    scores = cross_val_score(model, x, y, scoring=_pr_auc_scorer, cv=cv, n_jobs=-1)
    return float(np.mean(scores))


def optimize_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any], float, optuna.Study]:
    n_trials = n_trials or config.N_TRIALS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
        model = make_random_forest(**params)
        return _cv_pr_auc(model, x_train, y_train)

    study = optuna.create_study(direction="maximize", study_name="rf_pr_auc")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=config.OPTUNA_TIMEOUT_SEC,
        show_progress_bar=False,
    )
    best_model = make_random_forest(**study.best_params)
    best_model.fit(x_train, y_train)
    return best_model, dict(study.best_params), float(study.best_value), study


def optimize_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any], float, optuna.Study]:
    n_trials = n_trials or config.N_TRIALS
    spw = scale_pos_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": spw,
        }
        model = make_xgboost(y_train=y_train, **params)
        return _cv_pr_auc(model, x_train, y_train)

    study = optuna.create_study(direction="maximize", study_name="xgb_pr_auc")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=config.OPTUNA_TIMEOUT_SEC,
        show_progress_bar=False,
    )
    best_params = dict(study.best_params)
    best_params["scale_pos_weight"] = spw
    best_model = make_xgboost(y_train=y_train, **best_params)
    best_model.fit(x_train, y_train)
    return best_model, best_params, float(study.best_value), study


def optimize_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_trials: int | None = None,
) -> OptimizeResult:
    """Run Optuna for RF and XGBoost; return fitted winners + holdout metrics."""
    print(f"Optuna RF ({n_trials or config.N_TRIALS} trials)...")
    rf_model, rf_params, rf_cv, rf_study = optimize_random_forest(x_train, y_train, n_trials)

    print(f"Optuna XGBoost ({n_trials or config.N_TRIALS} trials)...")
    xgb_model, xgb_params, xgb_cv, xgb_study = optimize_xgboost(x_train, y_train, n_trials)

    best_models = {"random_forest": rf_model, "xgboost": xgb_model}
    best_params = {"random_forest": rf_params, "xgboost": xgb_params}
    cv_scores = {"random_forest": rf_cv, "xgboost": xgb_cv}

    test_metrics: dict[str, dict[str, float]] = {}
    for name, model in best_models.items():
        proba = predict_proba_positive(model, x_test)
        test_metrics[name] = classification_metrics(y_test, proba)

    return OptimizeResult(
        best_models=best_models,
        best_params=best_params,
        cv_scores=cv_scores,
        test_metrics=test_metrics,
        studies={"random_forest": rf_study, "xgboost": xgb_study},
    )
