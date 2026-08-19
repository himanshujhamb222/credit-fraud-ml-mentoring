"""Bayesian hyperparameter optimization with Optuna (TPE)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

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

ModelKind = Literal["random_forest", "xgboost", "xgboost_2d"]
SamplerName = Literal["tpe", "random"]
ObjectiveFn = Callable[[optuna.Trial], float]
TrialCallback = Callable[[optuna.Study, optuna.trial.FrozenTrial], None]

RF_PARAM_GUIDE: dict[str, str] = {
    "n_estimators": "How many trees. More = more stable, slower.",
    "max_depth": "How deep each tree can split. Deeper = more detail, easier to overfit.",
    "min_samples_leaf": "Min rows in a leaf. Higher = simpler trees.",
}

RF_PARAM_RANGE: dict[str, str] = {
    "n_estimators": "100–400",
    "max_depth": "4–16",
    "min_samples_leaf": "1–20",
}

XGB_PARAM_GUIDE: dict[str, str] = {
    "n_estimators": "How many boosting rounds.",
    "max_depth": "Depth of each small tree (usually shallower than RF).",
    "learning_rate": "Step size per new tree. Smaller + more rounds often holds up better.",
}

XGB_PARAM_RANGE: dict[str, str] = {
    "n_estimators": "100–400",
    "max_depth": "2–8",
    "learning_rate": "0.01–0.2 (log)",
}

def _pr_auc_scorer(estimator: Any, x: pd.DataFrame, y: pd.Series) -> float:
    return float(average_precision_score(y, predict_proba_positive(estimator, x)))


@dataclass
class OptimizeResult:
    best_models: dict[str, Any]
    best_params: dict[str, dict[str, Any]]
    cv_scores: dict[str, float]
    test_metrics: dict[str, dict[str, float]]
    studies: dict[str, optuna.Study] = field(default_factory=dict)


def _cv_pr_auc(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
    n_jobs: int = -1,
) -> float:
    cv = StratifiedKFold(
        n_splits=n_splits or config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    scores = cross_val_score(model, x, y, scoring=_pr_auc_scorer, cv=cv, n_jobs=n_jobs)
    return float(np.mean(scores))


def rf_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Declare the Random Forest search space and ask the sampler for values."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def xgb_search_space(trial: optuna.Trial, spw: float) -> dict[str, Any]:
    """Declare the XGBoost search space. ``scale_pos_weight`` is fixed, not tuned."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "scale_pos_weight": spw,
    }


def xgb_2d_search_space(trial: optuna.Trial, spw: float) -> dict[str, Any]:
    """TPE-vs-random demo: only depth and learning rate move, so the scatter is readable."""
    return {
        "n_estimators": 80,
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "scale_pos_weight": spw,
    }


def build_objective(
    model_kind: ModelKind,
    x: pd.DataFrame,
    y: pd.Series,
    cv_folds: int | None = None,
    n_jobs: int = -1,
) -> ObjectiveFn:
    """Return an Optuna objective that CV-scores RF or XGBoost PR-AUC."""
    folds = cv_folds or config.CV_FOLDS
    spw = scale_pos_weight(y) if model_kind in ("xgboost", "xgboost_2d") else None

    def objective(trial: optuna.Trial) -> float:
        if model_kind == "random_forest":
            params = rf_search_space(trial)
            model = make_random_forest(**params)
        elif model_kind == "xgboost":
            assert spw is not None
            params = xgb_search_space(trial, spw)
            model = make_xgboost(y_train=y, **params)
        elif model_kind == "xgboost_2d":
            assert spw is not None
            params = xgb_2d_search_space(trial, spw)
            model = make_xgboost(y_train=y, **params)
        else:
            raise ValueError(f"Unknown model_kind={model_kind!r}")
        return _cv_pr_auc(model, x, y, n_splits=folds, n_jobs=n_jobs)

    return objective


def create_hpo_study(
    study_name: str,
    sampler: SamplerName = "tpe",
    seed: int | None = None,
    n_startup_trials: int | None = None,
) -> optuna.Study:
    """
    ``n_startup_trials`` is how many random draws TPE makes before it
    starts concentrating. Keep it small in a live demo (3–5) so that shift is visible.
    """
    seed = config.RANDOM_STATE if seed is None else seed
    if sampler == "random":
        samp: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler == "tpe":
        startup = 5 if n_startup_trials is None else n_startup_trials
        samp = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup)
    else:
        raise ValueError(f"Unknown sampler={sampler!r}")
    return optuna.create_study(direction="maximize", study_name=study_name, sampler=samp)


def run_n_trials(
    study: optuna.Study,
    objective: ObjectiveFn,
    n_trials: int,
    callback: TrialCallback | None = None,
    timeout: int | None = None,
) -> optuna.Study:
    """Run ``n_trials`` (optionally one-at-a-time via callback for live UI)."""
    callbacks = [callback] if callback is not None else None
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=callbacks,
        show_progress_bar=False,
    )
    return study


def trials_frame(study: optuna.Study, n_startup: int | None = None) -> pd.DataFrame:
    """One row per trial: number, value, params, best_so_far, phase, near_best."""
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
        }
        row.update(trial.params)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty or "value" not in frame.columns:
        return frame
    completed = pd.to_numeric(frame["value"], errors="coerce")
    frame["best_so_far"] = completed.cummax()
    best = float(completed.max()) if completed.notna().any() else None
    if best is not None:
        frame["is_best"] = completed == best
        frame["near_best"] = completed >= (best - 0.01)
    else:
        frame["is_best"] = False
        frame["near_best"] = False
    if n_startup is None:
        n_startup = int(getattr(study.sampler, "_n_startup_trials", 5) or 5)
    frame["phase"] = np.where(frame["number"] < n_startup, "early", "later")
    return frame


def explain_recent_sampling(study: optuna.Study) -> str:
    """Plain-English caption of what the sampler is doing right now."""
    completed = [t for t in study.trials if t.value is not None]
    n = len(completed)
    if n == 0:
        return "No completed trials yet."
    if n < 4:
        return (
            f"Trial {n}: still sampling the full range. "
            "The first few trials are random so later ones have a history to use."
        )

    best = study.best_trial
    sampler_name = study.sampler.__class__.__name__
    later = completed[n // 2 :]
    focus = next(
        (key for key in ("max_depth", "depth", "learning_rate") if key in best.params),
        None,
    )
    if focus is None:
        numeric = [k for k, v in best.params.items() if isinstance(v, (int, float))]
        focus = numeric[0] if numeric else None

    is_tpe = sampler_name.startswith("TPE")
    if focus:
        vals = [t.params[focus] for t in later if focus in t.params]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            mean_v = float(sum(vals) / len(vals))
            if is_tpe:
                return (
                    f"Trials 1–{max(n // 2, 1)} were mostly random. After that TPE stays near "
                    f"`{focus}` ≈ {mean_v:.3g}, where PR-AUC peaked at {best.value:.3f} "
                    f"({best.params})."
                )
            return (
                f"Random search has no memory — trial {n} is still drawn from the full range. "
                f"Best so far: `{focus}`={best.params.get(focus)} "
                f"(CV PR-AUC {best.value:.3f})."
            )
    if is_tpe:
        return f"After {n} trials, TPE's best score is {best.value:.3f}."
    return f"After {n} random trials, best score is {best.value:.3f}."


def optimize_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any], float, optuna.Study]:
    n_trials = n_trials or config.N_TRIALS
    objective = build_objective("random_forest", x_train, y_train)
    study = create_hpo_study("rf_pr_auc", sampler="tpe")
    run_n_trials(study, objective, n_trials, timeout=config.OPTUNA_TIMEOUT_SEC)
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
    objective = build_objective("xgboost", x_train, y_train)
    study = create_hpo_study("xgb_pr_auc", sampler="tpe")
    run_n_trials(study, objective, n_trials, timeout=config.OPTUNA_TIMEOUT_SEC)
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
    cv_scores = {
        "random_forest": round(rf_cv, 2),
        "xgboost": round(xgb_cv, 2),
    }

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
