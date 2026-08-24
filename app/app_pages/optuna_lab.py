"""Optuna lab: short notes, then tune XGBoost on the UCI credit-default data."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from st_compat import (
    button,
    choice,
    dataframe,
    dataframe_in,
    form_submit_button,
    mermaid_chart,
    metric,
    plotly_chart,
    plotly_in,
)

from credit_fraud import config
from credit_fraud.data import load_raw_dataframe, split_xy, train_test_split_stratified
from credit_fraud.features import build_features
from credit_fraud.modeling import classification_metrics, make_xgboost, predict_proba_positive
from credit_fraud.optimize import (
    XGB_PARAM_GUIDE,
    XGB_PARAM_RANGE,
    SamplerName,
    build_objective,
    create_hpo_study,
    explain_recent_sampling,
    optimize_xgboost,
    run_n_trials,
    trials_frame,
    xgb_search_space,
)
from credit_fraud.optuna_plots import (
    plot_param_scatter,
    plot_pr_auc_intuition,
    plot_search_methods_overview,
    plot_search_scatter,
    plot_tpe_parzen_example,
    plot_tpe_vs_random_history,
    plot_trial_history,
    plot_tuning_task_overview,
)

STEPS = [
    "1. Why we tune",
    "2. Grid vs random vs Bayesian",
    "3. The loop and TPE",
    "4. TPE math",
    "5. XGB settings",
    "6. Why PR-AUC",
    "7. Code",
    "8. TPE vs random",
    "9. Tune XGBoost",
]

SAMPLER_LABELS = {"TPE (Bayesian)": "tpe", "Random": "random"}
SCATTER_X, SCATTER_Y = "max_depth", "learning_rate"


@st.cache_data(show_spinner="Preparing a session-sized train/test split...")
def prepare_hpo_data(max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_raw_dataframe()
    x, y = split_xy(df)
    cap = min(max(int(max_rows), 500), len(x))
    if len(x) > cap:
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=cap,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
    x_train, x_test, y_train, y_test = train_test_split_stratified(x, y)
    x_train_fe, x_test_fe, _ = build_features(x_train, x_test, y_train)
    return x_train_fe, x_test_fe, y_train, y_test


def _baseline(y: pd.Series) -> float:
    return float((y == config.POSITIVE_LABEL).mean())


def _kpi_row(study: Any, n_target: int | None = None) -> None:
    completed = [t for t in study.trials if t.value is not None]
    n = len(completed)
    label = f"{n} / {n_target}" if n_target else str(n)
    current = completed[-1].value if completed else None
    best = study.best_value if completed else None
    c1, c2, c3 = st.columns(3)
    with c1:
        metric("Trials", label)
    with c2:
        metric("Latest PR-AUC", f"{current:.3f}" if current is not None else "—")
    with c3:
        metric("Best so far", f"{best:.3f}" if best is not None else "—")


def _show_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    show = frame.drop(columns=["state"], errors="ignore")
    dataframe(show, hide_index=True)


def render_live_study(study: Any, n_target: int | None = None, baseline: float | None = None) -> None:
    frame = trials_frame(study)
    _kpi_row(study, n_target=n_target)
    st.info(explain_recent_sampling(study))
    plotly_chart(plot_trial_history(frame, baseline=baseline))
    if SCATTER_X in frame.columns and SCATTER_Y in frame.columns:
        plotly_chart(plot_param_scatter(frame, SCATTER_X, SCATTER_Y, title="Star = best. Orange = later trials."))
    _show_table(frame)
    completed = [t for t in study.trials if t.value is not None]
    if completed:
        st.subheader("Best settings so far")
        st.json(study.best_params)


def render_optuna_builtin(study: Any) -> None:
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < 3:
        st.caption("Need at least 3 finished trials for Optuna's built-in plots.")
        return
    from optuna.visualization import (
        plot_contour,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )

    st.subheader("Optuna plots")
    plotly_chart(plot_optimization_history(study))
    try:
        plotly_chart(plot_parallel_coordinate(study))
    except (ValueError, RuntimeError) as exc:
        st.caption(f"Parallel coordinates skipped: {exc}")
    params = [p for p in (SCATTER_X, SCATTER_Y) if p in study.best_params]
    if len(params) == 2:
        try:
            plotly_chart(plot_contour(study, params=params))
        except (ValueError, RuntimeError) as exc:
            st.caption(f"Contour skipped: {exc}")
    try:
        plotly_chart(plot_slice(study))
    except (ValueError, RuntimeError) as exc:
        st.caption(f"Slice skipped: {exc}")
    try:
        plotly_chart(plot_param_importances(study))
    except (ValueError, RuntimeError) as exc:
        st.caption(f"Importances skipped (need more diverse trials): {exc}")


def _holdout_block(
    study: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    if study.best_trial.value is None:
        return
    model = make_xgboost(y_train=y_train, **dict(study.best_params))
    model.fit(x_train, y_train)
    proba = predict_proba_positive(model, x_test)
    metrics = classification_metrics(y_test, proba)
    base = _baseline(y_test)
    dummy_acc = 1.0 - base
    with st.container(border=True):
        st.subheader("Hold-out")
        st.caption(
            f"Always-predict-no-default accuracy ≈ {dummy_acc:.0%}. "
            f"Random ranking PR-AUC ≈ {base:.2f}. A model around 0.50 is more than 2× that, not a coin flip."
        )
        cols = st.columns(3)
        cols[0].metric("Hold-out PR-AUC", f"{metrics['pr_auc']:.2f}")
        cols[1].metric("Hold-out ROC-AUC", f"{metrics['roc_auc']:.2f}")
        cols[2].metric("CV best (Optuna)", f"{study.best_value:.3f}")


def step_why() -> None:
    st.subheader("Why we tune")
    plotly_chart(plot_tuning_task_overview(positive_rate=0.22))
    st.caption(
        "Features stay fixed. Optuna searches XGB depth, learning rate, and tree count to maximize CV PR-AUC."
    )


def step_search() -> None:
    st.subheader("Grid vs random vs Bayesian")
    plotly_chart(plot_search_methods_overview())
    st.caption(
        "Grid tries every combo (explodes). Random has no memory. TPE samples near past winners with occasional exploration."
    )


def step_loop() -> None:
    st.subheader("The loop")
    mermaid_chart(
        """
flowchart LR
    A[pick settings] --> B[train XGB CV]
    B --> C[PR-AUC score]
    C --> D[store trial]
    D --> E{TPE?}
    E -->|yes| F[near good scores]
    E -->|random| G[uniform draw]
    F --> A
    G --> A
"""
    )
    st.caption(
        "First 3–5 trials are random. Then TPE splits past trials into high vs low PR-AUC and samples more from the high pile."
    )


def step_tpe_math() -> None:
    st.subheader("TPE math")
    st.caption(
        "Reference: Tree-structured Parzen Estimator, built from AUC (same maths as our PR-AUC)."
    )
    st.markdown(
        "In one sentence: **TPE learns where good hyperparameters live and samples more heavily from those regions.**"
    )

    obs, cutoff, densities, want, why, process = st.tabs(
        [
            "1. Observations",
            "2. The cutoff",
            "3. Two densities",
            "4. What TPE wants",
            "5. Why it works",
            "6. Whole process",
        ]
    )

    with obs:
        st.markdown(
            """
Hyperparameters $x$. For this example:

$x = (\\text{learning rate},\\ \\text{max depth})$

For every choice of $x$ we train XGBoost and get an AUC:

$y = f(x)$

Example: $x = (0.05,\\ 5)$ might produce $y = 0.84$.

TPE observes pairs $(x, y)$.
"""
        )

    with cutoff:
        st.markdown(
            """
Suppose we have 10 trials. Choose a cutoff $y^*$ that separates good and bad trials.

Because we **maximise** AUC:

- Good: $y > y^*$
- Bad: $y \\leq y^*$

Example: $y^* = 0.80$. Then $0.84$, $0.83$, $0.81$ are good; lower AUCs are bad.
"""
        )

    with densities:
        st.markdown(
            """
The Parzen estimator fits **two** densities over the hyperparameters.

Good trials: $\\ell(x) = P(x \\mid y > y^*)$

Bad trials: $g(x) = P(x \\mid y \\leq y^*)$

Suppose good learning rates are $[0.04, 0.05, 0.07, 0.08]$ and bad ones are
$[0.15, 0.20, 0.25, 0.30]$. Values around $0.05$–$0.08$ have high $\\ell(x)$ and low $g(x)$.
"""
        )
        plotly_chart(plot_tpe_parzen_example())

    with want:
        st.markdown(
            """
TPE wants an $x$ that is **likely under the good density** and **unlikely under the bad density**.
That is a large ratio

$\\dfrac{\\ell(x)}{g(x)}$
"""
        )
        dataframe(
            pd.DataFrame(
                {
                    "learning rate": ["0.05", "0.10", "0.20"],
                    "ℓ(x) good": ["0.40", "0.20", "0.03"],
                    "g(x) bad": ["0.05", "0.15", "0.30"],
                    "ℓ(x) / g(x)": ["8", "1.33", "0.10"],
                }
            ),
            hide_index=True,
        )
        st.markdown(
            r"""
TPE therefore prefers something around $x = 0.05$, because

$\dfrac{\ell(0.05)}{g(0.05)} = 8$
"""
        )

    with why:
        st.markdown(
            r"""
Ordinary Bayesian optimisation models $P(y \mid x)$:
*if I choose these hyperparameters, how good will my model be?*

TPE flips it and models $P(x \mid y)$:
*given that the result was good, what hyperparameters were likely used?*

Bayes:

$P(y \mid x) = \dfrac{P(x \mid y)\, P(y)}{P(x)}$

With the two densities

$\ell(x) = P(x \mid y > y^*),\quad g(x) = P(x \mid y \leq y^*)$

TPE searches for

$x_{\text{next}} = \arg\max_x \dfrac{\ell(x)}{g(x)}$
"""
        )

    with process:
        mermaid_chart(
            """
flowchart TD
    A["Choose hyperparameters x"] --> B[Train XGBoost]
    B --> C["Get AUC = y"]
    C --> D["Separate trials using y*"]
    D --> E[Good trials]
    D --> F[Bad trials]
    E --> G["l(x) = P(x | good)"]
    F --> H["g(x) = P(x | bad)"]
    G --> I["Choose x where l(x)/g(x) is large"]
    H --> I
    I --> J[Try those hyperparameters]
    J --> K[Get new AUC]
    K --> D
"""
        )
        st.info("TPE learns where good hyperparameters live and samples more heavily from those regions.")


def _knob_table(guide: dict[str, str], ranges: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setting": list(guide),
            "meaning": list(guide.values()),
            "range": [ranges[key] for key in guide],
        }
    )


def step_knobs() -> None:
    st.subheader("Settings we search")
    st.caption(
        "Only the knobs that move the score most. Everything else stays at `make_xgboost` defaults. "
        "`scale_pos_weight` is set from n_good / n_default and is not searched."
    )
    dataframe(_knob_table(XGB_PARAM_GUIDE, XGB_PARAM_RANGE), hide_index=True)


def step_prauc() -> None:
    st.subheader("Why PR-AUC")
    plotly_chart(plot_pr_auc_intuition(positive_rate=0.22))
    st.caption(
        "~22% default rate. Always predicting 'no default' is ~78% accurate but useless. "
        "PR-AUC rewards ranking actual defaults near the top. Optuna maximizes it (`direction='maximize'`)."
    )


def step_code() -> None:
    st.subheader("Code")
    st.caption("`optimize.py` runs the search. `make_xgboost` in `modeling.py` builds the model.")
    mermaid_chart(
        """
flowchart LR
    A["xgb_search_space"] --> B["make_xgboost"]
    B --> C["CV PR-AUC"]
    C --> D["study.optimize"]
    D --> A
    D --> E["best_params → refit"]
"""
    )
    dataframe(
        pd.DataFrame(
            {
                "Piece": [
                    "xgb_search_space",
                    "make_xgboost",
                    "build_objective",
                    "create_hpo_study",
                    "run_n_trials",
                    "optimize_xgboost",
                ],
                "Does": [
                    "Ask Optuna for trees, depth, learning rate.",
                    "Build an unfitted XGBClassifier (defaults + trial params).",
                    "One trial: those params → mean CV PR-AUC.",
                    "Study that maximises PR-AUC (TPE or random).",
                    "Repeat the objective N times.",
                    "All of the above, then refit the winner on all train rows.",
                ],
            }
        ),
        hide_index=True,
    )

    space, factory, objective, study, wrap = st.tabs(
        [
            "1. Search space",
            "2. make_xgboost",
            "3. Objective",
            "4. Study",
            "5. Wrapper",
        ]
    )

    with space:
        st.markdown(
            "`suggest_int` / `suggest_float`: Optuna, give me a value in this range and remember it. "
            "`log=True` thinks in orders of magnitude (0.01 vs 0.1 matters more than 0.11 vs 0.12). "
            "First few trials are random; later TPE prefers values that sat next to high PR-AUC."
        )
        st.code('max_depth = trial.suggest_int("max_depth", 2, 8)', language="python")
        st.code(inspect.getsource(xgb_search_space), language="python")
        st.caption("`scale_pos_weight` is passed in, not suggested — it is fixed from the label mix.")

    with factory:
        st.markdown(
            """
`make_xgboost` is a **factory**. It does **not** train.

1. Start from defaults: 200 trees, depth 4, lr 0.05, subsample/colsample 0.8, `hist`, `eval_metric="aucpr"`.
2. If `y_train` is given and `scale_pos_weight` is not already in params, set it to n_good / n_default.
3. Overlay `**params` from the trial (the three knobs Optuna searched).
4. Return an **unfitted** `XGBClassifier`.

CV later fits it on each fold. After the study, the wrapper calls it again with `best_params` and `.fit()` on all train data.
"""
        )
        st.code(inspect.getsource(make_xgboost), language="python")

    with objective:
        st.markdown(
            "Called once per trial. The lab always passes `model_kind='xgboost'`. "
            "That score becomes `trial.value`. Higher is better."
        )
        st.code(
            """def objective(trial):
    params = xgb_search_space(trial, spw)       # 3 knobs from Optuna
    model = make_xgboost(y_train=y, **params)   # unfitted
    return cv_pr_auc(model, x, y)               # mean stratified PR-AUC
""",
            language="python",
        )
        with st.expander("Full `build_objective` (also has an RF branch for `run_training()`)"):
            st.code(inspect.getsource(build_objective), language="python")

    with study:
        st.markdown(
            '`direction="maximize"`. TPE: `n_startup_trials` random draws, then concentrate. '
            "Random: every trial is still a uniform draw — no memory."
        )
        st.code(inspect.getsource(create_hpo_study), language="python")
        st.code(inspect.getsource(run_n_trials), language="python")

    with wrap:
        st.markdown(
            "Session glue: build objective → create study → run trials → "
            "`make_xgboost(**best_params).fit(...)`. The lab does the same loop live, with charts after each trial."
        )
        st.code(inspect.getsource(optimize_xgboost), language="python")


def step_compare() -> None:
    st.subheader("TPE vs random")
    st.markdown(
        "XGBoost on the UCI sample. TPE and random get the **same budget**. "
        "Star = TPE best. Later TPE (orange) should sit near that star; random (x) stays spread out. "
        "Dashed line = random ranking (default rate)."
    )
    with st.form("demo_compare", border=True):
        n_trials = st.slider("Trials each", 10, 30, 20, step=5)
        max_rows = st.slider("Rows", 1000, 4000, 2000, step=500)
        run_demo = form_submit_button("Run TPE vs random")

    if run_demo:
        x_train, x_test, y_train, y_test = prepare_hpo_data(max_rows)
        _ = x_test, y_test
        objective = build_objective("xgboost_2d", x_train, y_train, cv_folds=2)
        startup = max(3, n_trials // 5)
        tpe = create_hpo_study("xgb_tpe", sampler="tpe", n_startup_trials=startup)
        rnd = create_hpo_study("xgb_random", sampler="random", seed=config.RANDOM_STATE + 7)
        with st.status("Training on the UCI sample…", expanded=True) as status:
            st.write(f"{len(x_train):,} train rows · 2-fold CV · metric = PR-AUC")
            st.write("TPE…")
            run_n_trials(tpe, objective, n_trials)
            st.write("Random…")
            run_n_trials(rnd, objective, n_trials)
            status.update(label="Done", state="complete")
        st.session_state.demo_tpe = tpe
        st.session_state.demo_random = rnd
        st.session_state.demo_baseline = _baseline(y_train)

    tpe = st.session_state.get("demo_tpe")
    rnd = st.session_state.get("demo_random")
    baseline = st.session_state.get("demo_baseline")
    if tpe is None or rnd is None:
        st.info("Run TPE vs random. This trains real models — expect a minute or two.")
        return

    tpe_frame = trials_frame(tpe)
    rnd_frame = trials_frame(rnd)
    plotly_chart(
        plot_search_scatter(
            tpe_frame,
            rnd_frame,
            title="max_depth vs learning_rate — star is TPE best",
        )
    )
    plotly_chart(plot_tpe_vs_random_history(tpe_frame, rnd_frame, baseline=baseline))
    if baseline is not None:
        st.caption(f"Dashed line: random ranking ≈ {baseline:.2f}. A score near 0.50 is more than 2× that.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**TPE**")
        st.info(explain_recent_sampling(tpe))
        metric("Best CV PR-AUC", f"{tpe.best_value:.3f}")
        st.json(tpe.best_params)
    with c2:
        st.markdown("**Random**")
        st.info(explain_recent_sampling(rnd))
        metric("Best CV PR-AUC", f"{rnd.best_value:.3f}")
        st.json(rnd.best_params)


def step_live() -> None:
    st.subheader("Tune XGBoost")
    st.markdown(
        "Same UCI data. The three knobs in the search space. Charts update after each trial. "
        "Star = best. Flip sampler to Random to show it does not cluster."
    )
    with st.expander("What the settings mean"):
        dataframe(_knob_table(XGB_PARAM_GUIDE, XGB_PARAM_RANGE), hide_index=True)

    with st.form("live_hpo", border=True):
        sampler_label = choice(
            "Sampler",
            list(SAMPLER_LABELS),
            default="TPE (Bayesian)",
            key="live_sampler_label",
            horizontal=True,
        )
        n_trials = st.slider("Trials", 3, 20, 8)
        max_rows = st.slider("Rows", 1000, 8000, 4000, step=500)
        cv_folds = st.slider("CV folds", 2, 3, 2)
        run_live = form_submit_button("Start tuning")

    sampler: SamplerName = SAMPLER_LABELS.get(sampler_label, "tpe")  # type: ignore[assignment]

    if run_live:
        x_train, x_test, y_train, y_test = prepare_hpo_data(max_rows)
        objective = build_objective("xgboost", x_train, y_train, cv_folds=cv_folds)
        startup = max(3, min(5, n_trials // 2))
        study = create_hpo_study(
            f"xgboost_{sampler}",
            sampler=sampler,
            n_startup_trials=startup,
        )
        kpi_slot = st.empty()
        caption_slot = st.empty()
        chart_slot = st.empty()
        scatter_slot = st.empty()
        table_slot = st.empty()
        baseline = _baseline(y_train)

        def _on_trial(current_study: Any, _trial: Any) -> None:
            frame = trials_frame(current_study)
            with kpi_slot.container():
                _kpi_row(current_study, n_target=n_trials)
            caption_slot.info(explain_recent_sampling(current_study))
            plotly_in(chart_slot, plot_trial_history(frame, baseline=baseline))
            if SCATTER_X in frame.columns and SCATTER_Y in frame.columns:
                plotly_in(scatter_slot, plot_param_scatter(frame, SCATTER_X, SCATTER_Y))
            dataframe_in(table_slot, frame.drop(columns=["state"], errors="ignore"), hide_index=True)

        with st.status(f"Tuning XGBoost with {sampler_label}…", expanded=True) as status:
            st.write(
                f"{len(x_train):,} train rows · {x_train.shape[1]} features · "
                f"{cv_folds}-fold CV · metric = PR-AUC · random ranking ≈ {baseline:.2f}"
            )
            run_n_trials(study, objective, n_trials, callback=_on_trial)
            status.update(label="Tuning finished", state="complete")

        st.session_state.live_study = study
        st.session_state.live_meta = {
            "sampler": sampler,
            "n_trials": n_trials,
            "max_rows": max_rows,
            "cv_folds": cv_folds,
            "sampler_label": sampler_label,
            "baseline": baseline,
        }
        render_optuna_builtin(study)
        _holdout_block(study, x_train, x_test, y_train, y_test)
        return

    study = st.session_state.get("live_study")
    meta = st.session_state.get("live_meta") or {}
    if study is None:
        st.info("Choose a sampler, then start tuning.")
        return

    st.caption(
        f"Last run: XGBoost · {meta.get('sampler_label', meta.get('sampler'))} · "
        f"{meta.get('max_rows', '?')} rows · {meta.get('cv_folds', '?')}-fold CV"
    )
    render_live_study(
        study,
        n_target=meta.get("n_trials"),
        baseline=meta.get("baseline"),
    )
    render_optuna_builtin(study)

    if button("Score best settings on hold-out", key="score_holdout"):
        x_train, x_test, y_train, y_test = prepare_hpo_data(int(meta.get("max_rows") or 4000))
        _holdout_block(study, x_train, x_test, y_train, y_test)


STEP_RENDERERS = {
    STEPS[0]: step_why,
    STEPS[1]: step_search,
    STEPS[2]: step_loop,
    STEPS[3]: step_tpe_math,
    STEPS[4]: step_knobs,
    STEPS[5]: step_prauc,
    STEPS[6]: step_code,
    STEPS[7]: step_compare,
    STEPS[8]: step_live,
}


def _shift_step(delta: int) -> None:
    current = st.session_state.get("optuna_step_radio", STEPS[0])
    idx = STEPS.index(current) if current in STEPS else 0
    st.session_state.optuna_step_radio = STEPS[max(0, min(len(STEPS) - 1, idx + delta))]


def render() -> None:
    st.session_state.setdefault("demo_tpe", None)
    st.session_state.setdefault("demo_random", None)
    st.session_state.setdefault("demo_baseline", None)
    st.session_state.setdefault("live_study", None)
    st.session_state.setdefault("live_meta", {})
    if st.session_state.get("optuna_step_radio") not in STEPS:
        st.session_state.optuna_step_radio = STEPS[0]

    st.title("Optuna lab")
    st.caption("UCI credit default. Tune XGBoost with Optuna. Previous / Next to move through.")

    current = st.sidebar.radio(
        "Session step",
        options=STEPS,
        index=0,
        key="optuna_step_radio",
    )
    idx = STEPS.index(current) if current in STEPS else 0

    prev_col, _, next_col = st.columns([1, 4, 1])
    with prev_col:
        button(
            "Previous",
            key="optuna_prev",
            disabled=idx == 0,
            on_click=_shift_step,
            args=(-1,),
        )
    with next_col:
        button(
            "Next",
            key="optuna_next",
            disabled=idx == len(STEPS) - 1,
            on_click=_shift_step,
            args=(1,),
        )

    st.markdown(f"**{current}** · step {idx + 1} of {len(STEPS)}")
    STEP_RENDERERS[current]()
