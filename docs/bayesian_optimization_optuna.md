# Bayesian optimisation & Optuna

UCI Default of Credit Card Clients. Predict next-month default (~22%). Tune XGBoost settings.

- code: [`src/credit_fraud/optimize.py`](../src/credit_fraud/optimize.py)
- lab: `streamlit run app/streamlit_app.py` → **Optuna lab**
- notebook: [`notebooks/02_optuna_bayesian.ipynb`](../notebooks/02_optuna_bayesian.ipynb)

---

## Session map

| Step | What |
|------|------|
| 1–2 | Why tune; grid vs random vs Bayesian |
| 3–4 | Loop, then TPE math (densities $\ell$ and $g$) |
| 5–6 | XGB settings we actually search, PR-AUC |
| 7 | `make_xgboost` and `optimize.py` |
| 8 | TPE vs random on **real** UCI data |
| 9 | Live XGBoost search |

Demo: ~2k rows, 2-fold, 20 trials each (step 8). Full training is `run_training()`.

---

## 1. Why we tune

Features stay fixed. We change **XGBoost settings** (depth, trees, learning rate). Each try trains a model and scores **PR-AUC**. You cannot try every combo. Optuna picks the next try from past scores.

---

## 2. Grid vs random vs Bayesian

| Method | Does | Problem |
|--------|------|---------|
| Grid | Every combo | Explodes with 5–6 settings |
| Random | Draw uniformly | No memory |
| Bayesian (TPE) | Next try looks like high-scoring settings | Slightly more machinery |

---

## 3. The loop

```
pick settings → train, CV PR-AUC → store → pick next
```

First 3–5 TPE trials are random. Then it splits past trials into high vs low PR-AUC and samples more from the high pile.

---

## 4. Settings

**XGB:** `n_estimators` 100–400, `max_depth` 2–8, `learning_rate` 0.01–0.2 (log). `scale_pos_weight` is `n_good / n_default`, not searched. Other knobs stay at `make_xgboost` defaults.

---

## 5. PR-AUC

Never-predict-default accuracy ≈ 78%. Useless.

Random ranking PR-AUC ≈ default rate ≈ **0.22**. A model at **0.50–0.55** is more than 2× that. Not a coin flip (that would be ROC 0.50).

---

## 6. Code

```python
max_depth = trial.suggest_int("max_depth", 2, 8)
```

“Give me an integer from 2 to 8, and remember it for this trial.” Same idea: `suggest_float`, `suggest_categorical`. `log=True` on `learning_rate` means 0.01 vs 0.1 matters more than 0.11 vs 0.12.

```python
def objective(trial):
    params = xgb_search_space(trial, spw)
    model = make_xgboost(y_train=y_train, **params)  # factory, does not fit
    return cv_pr_auc(model, x_train, y_train)

study = create_hpo_study("xgb_pr_auc", sampler="tpe")
run_n_trials(study, objective, n_trials=10)
study.best_params
```

`make_xgboost` only **builds** the model (defaults + trial params + `scale_pos_weight`). CV fits it. After the study, `optimize_xgboost` refits `best_params` on all train rows.

Step 8 runs TPE and random with the same budget. TPE should cluster near the star (best). Random stays spread out.

---

## 7. What to look at

- History chart: dashed line ≈ 0.22. Best-so-far should sit near 0.50+.
- Scatter: **star = best**. Later TPE points (orange) near the star. Random (x) everywhere.
- Table: `phase` = early vs later; `near_best` = within 0.01 of the best score.

TPE does not guarantee the global best. It spends a small budget better than a grid.

Math: [appendix](#appendix-tpe-math).

---

## Appendix: TPE math (if asked)

```
θ*  =  arg max  f(θ)     f = CV PR-AUC
```

TPE fits two densities over settings (high vs low score) and picks θ that maximises **ℓ(θ) / g(θ)**.
