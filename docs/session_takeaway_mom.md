SESSION TAKEAWAY (MOM)
Credit default modelling walkthrough

Dataset: UCI Taiwan credit default (LIMIT_BAL, PAY_*, BILL_AMT_*, PAY_AMT_*, AGE, …)
App: streamlit run app/streamlit_app.py  (pages: EDA and modelling · Optuna lab)
Notebook for this session: notebooks/02_optuna_bayesian.ipynb


────────────────────────────────────────
1) PROCESS WE WALKED (STREAMLIT)
────────────────────────────────────────

1. Target & EDA
   - Check class balance (~22% defaults).
   - Look at raw feature distributions (dropdown).
   - Why: understand the problem and what each column means.

2. Rank & Plot
   - For each feature: put values into bins on the X-axis.
   - Bars = number of clients in that bin.
   - Line = capture rate (share of all defaults that fall in that bin).
   - Why: see which feature values capture more defaults.

3. Feature engineering
   - Build simple explainable features (e.g. utilisation = bill/limit, pay ratio, avg/max payment delay).
   - Fit transforms on train only (no leakage).
   - Why: give the model clearer business signals.

4. Modelling
   - Start with simple baselines.
   - Tune Random Forest and XGBoost with Optuna.
   - Pick the winner using PR-AUC (better than accuracy when classes are imbalanced).
   - Review feature importance in the Modelling tab.
   - Why: fair comparison + a model you can explain.

One-liner: load → EDA → rank/plot → features → baselines → Optuna tune → select best → view in Streamlit / serve via FastAPI.


────────────────────────────────────────
2) SAMPLE PLOT (INSERT THIS IMAGE IN THE DOC)
────────────────────────────────────────

Image file (attach/insert into your doc):
docs/assets/sample_capture_PAY_0.png

What the plot shows (PAY_0 = latest repayment status):
- X-axis: bins of PAY_0
- Bars: how many clients sit in each bin
- Line: capture rate = defaults in bin / all defaults

How to read it:
- Tall bar → that value is common.
- High line → that bin holds a large share of the defaults.
- In Streamlit → Rank & Plot, use the dropdown to change the feature (LIMIT_BAL, AGE, PAY_2, …).


────────────────────────────────────────
3) OPTUNA / BAYESIAN OPTIMISATION (LIVE DEMO)
────────────────────────────────────────

Goal: find RF/XGB settings that rank next-month defaults well (PR-AUC), without training every combo.

- Grid / random: no memory (random) or too many combos (grid).
- Optuna TPE: next try looks like settings that scored well.

Loop: pick settings → train, CV PR-AUC → store → pick next. First 3–5 TPE trials are random; then they cluster near the best.

PR-AUC: random ranking ≈ 0.22 (default rate). A model at 0.50+ is more than 2× that, not a coin flip.

Live demo (Optuna lab):
1–5. Why, methods, loop, settings, PR-AUC.
6. `suggest_int` / `optimize.py`.
7. TPE vs random on real UCI data (only max_depth + learning_rate). Star = best. Later TPE near the star.
8. Full RF/XGB search; hold-out PR-AUC.

Code walk: src/credit_fraud/optimize.py (search_space → objective → run_n_trials).
Spoken script: docs/bayesian_optimization_optuna.md
Step-through: notebooks/02_optuna_bayesian.ipynb


────────────────────────────────────────
4) HANDY COMMANDS
────────────────────────────────────────

streamlit run app/streamlit_app.py

python -c "from credit_fraud.pipeline import run_training; run_training()"

# Optuna teaching notebook
# notebooks/02_optuna_bayesian.ipynb


────────────────────────────────────────
5) OPTIONAL DEEPER READING
────────────────────────────────────────

docs/bayesian_optimization_optuna.md
docs/xgboost_guide.md
docs/dataset_features.md
