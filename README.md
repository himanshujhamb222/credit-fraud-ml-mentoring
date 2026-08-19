# Credit Fraud / Default Mentoring Repo

Session-friendly base for **imbalanced credit risk modeling**: EDA → rank & plot → feature engineering → benchmark → Random Forest / XGBoost with **Optuna** → selection → **Streamlit** + **FastAPI**.

Everything is parameterized in [`src/credit_fraud/config.py`](src/credit_fraud/config.py) — **no CLI args**.

## Teaching guides

| Guide | Path |
|-------|------|
| **Session takeaway (MOM)** — formatted Word doc (upload to Google Drive → Open with Google Docs) | [`docs/session_takeaway_mom.docx`](docs/session_takeaway_mom.docx) |
| Dataset & feature meanings | [`docs/dataset_features.md`](docs/dataset_features.md) |
| Bayesian optimisation + Optuna (layman session script) | [`docs/bayesian_optimization_optuna.md`](docs/bayesian_optimization_optuna.md) |
| Optuna step-by-step notebook (RF / XGBoost, TPE vs random on UCI data) | [`notebooks/02_optuna_bayesian.ipynb`](notebooks/02_optuna_bayesian.ipynb) |
| XGBoost with math | [`docs/xgboost_guide.md`](docs/xgboost_guide.md) |

## Default dataset (interpretable)

**UCI Default of Credit Card Clients** (Taiwan) — columns like `LIMIT_BAL`, `AGE`, `PAY_*`, `BILL_AMT_*`, `PAY_AMT_*` (not anonymized `V1`…`V28`).

Target: `default_next_month` (~22% positives). Full dictionary in the dataset doc above / `FEATURE_DICTIONARY` in config.

## Setup

```bash
cd Naresh
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Swap data for the live session

Edit `src/credit_fraud/config.py`:

1. Drop a CSV in `input/` and set `DATA_PATH`, **or**
2. Change `DATA_URL`.

Also set `TARGET_COL` if your label is not `default_next_month`.

## Train (end-to-end)

```bash
python -c "from credit_fraud.pipeline import run_training; run_training()"
```

Or step through [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb). For the HPO session, use [`notebooks/02_optuna_bayesian.ipynb`](notebooks/02_optuna_bayesian.ipynb).

Artifacts land in `artifacts/` (`best_model.joblib`, `metrics.json`, rankings, Plotly HTML figures).

## Streamlit (EDA + Optuna lab)

```bash
streamlit run app/streamlit_app.py
```

- **Optuna lab** — notes → TPE math → `optimize.py` / `make_xgboost` → TPE vs random on UCI data → live XGBoost search. Previous / Next.
- **EDA and modelling** — target, rank-and-plot, features, importance (train first for the last tab).

## FastAPI (inference)

```bash
uvicorn api.main:app --reload --app-dir .
```

- `GET /health` — model loaded?
- `GET /metrics` — selection metrics JSON
- `POST /predict` — score one row

## Mentoring map

| Step | Module |
|------|--------|
| Config | `credit_fraud.config` |
| Load / split | `credit_fraud.data` |
| EDA (X + y) | `credit_fraud.eda` |
| Rank & plot (Plotly bins × count + capture rate) | `credit_fraud.rank_plot` |
| Features | `credit_fraud.features` |
| Benchmark | `credit_fraud.benchmark` |
| Models | `credit_fraud.modeling` |
| Optuna HPO (search spaces, 2-param demo, stepwise trials) | `credit_fraud.optimize` |
| Selection | `credit_fraud.selection` |
| Orchestrator | `credit_fraud.pipeline` |

Primary selection metric: **PR-AUC** (class imbalance).
