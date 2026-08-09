# Credit Fraud Mentoring Repo

Session-friendly base for walking through **imbalanced credit-fraud modeling**: EDA → rank & plot → feature engineering → benchmark → Random Forest / XGBoost with **Optuna** Bayesian HPO → model selection → **Streamlit** + **FastAPI**.

Everything is parameterized in [`src/credit_fraud/config.py`](src/credit_fraud/config.py) — **no CLI args**.

## Setup

```bash
cd Naresh
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

## Swap data for the live session

Edit `src/credit_fraud/config.py`:

1. Drop a CSV in `input/` and set `DATA_PATH` to that file, **or**
2. Change `DATA_URL`, **or**
3. Rely on OpenML fallback (`OPENML_DATASET_NAME`).

Also set `TARGET_COL` if your label column is not `Class`.

`MAX_ROWS` defaults to `50000` for snappy demos; set to `None` to use the full file.

## Train (end-to-end)

```bash
python -c "from credit_fraud.pipeline import run_training; run_training()"
```

Or step through [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb) (thin notebook that only calls package functions).

Artifacts land in `artifacts/`:

- `best_model.joblib`
- `metrics.json`
- `feature_importances.csv`
- `feature_ranking.csv`
- figures under `artifacts/figures/`

## Streamlit (EDA + importance)

```bash
streamlit run app/streamlit_app.py
```

## FastAPI (inference)

```bash
uvicorn api.main:app --reload --app-dir .
```

- `GET /health` — model loaded?
- `GET /metrics` — selection metrics JSON
- `POST /predict` — score one row

Example:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/metrics
```

## Mentoring map

| Step | Module |
|------|--------|
| Config | `credit_fraud.config` |
| Load / split | `credit_fraud.data` |
| EDA (X + y) | `credit_fraud.eda` |
| Rank & plot | `credit_fraud.rank_plot` |
| Features | `credit_fraud.features` |
| Benchmark | `credit_fraud.benchmark` |
| Models | `credit_fraud.modeling` |
| Optuna HPO | `credit_fraud.optimize` |
| Selection | `credit_fraud.selection` |
| Orchestrator | `credit_fraud.pipeline` |

Primary selection metric: **PR-AUC** (appropriate for class imbalance).
