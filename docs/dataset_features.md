# Dataset: UCI Default of Credit Card Clients

We use the **Taiwan credit-card default** dataset (UCI id 350), not the anonymized Kaggle fraud file with `V1`…`V28`.

| | |
|---|---|
| Rows | 30,000 clients |
| Target | `default_next_month` (1 ≈ 22% defaults) |
| Why this set | Every column has a clear business meaning — easy to mentor |

Source: [UCI repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
Configured in [`src/credit_fraud/config.py`](../src/credit_fraud/config.py) (`DATA_URL`, `TARGET_COL`).

## Feature dictionary

| Feature | Meaning |
|---------|---------|
| `LIMIT_BAL` | Credit limit (NT$) |
| `SEX` | 1 = male, 2 = female |
| `EDUCATION` | 1 = grad school, 2 = university, 3 = high school, 4+ = other/unknown |
| `MARRIAGE` | 1 = married, 2 = single, 3 = other |
| `AGE` | Age in years |
| `PAY_0` … `PAY_6` | Repayment status by month (−1 = paid duly, positive = months delayed) |
| `BILL_AMT1` … `BILL_AMT6` | Monthly bill statement amounts (NT$) |
| `PAY_AMT1` … `PAY_AMT6` | Monthly amounts paid (NT$) |
| `default_next_month` | **Target** — defaulted next month |

## Engineered features (in `features.py`)

| Feature | Idea |
|---------|------|
| `LIMIT_BAL_log1p` | Log credit limit (skew) |
| `UTIL_RATIO_1` | Recent utilisation = bill / limit |
| `PAY_RATIO_1` | Share of last bill paid |
| `AVG_PAY_STATUS` / `MAX_PAY_STATUS` | Typical / worst lateness |

## Session swap

Put your CSV in `input/` and set `DATA_PATH` + `TARGET_COL` in `config.py`. Column names in rank/plot/Streamlit will follow **your** schema.
