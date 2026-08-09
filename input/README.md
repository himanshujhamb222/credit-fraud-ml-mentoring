# Input data (session swap)

Drop a CSV here and point `DATA_PATH` in `src/credit_fraud/config.py` at it, for example:

```python
DATA_PATH = PROJECT_ROOT / "input" / "my_fraud_sample.csv"
DATA_URL = None  # skip download when a local path is set and exists
```

Requirements:

- Binary target column (name must match `TARGET_COL`, default `Class`)
- Numeric features preferred (PCA components / amount / time style columns work well)
