# Input data (session swap)

Drop a CSV here and point `DATA_PATH` in `src/credit_fraud/config.py` at it, for example:

```python
DATA_PATH = PROJECT_ROOT / "input" / "my_sample.csv"
DATA_URL = None
TARGET_COL = "default_next_month"  # or your label column
```

Default project data is the **UCI credit-default** file (downloaded to `data/credit_default.csv`) with explainable columns — see `docs/dataset_features.md`.
