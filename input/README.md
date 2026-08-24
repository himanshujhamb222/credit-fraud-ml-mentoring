# Input data (session swap)

Drop a CSV here and point `DATA_PATH` in `src/credit_fraud/config.py` at it, for example:

```python
DATA_PATH = PROJECT_ROOT / "input" / "my_sample.csv"
DATA_URL = None
TARGET_COL = "default_next_month"  # or your label column
```

Default project data is the **UCI credit-default** file (downloaded to `data/credit_default.csv`) with explainable columns — see `docs/dataset_features.md`.

## NLP POS corpus (CRF lab)

`input/nlp_pos/` holds the toy POS tagging TSVs for the **CRF lab** Streamlit page. Copy that folder to reuse the session dataset elsewhere; see `input/nlp_pos/README.md`.
