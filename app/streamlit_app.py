"""Streamlit entry: Optuna + CRF + EDA + transformer + DIET + Naive Bayes labs.

Works on Streamlit 1.32+ (no st.navigation required). The script directory
is ``app/``, so page imports are ``app_pages.*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent
_SRC = _ROOT / "src"
for path in (_APP_DIR, _SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

st.set_page_config(page_title="Credit default mentoring", layout="wide")

page = st.sidebar.radio(
    "Page",
    options=[
        "Optuna lab",
        "CRF lab",
        "EDA and modelling",
        "Transformer lab",
        "DIET lab",
        "Naive Bayes lab",
    ],
    index=0,
    key="nav_page",
)

try:
    if page == "Optuna lab":
        from app_pages.optuna_lab import render as render_optuna

        render_optuna()
    elif page == "CRF lab":
        from app_pages.crf_lab import render as render_crf

        render_crf()
    elif page == "Transformer lab":
        from app_pages.transformer_lab import render as render_transformer

        render_transformer()
    elif page == "DIET lab":
        from app_pages.diet_lab import render as render_diet

        render_diet()
    elif page == "Naive Bayes lab":
        from app_pages.naive_bayes_lab import render as render_nb

        render_nb()
    else:
        from app_pages.eda import render as render_eda

        render_eda()
except ModuleNotFoundError as exc:
    st.error(f"Missing package: `{exc.name}`.")
    st.markdown(
        "This app needs the project environment (Optuna, XGBoost, scikit-learn). "
        "From the repo root:"
    )
    st.code(
        "source .venv/bin/activate   # or: python -m venv .venv && pip install -e \".[dev]\"\n"
        "streamlit run app/streamlit_app.py",
        language="bash",
    )
