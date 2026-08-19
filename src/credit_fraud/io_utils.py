"""Save / load models, metrics, and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from credit_fraud import config


def ensure_dirs() -> None:
    """Create artifact directories if missing."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(model: Any, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or config.MODEL_PATH
    joblib.dump(model, path)
    return path


def load_model(path: Path | None = None) -> Any:
    path = path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def save_figure(fig: plt.Figure, name: str) -> Path:
    ensure_dirs()
    path = config.FIGURES_DIR / name
    if not str(name).endswith(".png"):
        path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def save_plotly_figure(fig: Any, name: str) -> Path:
    """Persist a Plotly figure as interactive HTML under artifacts/figures/."""
    ensure_dirs()
    stem = name[:-5] if str(name).endswith(".html") else name
    path = config.FIGURES_DIR / f"{stem}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def round_floats(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round float columns for mentoring-friendly tables."""
    out = df.copy()
    float_cols = out.select_dtypes(include="floating").columns
    if len(float_cols):
        out[float_cols] = out[float_cols].round(decimals)
    return out


def round_metrics(metrics: dict[str, float], decimals: int = 2) -> dict[str, float]:
    return {k: round(float(v), decimals) if isinstance(v, (int, float)) else v for k, v in metrics.items()}


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    round_floats(df).to_csv(path, index=False)
    return path
