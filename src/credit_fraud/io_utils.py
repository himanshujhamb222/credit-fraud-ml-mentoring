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
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
