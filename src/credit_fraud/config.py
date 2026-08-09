"""All knobs live here — no CLI args. Change values for the live session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data source (priority: existing DATA_PATH → DATA_URL download → OpenML)
# ---------------------------------------------------------------------------
# Session tip: put a CSV in `input/` and set DATA_PATH to that file.
DATA_PATH: Path | None = PROJECT_ROOT / "data" / "creditcard.csv"
DATA_URL: str | None = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)
# Fallback if URL/path fail: OpenML dataset name (creditcard = Kaggle-style fraud)
OPENML_DATASET_NAME: str | None = "CreditCardFraudDetection"
TARGET_COL = "Class"
POSITIVE_LABEL = 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
IMPORTANCES_PATH = ARTIFACTS_DIR / "feature_importances.csv"
RANKING_PATH = ARTIFACTS_DIR / "feature_ranking.csv"
EDA_SUMMARY_PATH = ARTIFACTS_DIR / "eda_summary.json"

# ---------------------------------------------------------------------------
# Split / reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

# Optional row subsample for faster mentoring demos (None = use all rows)
# Creditcard is large; 50k keeps imbalance signal while staying snappy.
MAX_ROWS: int | None = 50_000

# ---------------------------------------------------------------------------
# Optuna / models
# ---------------------------------------------------------------------------
N_TRIALS = 15  # bump for better search; keep small for session speed
OPTUNA_TIMEOUT_SEC: int | None = None
PRIMARY_METRIC = "pr_auc"  # primary for imbalanced problems


@dataclass(frozen=True)
class Paths:
    """Convenience bundle for artifact locations."""

    root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    artifacts: Path = ARTIFACTS_DIR
    figures: Path = FIGURES_DIR
    model: Path = MODEL_PATH
    metrics: Path = METRICS_PATH
    importances: Path = IMPORTANCES_PATH
    ranking: Path = RANKING_PATH
    eda_summary: Path = EDA_SUMMARY_PATH


PATHS = Paths()
