"""All knobs live here — no CLI args. Change values for the live session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data source (priority: existing DATA_PATH → DATA_URL download → OpenML)
# ---------------------------------------------------------------------------
# UCI "Default of Credit Card Clients" (Taiwan) — interpretable features
# (LIMIT_BAL, AGE, PAY_*, BILL_AMT_*, PAY_AMT_*, …) instead of PCA V1…V28.
# Session tip: drop your own CSV in `input/` and point DATA_PATH at it.
DATA_PATH: Path | None = PROJECT_ROOT / "data" / "credit_default.csv"
DATA_URL: str | None = "https://archive.ics.uci.edu/static/public/350/data.csv"
OPENML_DATASET_NAME: str | None = None  # URL is primary; set if you prefer OpenML
TARGET_COL = "default_next_month"
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

# Full UCI file is 30k rows — keep None to use all (fast enough for mentoring)
MAX_ROWS: int | None = None

# ---------------------------------------------------------------------------
# Optuna / models
# ---------------------------------------------------------------------------
N_TRIALS = 15  # bump for better search; keep small for session speed
OPTUNA_TIMEOUT_SEC: int | None = None
PRIMARY_METRIC = "pr_auc"  # primary for imbalanced problems

# Rank & plot: quantile bins for independent-variable capture charts
RANK_N_BINS = 10
RANK_PLOT_TOP_N = 10  # how many top features get a capture chart saved


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

# UCI raw codes → mentoring-friendly names (applied in data.load)
UCI_COLUMN_MAP: dict[str, str] = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": "default_next_month",
}

# Short explanations for mentoring / Streamlit
FEATURE_DICTIONARY: dict[str, str] = {
    "LIMIT_BAL": "Credit limit (NT dollars), individual + family/supplementary.",
    "SEX": "Gender code: 1 = male, 2 = female.",
    "EDUCATION": "1=graduate school, 2=university, 3=high school, 4=others, 5/6=unknown.",
    "MARRIAGE": "1=married, 2=single, 3=others.",
    "AGE": "Age in years.",
    "PAY_0": "Repayment status in Sep 2005 (−1=pay duly, 1=1m delay, … 9=9m+).",
    "PAY_2": "Repayment status in Aug 2005 (same coding as PAY_0).",
    "PAY_3": "Repayment status in Jul 2005.",
    "PAY_4": "Repayment status in Jun 2005.",
    "PAY_5": "Repayment status in May 2005.",
    "PAY_6": "Repayment status in Apr 2005.",
    "BILL_AMT1": "Bill statement amount in Sep 2005 (NT$).",
    "BILL_AMT2": "Bill statement amount in Aug 2005 (NT$).",
    "BILL_AMT3": "Bill statement amount in Jul 2005 (NT$).",
    "BILL_AMT4": "Bill statement amount in Jun 2005 (NT$).",
    "BILL_AMT5": "Bill statement amount in May 2005 (NT$).",
    "BILL_AMT6": "Bill statement amount in Apr 2005 (NT$).",
    "PAY_AMT1": "Amount paid in Sep 2005 (NT$).",
    "PAY_AMT2": "Amount paid in Aug 2005 (NT$).",
    "PAY_AMT3": "Amount paid in Jul 2005 (NT$).",
    "PAY_AMT4": "Amount paid in Jun 2005 (NT$).",
    "PAY_AMT5": "Amount paid in May 2005 (NT$).",
    "PAY_AMT6": "Amount paid in Apr 2005 (NT$).",
    "default_next_month": "Target: 1 = defaulted next month, 0 = did not.",
    # Engineered
    "LIMIT_BAL_log1p": "log(1 + credit limit) — stabilises heavy right skew.",
    "UTIL_RATIO_1": "BILL_AMT1 / LIMIT_BAL — recent credit utilisation.",
    "PAY_RATIO_1": "PAY_AMT1 / (BILL_AMT1 + 1) — share of last bill paid.",
    "AVG_PAY_STATUS": "Mean of PAY_0…PAY_6 — overall lateness tendency.",
    "MAX_PAY_STATUS": "Worst (max) repayment delay across the 6 months.",
}
