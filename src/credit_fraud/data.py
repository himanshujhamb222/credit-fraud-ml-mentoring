"""Load credit-default style data and create stratified splits."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_fraud import config
from credit_fraud.io_utils import ensure_dirs


def _download_url(url: str, dest: Path) -> Path:
    ensure_dirs()
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading data from {url} -> {dest}")
    urlretrieve(url, dest)  # noqa: S310 — mentoring demo URL from config
    return dest


def _load_openml(dataset_name: str) -> pd.DataFrame:
    """Fallback loader via OpenML (may be slow the first time)."""
    import openml

    print(f"Loading OpenML dataset: {dataset_name}")
    dataset = openml.datasets.get_dataset(dataset_name)
    frame, *_ = dataset.get_data(dataset_format="dataframe")
    return frame


def resolve_data_path() -> Path | None:
    """Return a local CSV path if configured and present."""
    if config.DATA_PATH is not None and Path(config.DATA_PATH).exists():
        return Path(config.DATA_PATH)
    return None


def load_raw_dataframe() -> pd.DataFrame:
    """
    Load the working dataset.

    Priority:
    1. Existing ``DATA_PATH``
    2. Download ``DATA_URL`` into ``DATA_PATH``
    3. OpenML ``OPENML_DATASET_NAME`` (optional fallback)
    """
    ensure_dirs()
    path = resolve_data_path()
    if path is not None:
        print(f"Loading local CSV: {path}")
        df = pd.read_csv(path)
    elif config.DATA_URL:
        dest = Path(config.DATA_PATH) if config.DATA_PATH else config.DATA_DIR / "credit_default.csv"
        try:
            _download_url(config.DATA_URL, dest)
            df = pd.read_csv(dest)
        except Exception as exc:  # noqa: BLE001 — fall through to OpenML
            print(f"URL download failed ({exc}); trying OpenML fallback...")
            if not config.OPENML_DATASET_NAME:
                raise
            df = _load_openml(config.OPENML_DATASET_NAME)
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest, index=False)
    elif config.OPENML_DATASET_NAME:
        df = _load_openml(config.OPENML_DATASET_NAME)
        dest = Path(config.DATA_PATH) if config.DATA_PATH else config.DATA_DIR / "credit_default.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
    else:
        raise FileNotFoundError(
            "No data source configured. Set DATA_PATH, DATA_URL, or OPENML_DATASET_NAME in config.py"
        )

    df = _normalize_columns(df)
    if config.TARGET_COL not in df.columns:
        raise KeyError(
            f"Target column '{config.TARGET_COL}' not in data. Columns: {list(df.columns)}"
        )

    if config.MAX_ROWS is not None and len(df) > config.MAX_ROWS:
        df, _ = train_test_split(
            df,
            train_size=config.MAX_ROWS,
            stratify=df[config.TARGET_COL],
            random_state=config.RANDOM_STATE,
        )
        df = df.reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows (MAX_ROWS={config.MAX_ROWS})")

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UCI X*/Y codes to interpretable names and clean the target."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # UCI static CSV uses X1…X23, Y
    rename = {k: v for k, v in config.UCI_COLUMN_MAP.items() if k in out.columns}
    if rename:
        out = out.rename(columns=rename)

    # Drop pure ID if present
    for id_col in ("ID", "id", "Id"):
        if id_col in out.columns:
            out = out.drop(columns=[id_col])

    # Target aliases from session CSVs / older dumps
    if config.TARGET_COL not in out.columns:
        for candidate in (
            "default.payment.next.month",
            "default payment next month",
            "Class",
            "class",
            "target",
            "Y",
        ):
            if candidate in out.columns:
                out = out.rename(columns={candidate: config.TARGET_COL})
                break

    if config.TARGET_COL in out.columns:
        out[config.TARGET_COL] = pd.to_numeric(out[config.TARGET_COL], errors="coerce")
        out = out.dropna(subset=[config.TARGET_COL])
        out[config.TARGET_COL] = out[config.TARGET_COL].astype(int)

    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[config.TARGET_COL]
    x = df.drop(columns=[config.TARGET_COL])
    # Keep only numeric features for the baseline mentoring path
    x = x.select_dtypes(include="number").copy()
    return x, y


def train_test_split_stratified(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split — always use for imbalanced labels."""
    return train_test_split(
        x,
        y,
        test_size=test_size if test_size is not None else config.TEST_SIZE,
        random_state=random_state if random_state is not None else config.RANDOM_STATE,
        stratify=y,
    )


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Convenience: load → X/y → stratified split. Returns X_train, X_test, y_train, y_test, raw."""
    raw = load_raw_dataframe()
    x, y = split_xy(raw)
    x_train, x_test, y_train, y_test = train_test_split_stratified(x, y)
    return x_train, x_test, y_train, y_test, raw


def feature_glossary(columns: list[str] | None = None) -> pd.DataFrame:
    """Return a mentoring-friendly feature dictionary as a table."""
    cols = columns or list(config.FEATURE_DICTIONARY.keys())
    rows = []
    for col in cols:
        rows.append(
            {
                "feature": col,
                "description": config.FEATURE_DICTIONARY.get(col, "(see session data dictionary)"),
            }
        )
    return pd.DataFrame(rows)
