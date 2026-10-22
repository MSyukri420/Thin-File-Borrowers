"""Read the raw Home Credit CSVs and save processed datasets.

Usage:
    from src import io
    bureau = io.read_table("bureau")
    io.save_dataset(df, "dataset/processed/combined.pkl")

Cleaning and combining the tables lives in src/preprocess.py.
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"

TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
]

# bureau_balance has 27M rows; reading STATUS as category saves ~1.5 GB.
_DTYPES = {"bureau_balance": {"STATUS": "category"}}


def read_table(name, data_dir=DATA_DIR):
    """Read one raw table, e.g. ``read_table("bureau")``."""
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}, expected one of {TABLES}")
    df = pd.read_csv(Path(data_dir) / f"{name}.csv", dtype=_DTYPES.get(name))
    log.debug("read %s: %s rows", name, f"{len(df):,}")
    return df


def save_dataset(df, path):
    """Save by file extension: .parquet (needs pyarrow), .pkl, or .csv (loses dtypes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".pkl":
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)
    log.info("saved %s", path)
