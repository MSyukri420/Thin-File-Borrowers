"""Clean the Home Credit tables and combine them into one applicant-level dataframe.

Usage:
    from src.preprocess import build_dataset
    df = build_dataset()             # train + test, one row per SK_ID_CURR

    python -m src.preprocess         # build and save to dataset/processed/combined.pkl

Train and test rows are stacked together (``IS_TRAIN`` tells them apart, ``TARGET``
is NaN for test). Only row-level cleaning is done here: anything fitted on the data
(outlier caps, imputation, encoding) belongs after the train/validation split.

No new features are invented. Application columns are kept as they are, and every
original column of the history tables is summarised per applicant with the same
plain rule (see ``_aggregate``):
    PREFIX_COUNT                  number of rows the applicant has in the table
    PREFIX_COLUMN_MEAN/MIN/MAX/SUM  for each numeric column
    PREFIX_COLUMN_VALUE_SHARE     for each text column, share of rows with that value
The only added columns are ``IS_TRAIN`` and ``IS_THIN_FILE`` (no bureau record), which
label rows for splitting and evaluation and are not model features.
"""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import io

log = logging.getLogger(__name__)

# Home Credit encodes "no date" as 365243 days (~1000 years) in the DAYS_* columns.
DAYS_SENTINEL = 365243

# Statistics computed for every numeric column of the history tables.
NUM_STATS = ["mean", "min", "max", "sum"]

# Per-applicant row counts are 0, not missing, when the applicant has no rows in a table.
_COUNT_COL = re.compile(r"^(BUREAU|BB|PREV|POS|INST|CC)_COUNT$")


def _name(value):
    """Turn a category value into a safe column-name part: "Bad debt" -> "BAD_DEBT"."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").upper()


def _aggregate(df, prefix, by="SK_ID_CURR"):
    """Summarise every original column of ``df`` per ``by`` value.

    ID columns (SK_ID_*) are skipped. Sums of all-missing groups stay NaN rather than 0.
    """
    ids = [c for c in df.columns if c.startswith("SK_ID_")]
    cats = df.select_dtypes(["object", "category"]).columns.tolist()
    nums = [c for c in df.columns if c not in ids and c not in cats]

    groups = df.groupby(by)
    parts = [groups.size().rename("COUNT").to_frame()]
    for stat in NUM_STATS:
        if stat == "sum":
            out = groups[nums].sum(min_count=1)
        else:
            out = groups[nums].agg(stat)
        parts.append(out.add_suffix(f"_{stat}"))
    for col in cats:
        dummies = pd.get_dummies(df[col], dtype="uint8")
        dummies.columns = [f"{col}_{_name(value)}_SHARE" for value in dummies.columns]
        parts.append(dummies.groupby(df[by]).mean())

    out = pd.concat(parts, axis=1)
    out.columns = [f"{prefix}_{col}".upper() for col in out.columns]
    return out


def _replace_placeholders(df):
    """365243 in DAYS_* columns and "XNA" in text columns both mean missing."""
    days_cols = [c for c in df.columns if c.startswith("DAYS_")]
    df[days_cols] = df[days_cols].replace(DAYS_SENTINEL, np.nan)
    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].replace("XNA", np.nan)


def clean_application(train, test):
    """Stack train and test applications and fix their placeholder values."""
    train["IS_TRAIN"] = True
    test["IS_TRAIN"] = False
    app = pd.concat([train, test], ignore_index=True)

    # DAYS_EMPLOYED = 365243 marks pensioners / unemployed (same rows as ORGANIZATION_TYPE = XNA).
    _replace_placeholders(app)
    app["NAME_FAMILY_STATUS"] = app["NAME_FAMILY_STATUS"].replace("Unknown", np.nan)
    return app


def aggregate_bureau(bureau, bureau_balance):
    """Credit bureau loans at other lenders (BUREAU_*) and their monthly statuses (BB_*)."""
    _replace_placeholders(bureau)
    # End/update dates before the loan was opened are impossible (some are ~115 years ago).
    for col in ["DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"]:
        bureau.loc[bureau[col] < bureau["DAYS_CREDIT"], col] = np.nan
    # Negative debt / limit are overpayments; treat as zero outstanding.
    for col in ["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT"]:
        bureau[col] = bureau[col].clip(lower=0)

    # bureau_balance has no SK_ID_CURR; look it up through the bureau loan id.
    # STATUS: 0-5 = days-past-due bucket, C = closed, X = unknown.
    owner = bureau.set_index("SK_ID_BUREAU")["SK_ID_CURR"]
    bureau_balance["SK_ID_CURR"] = bureau_balance["SK_ID_BUREAU"].map(owner).fillna(-1).astype("int64")
    bb = _aggregate(bureau_balance, "BB").drop(index=-1, errors="ignore")

    return _aggregate(bureau, "BUREAU").join(bb)


def aggregate_previous_application(prev):
    """Earlier applications to Home Credit."""
    _replace_placeholders(prev)
    prev["SELLERPLACE_AREA"] = prev["SELLERPLACE_AREA"].replace(-1, np.nan)
    return _aggregate(prev, "PREV")


def aggregate_pos_cash(pos):
    """Monthly snapshots of previous point-of-sale and cash loans."""
    _replace_placeholders(pos)
    return _aggregate(pos, "POS")


def aggregate_installments(inst):
    """Each scheduled repayment of previous Home Credit loans vs what was actually paid."""
    _replace_placeholders(inst)
    return _aggregate(inst, "INST")


def aggregate_credit_card(cc):
    """Monthly snapshots of previous Home Credit credit cards."""
    _replace_placeholders(cc)
    return _aggregate(cc, "CC")


# Each aggregate function and the raw tables it takes, in argument order.
AGGREGATES = [
    (aggregate_bureau, ["bureau", "bureau_balance"]),
    (aggregate_previous_application, ["previous_application"]),
    (aggregate_pos_cash, ["POS_CASH_balance"]),
    (aggregate_installments, ["installments_payments"]),
    (aggregate_credit_card, ["credit_card_balance"]),
]


def finalize(df):
    """Fill counts, add the thin-file label and shrink dtypes once all tables are joined."""
    count_cols = [c for c in df.columns if _COUNT_COL.search(c)]
    df[count_cols] = df[count_cols].fillna(0)

    # Thin file = no credit bureau record at any other lender. A label, not a feature.
    df["IS_THIN_FILE"] = df["BUREAU_COUNT"].eq(0).astype("int8")

    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].astype("category")
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")
    for col in df.select_dtypes("int64").columns:
        if col != "SK_ID_CURR":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def build_dataset(data_dir=io.DATA_DIR):
    """Read, clean and combine every table into one row per applicant (train + test)."""
    start = time.perf_counter()

    df = clean_application(io.read_table("application_train", data_dir),
                           io.read_table("application_test", data_dir))
    log.info("application: %s rows", f"{len(df):,}")

    # Tables are read one step at a time so only one large raw table is in memory.
    for aggregate, tables in AGGREGATES:
        agg = aggregate(*[io.read_table(name, data_dir) for name in tables])
        df = df.join(agg, on="SK_ID_CURR")
        log.info("%s: +%d columns (%.0fs)", aggregate.__name__, agg.shape[1],
                 time.perf_counter() - start)

    df = finalize(df)
    log.info("combined: %s rows x %d columns, %.0f MB (%.0fs)", f"{len(df):,}", df.shape[1],
             df.memory_usage(deep=True).sum() / 1e6, time.perf_counter() - start)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=io.DATA_DIR)
    parser.add_argument("--out", type=Path, default=io.DATA_DIR / "processed" / "combined.pkl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    io.save_dataset(build_dataset(args.data_dir), args.out)


if __name__ == "__main__":
    main()
