"""Tests for src/io.py and src/preprocess.py.

    python -m pytest test.py                  # everything (~1 min, real data included)
    python -m pytest test.py -k "not RealData" # synthetic tests only (~1 s)

The synthetic tests write tiny hand-made CSVs with known answers, so each cleaning
and aggregation rule is checked exactly. The real-data tests run the full build on
dataset/ and are skipped if the raw files are not there.

The combined dataframes are written to output/ so the results can be inspected:
    output/synthetic_combined.csv   the 3 synthetic applicants
    output/combined.csv             the full real dataset
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import io, preprocess

NAN = np.nan
SENTINEL = preprocess.DAYS_SENTINEL
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# --------------------------------------------------------------------------- #
# Synthetic dataset
#   applicant 1: train, thick file (bureau + every Home Credit history table)
#   applicant 2: train, thin file (no history at all) with every app-level anomaly
#   applicant 3: test, bureau + previous application only
# --------------------------------------------------------------------------- #

TABLES = {
    "application_train": {
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "CODE_GENDER": ["F", "XNA"],
        "NAME_FAMILY_STATUS": ["Married", "Unknown"],
        "ORGANIZATION_TYPE": ["Business Entity Type 3", "XNA"],
        "DAYS_EMPLOYED": [-1000, SENTINEL],
        "AMT_INCOME_TOTAL": [100000.0, 50000.0],
    },
    "application_test": {
        "SK_ID_CURR": [3],
        "CODE_GENDER": ["M"],
        "NAME_FAMILY_STATUS": ["Single"],
        "ORGANIZATION_TYPE": ["Self-employed"],
        "DAYS_EMPLOYED": [-500],
        "AMT_INCOME_TOTAL": [80000.0],
    },
    "bureau": {
        "SK_ID_CURR": [1, 1, 3],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": ["Active", "Closed", "Bad debt"],
        "DAYS_CREDIT": [-500, -1000, -200],
        # loan 11 "ends" ~115 years before it was opened -> NaN
        "DAYS_CREDIT_ENDDATE": [NAN, -42000, 100],
        "DAYS_ENDDATE_FACT": [NAN, -900, NAN],
        "DAYS_CREDIT_UPDATE": [-10, -900, -5],
        "CREDIT_DAY_OVERDUE": [0, 5, 0],
        "AMT_CREDIT_MAX_OVERDUE": [NAN, 200.0, NAN],
        "CNT_CREDIT_PROLONG": [0, 1, 0],
        "AMT_CREDIT_SUM": [1000.0, 3000.0, 500.0],
        # negative debt / limit -> clipped to 0
        "AMT_CREDIT_SUM_DEBT": [400.0, -50.0, 500.0],
        "AMT_CREDIT_SUM_LIMIT": [0.0, -10.0, 0.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 100.0],
    },
    "bureau_balance": {
        "SK_ID_BUREAU": [10, 10, 10, 11, 11],
        "MONTHS_BALANCE": [0, -1, -2, 0, -1],
        "STATUS": ["0", "1", "C", "X", "3"],
    },
    "previous_application": {
        "SK_ID_PREV": [100, 101, 102],
        "SK_ID_CURR": [1, 1, 3],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Canceled"],
        "AMT_APPLICATION": [1000.0, 2000.0, 500.0],
        "AMT_CREDIT": [800.0, 0.0, 0.0],
        "AMT_ANNUITY": [100.0, NAN, NAN],
        "RATE_DOWN_PAYMENT": [0.1, NAN, NAN],
        "CNT_PAYMENT": [12.0, NAN, NAN],
        "DAYS_DECISION": [-100, SENTINEL, -50],
        "SELLERPLACE_AREA": [-1, 50, 10],
        "NAME_PORTFOLIO": ["POS", "XNA", "Cash"],
    },
    "POS_CASH_balance": {
        "SK_ID_PREV": [100, 100],
        "SK_ID_CURR": [1, 1],
        "MONTHS_BALANCE": [-1, -2],
        "SK_DPD": [0, 10],
        "SK_DPD_DEF": [0, 2],
    },
    "installments_payments": {
        "SK_ID_PREV": [100, 100, 100],
        "SK_ID_CURR": [1, 1, 1],
        "DAYS_INSTALMENT": [-60, -30, -10],
        # early by 5, late by 10, never paid
        "DAYS_ENTRY_PAYMENT": [-65, -20, NAN],
        # last instalment is 0 -> PAY_RATIO must be NaN, not inf
        "AMT_INSTALMENT": [100.0, 100.0, 0.0],
        "AMT_PAYMENT": [100.0, 60.0, 50.0],
    },
    "credit_card_balance": {
        "SK_ID_PREV": [200, 200],
        "SK_ID_CURR": [1, 1],
        "MONTHS_BALANCE": [-1, -2],
        "AMT_BALANCE": [500.0, 200.0],
        # zero limit -> UTILIZATION must be NaN, not inf
        "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 0.0],
        "AMT_DRAWINGS_CURRENT": [100.0, 0.0],
        "CNT_DRAWINGS_CURRENT": [1, 0],
        "SK_DPD": [0, 3],
        "SK_DPD_DEF": [0, 0],
    },
}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("dataset")
    for name, columns in TABLES.items():
        pd.DataFrame(columns).to_csv(path / f"{name}.csv", index=False)
    return path


@pytest.fixture(scope="module")
def df(data_dir):
    return preprocess.build_dataset(data_dir).set_index("SK_ID_CURR")


# --------------------------------------------------------------------------- #
# io: reading raw tables
# --------------------------------------------------------------------------- #

def test_read_table_returns_raw_data(data_dir):
    app = io.read_table("application_train", data_dir)
    assert app.shape == (2, 7)
    # io only reads; placeholders are still there for preprocess to clean
    assert app["DAYS_EMPLOYED"].tolist() == [-1000, SENTINEL]
    assert app["CODE_GENDER"].tolist() == ["F", "XNA"]


def test_read_table_bureau_balance_status_is_category(data_dir):
    bb = io.read_table("bureau_balance", data_dir)
    assert bb["STATUS"].dtype == "category"
    assert bb["STATUS"].astype(str).tolist() == ["0", "1", "C", "X", "3"]


def test_read_table_rejects_unknown_name(data_dir):
    with pytest.raises(ValueError, match="unknown table"):
        io.read_table("bureau.csv", data_dir)


def test_read_table_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_table("bureau", tmp_path)


# --------------------------------------------------------------------------- #
# preprocess helpers
# --------------------------------------------------------------------------- #

def test_aggregates_work_on_in_memory_frames():
    # preprocess needs no files: any dataframe with the right columns works
    out = preprocess.aggregate_pos_cash(pd.DataFrame(TABLES["POS_CASH_balance"]))
    assert out.index.tolist() == [1]
    assert out.loc[1, "POS_SK_DPD_MAX"] == 10


def test_aggregate_summarises_every_column():
    frame = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [7, 8, 9],
        "AMT": [10.0, 30.0, NAN],
        "KIND": ["Bad debt", "Closed", "Closed"],
    })
    out = preprocess._aggregate(frame, "T")
    # ids are not summarised; text values become safe column names
    assert out.columns.tolist() == [
        "T_COUNT", "T_AMT_MEAN", "T_AMT_MIN", "T_AMT_MAX", "T_AMT_SUM",
        "T_KIND_BAD_DEBT_SHARE", "T_KIND_CLOSED_SHARE",
    ]
    assert out.loc[1].tolist() == [2, 20.0, 10.0, 30.0, 40.0, 0.5, 0.5]
    # a sum over only missing values is unknown, not 0
    assert np.isnan(out.loc[2, "T_AMT_SUM"])


# --------------------------------------------------------------------------- #
# Combined frame shape
# --------------------------------------------------------------------------- #

def test_one_row_per_applicant(df):
    assert sorted(df.index) == [1, 2, 3]


def test_train_and_test_are_stacked(df):
    assert df["IS_TRAIN"].tolist() == [True, True, False]
    assert df.loc[[1, 2], "TARGET"].tolist() == [0, 1]
    assert np.isnan(df.loc[3, "TARGET"])


def test_dtypes_are_compact(df):
    assert "float64" not in df.dtypes.astype(str).tolist()
    assert "object" not in df.dtypes.astype(str).tolist()
    assert df["CODE_GENDER"].dtype == "category"


def test_only_original_columns_are_used(df):
    # every column is an application column, a label, or a plain summary of a raw column
    tables = {"BUREAU": "bureau", "BB": "bureau_balance", "PREV": "previous_application",
              "POS": "POS_CASH_balance", "INST": "installments_payments", "CC": "credit_card_balance"}
    allowed = set(TABLES["application_train"]) | {"IS_TRAIN", "IS_THIN_FILE"}
    shares = []
    for prefix, table in tables.items():
        raw = pd.DataFrame(TABLES[table])
        allowed.add(f"{prefix}_COUNT")
        for col in raw.columns:
            if col.startswith("SK_ID_"):
                continue
            if raw[col].dtype == object:
                shares.append(f"{prefix}_{col}_")
            else:
                allowed.update(f"{prefix}_{col}_{stat.upper()}" for stat in preprocess.NUM_STATS)
    for col in df.reset_index().columns:
        is_share = col.endswith("_SHARE") and col.startswith(tuple(shares))
        assert col in allowed or is_share, col


# --------------------------------------------------------------------------- #
# Application cleaning
# --------------------------------------------------------------------------- #

def test_days_employed_sentinel_becomes_nan(df):
    assert np.isnan(df.loc[2, "DAYS_EMPLOYED"])
    assert df.loc[1, "DAYS_EMPLOYED"] == -1000


def test_placeholder_categories_become_nan(df):
    for col in ["CODE_GENDER", "NAME_FAMILY_STATUS", "ORGANIZATION_TYPE"]:
        assert pd.isna(df.loc[2, col]), col
        assert "XNA" not in df[col].cat.categories
    assert "Unknown" not in df["NAME_FAMILY_STATUS"].cat.categories


# --------------------------------------------------------------------------- #
# Bureau + bureau_balance
# --------------------------------------------------------------------------- #

def test_bureau_counts_and_shares(df):
    assert df.loc[1, "BUREAU_COUNT"] == 2
    assert df.loc[1, "BUREAU_CREDIT_ACTIVE_ACTIVE_SHARE"] == pytest.approx(0.5)
    assert df.loc[1, "BUREAU_CREDIT_ACTIVE_CLOSED_SHARE"] == pytest.approx(0.5)
    assert df.loc[3, "BUREAU_CREDIT_ACTIVE_BAD_DEBT_SHARE"] == 1


def test_bureau_impossible_end_date_dropped(df):
    # loan 10 has no end date, loan 11's -42000 is before it opened -> nothing left
    assert np.isnan(df.loc[1, "BUREAU_DAYS_CREDIT_ENDDATE_MAX"])
    assert np.isnan(df.loc[1, "BUREAU_DAYS_CREDIT_ENDDATE_SUM"])


def test_bureau_negative_debt_clipped(df):
    assert df.loc[1, "BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] == pytest.approx(400.0)
    assert df.loc[1, "BUREAU_AMT_CREDIT_SUM_LIMIT_MIN"] == 0


def test_bureau_numeric_stats(df):
    row = df.loc[1]
    assert row["BUREAU_DAYS_CREDIT_MIN"] == -1000
    assert row["BUREAU_DAYS_CREDIT_MAX"] == -500
    assert row["BUREAU_DAYS_CREDIT_MEAN"] == -750
    assert row["BUREAU_CREDIT_DAY_OVERDUE_MAX"] == 5
    assert row["BUREAU_AMT_CREDIT_SUM_SUM"] == pytest.approx(4000.0)


def test_bureau_balance(df):
    # applicant 1 owns loans 10 and 11: 5 monthly rows, statuses 0, 1, C, X, 3
    row = df.loc[1]
    assert row["BB_COUNT"] == 5
    assert row["BB_MONTHS_BALANCE_MIN"] == -2
    assert row["BB_MONTHS_BALANCE_MEAN"] == pytest.approx(-0.8)
    for status in ["0", "1", "3", "C", "X"]:
        assert row[f"BB_STATUS_{status}_SHARE"] == pytest.approx(0.2), status
    # applicant 3's bureau loan has no monthly history
    assert df.loc[3, "BB_COUNT"] == 0


# --------------------------------------------------------------------------- #
# Home Credit history tables
# --------------------------------------------------------------------------- #

def test_previous_application(df):
    row = df.loc[1]
    assert row["PREV_COUNT"] == 2
    assert row["PREV_NAME_CONTRACT_STATUS_APPROVED_SHARE"] == pytest.approx(0.5)
    assert row["PREV_NAME_CONTRACT_STATUS_REFUSED_SHARE"] == pytest.approx(0.5)
    assert row["PREV_AMT_CREDIT_MEAN"] == pytest.approx(400.0)
    # 365243 in DAYS_DECISION is ignored
    assert row["PREV_DAYS_DECISION_MIN"] == -100
    assert row["PREV_DAYS_DECISION_MAX"] == -100
    # SELLERPLACE_AREA = -1 is missing
    assert row["PREV_SELLERPLACE_AREA_MIN"] == 50
    # "XNA" is missing: it gets no column and counts as "not POS"
    assert row["PREV_NAME_PORTFOLIO_POS_SHARE"] == pytest.approx(0.5)
    assert not any("XNA" in c for c in df.columns)
    assert df.loc[3, "PREV_NAME_CONTRACT_STATUS_CANCELED_SHARE"] == 1


def test_pos_cash(df):
    row = df.loc[1]
    assert row["POS_COUNT"] == 2
    assert row["POS_SK_DPD_MAX"] == 10
    assert row["POS_SK_DPD_MEAN"] == pytest.approx(5.0)
    assert row["POS_MONTHS_BALANCE_MIN"] == -2


def test_installments(df):
    row = df.loc[1]
    assert row["INST_COUNT"] == 3
    assert row["INST_AMT_PAYMENT_SUM"] == pytest.approx(210.0)
    assert row["INST_AMT_INSTALMENT_MIN"] == 0
    # the never-paid instalment has no payment date and is skipped
    assert row["INST_DAYS_ENTRY_PAYMENT_MEAN"] == pytest.approx(-42.5)


def test_credit_card(df):
    row = df.loc[1]
    assert row["CC_COUNT"] == 2
    assert row["CC_AMT_BALANCE_MAX"] == pytest.approx(500.0)
    assert row["CC_AMT_CREDIT_LIMIT_ACTUAL_MIN"] == 0
    assert row["CC_SK_DPD_MAX"] == 3


def test_no_infinite_values(df):
    numeric = df.select_dtypes("number")
    assert not np.isinf(numeric.to_numpy(dtype="float64")).any()


# --------------------------------------------------------------------------- #
# Thin-file handling
# --------------------------------------------------------------------------- #

def test_thin_file_label(df):
    assert df["IS_THIN_FILE"].tolist() == [0, 1, 0]


def test_counts_are_zero_not_nan_without_history(df):
    count_cols = [c for c in df.columns if preprocess._COUNT_COL.search(c)]
    assert count_cols == ["BUREAU_COUNT", "BB_COUNT", "PREV_COUNT", "POS_COUNT", "INST_COUNT", "CC_COUNT"]
    assert (df.loc[2, count_cols] == 0).all()


def test_summaries_stay_nan_without_history(df):
    for col in ["BUREAU_DAYS_CREDIT_MIN", "BUREAU_AMT_CREDIT_SUM_SUM", "PREV_AMT_CREDIT_MEAN",
                "PREV_NAME_CONTRACT_STATUS_APPROVED_SHARE", "INST_AMT_PAYMENT_SUM", "CC_SK_DPD_MAX"]:
        assert np.isnan(df.loc[2, col]), col


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def test_save_pickle_round_trip(df, tmp_path):
    path = tmp_path / "out" / "combined.pkl"
    io.save_dataset(df, path)
    pd.testing.assert_frame_equal(pd.read_pickle(path), df)


def test_write_synthetic_output_csv(df):
    path = OUTPUT_DIR / "synthetic_combined.csv"
    io.save_dataset(df.reset_index(), path)
    back = pd.read_csv(path)
    assert back.shape == df.reset_index().shape
    assert back["SK_ID_CURR"].tolist() == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Real data (skipped if dataset/ is missing)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not (io.DATA_DIR / "application_train.csv").exists(),
                    reason="raw Home Credit CSVs not found in dataset/")
class TestRealData:

    @pytest.fixture(scope="class")
    def real(self):
        return preprocess.build_dataset()

    @pytest.fixture(scope="class")
    def raw_ids(self):
        read = lambda name: pd.read_csv(io.DATA_DIR / f"{name}.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"]
        return {name: read(name) for name in ["application_train", "application_test", "bureau"]}

    def test_rows_match_raw_files(self, real, raw_ids):
        assert real["SK_ID_CURR"].is_unique
        assert real["IS_TRAIN"].sum() == len(raw_ids["application_train"])
        assert (~real["IS_TRAIN"]).sum() == len(raw_ids["application_test"])

    def test_target(self, real):
        train = real[real["IS_TRAIN"]]
        assert set(train["TARGET"].unique()) == {0, 1}
        assert train["TARGET"].mean() == pytest.approx(0.0807, abs=1e-3)
        assert real.loc[~real["IS_TRAIN"], "TARGET"].isna().all()

    def test_thin_file_matches_bureau(self, real, raw_ids):
        in_bureau = real["SK_ID_CURR"].isin(raw_ids["bureau"])
        assert (real["IS_THIN_FILE"] == (~in_bureau).astype("int8")).all()

    def test_no_sentinels_or_placeholders_left(self, real):
        # a sum of real dates is not a placeholder, so sums are skipped
        days = [c for c in real.columns if "DAYS" in c and not c.endswith("_SUM")]
        assert not (real[days] == SENTINEL).any().any()
        for col in real.select_dtypes("category").columns:
            assert "XNA" not in real[col].cat.categories, col

    def test_column_names_are_unique_and_safe(self, real):
        # LightGBM rejects special characters in feature names
        assert real.columns.is_unique
        assert all(re.fullmatch(r"[A-Z0-9_]+", c) for c in real.columns)

    def test_no_infinite_values(self, real):
        assert not np.isinf(real.select_dtypes("number").to_numpy(dtype="float64")).any()

    def test_counts_have_no_nan(self, real):
        count_cols = [c for c in real.columns if preprocess._COUNT_COL.search(c)]
        assert not real[count_cols].isna().any().any()

    def test_write_output_csv(self, real):
        path = OUTPUT_DIR / "combined.csv"
        io.save_dataset(real, path)
        back = pd.read_csv(path, usecols=["SK_ID_CURR"])
        assert len(back) == len(real)
