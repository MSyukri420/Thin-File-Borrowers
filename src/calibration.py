"""Check whether predicted default probabilities match actual default rates.

Usage:
    python -m src.calibration                      # feature set B, combined_fe123, default settings
    python -m src.calibration --features A --params tuned

Needs the cross-validation run and the --final run of src/train.py for the same version.
Corrections are fitted on the development set's out-of-fold predictions only (isotonic
regression, Platt scaling, and Platt scaling fitted separately for thin-file applicants)
and then checked on the locked test set, so the test set never shapes them. Results go to output/results/calibration_<set>_<version>.json.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from . import io, train

log = logging.getLogger(__name__)

N_BINS = 10


def calibration_table(y, pred, n_bins=N_BINS):
    """Equal-size bins of predicted risk: mean predicted vs actual default rate."""
    bins = pd.qcut(pd.Series(pred).rank(method="first"), n_bins, labels=False)
    table = (pd.DataFrame({"bin": bins, "pred": pred, "actual": y})
             .groupby("bin").agg(predicted=("pred", "mean"), actual=("actual", "mean"),
                                 applicants=("actual", "size")))
    return table.reset_index()


def expected_calibration_error(y, pred, n_bins=N_BINS):
    """Average gap between predicted and actual default rate, weighted by bin size."""
    t = calibration_table(y, pred, n_bins)
    return float((t["applicants"] * (t["predicted"] - t["actual"]).abs()).sum() / t["applicants"].sum())


def scores(y, pred):
    return {"brier": float(brier_score_loss(y, pred)), "log_loss": float(log_loss(y, pred)),
            "ece": expected_calibration_error(y, pred), "mean_predicted": float(pred.mean()),
            "actual_rate": float(y.mean())}


def fit_corrections(y_dev, pred_dev):
    """Isotonic and Platt corrections, fitted on development-set predictions."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pred_dev, y_dev)
    platt = LogisticRegression().fit(logit(np.clip(pred_dev, 1e-6, 1 - 1e-6)).reshape(-1, 1), y_dev)
    return {
        "isotonic": iso.predict,
        "platt": lambda p: platt.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1],
    }


def run(feature_set, data_path, param_set="default"):
    version = train.version_name(data_path, param_set)
    oof = pd.read_csv(train.RESULTS_DIR / f"lightgbm_{feature_set}_{version}_oof.csv")
    test = pd.read_csv(train.RESULTS_DIR / f"{train.final_name(feature_set, data_path, param_set)}_pred.csv")
    corrections = fit_corrections(oof["TARGET"].to_numpy(), oof["PRED"].to_numpy())

    y, raw = test["TARGET"].to_numpy(), test["PRED"].to_numpy()
    preds = {"raw": raw, **{name: fix(raw) for name, fix in corrections.items()}}
    # Platt scaling fitted separately for thin-file applicants and everyone else
    by_group = np.zeros(len(raw))
    for is_thin in (0, 1):
        dev_rows, test_rows = oof["IS_THIN_FILE"] == is_thin, test["IS_THIN_FILE"].to_numpy() == is_thin
        fix = fit_corrections(oof.loc[dev_rows, "TARGET"].to_numpy(), oof.loc[dev_rows, "PRED"].to_numpy())
        by_group[test_rows] = fix["platt"](raw[test_rows])
    preds["platt_by_group"] = by_group
    groups = {"overall": np.ones(len(y), bool), "thin_file": test["IS_THIN_FILE"].to_numpy() == 1}
    result = {"feature_set": feature_set, "version": version, "scores": {}, "bins": []}
    for group, mask in groups.items():
        result["scores"][group] = {name: scores(y[mask], p[mask]) for name, p in preds.items()}
        for name, p in preds.items():
            for row in calibration_table(y[mask], p[mask]).to_dict("records"):
                result["bins"].append({"group": group, "method": name, **row})
        for name, s in result["scores"][group].items():
            log.info("%-9s %-14s ECE %.4f, Brier %.5f, mean predicted %.4f vs actual %.4f",
                     group, name, s["ece"], s["brier"], s["mean_predicted"], s["actual_rate"])

    out = train.RESULTS_DIR / f"calibration_{feature_set}_{version}.json"
    out.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", choices=sorted(train.FEATURE_SETS), default="B")
    parser.add_argument("--data", type=Path, default=io.DATA_DIR / "processed" / "combined_fe123.pkl")
    parser.add_argument("--params", choices=sorted(train.PARAM_SETS), default="default")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.features, args.data, args.params)


if __name__ == "__main__":
    main()
