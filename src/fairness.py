"""Compare how the model treats groups of applicants on the locked test set.

Usage:
    python -m src.fairness                      # feature set B, combined_fe123, default settings
    python -m src.fairness --features A --params tuned

Gender is not a model input (src/train.py excludes CODE_GENDER), but other features can
stand in for it, so the check still matters. For each group (gender, age band, thin file):
- ranking: AUC within the group
- calibration: mean predicted default rate vs the actual rate
- decisions at one shared cutoff (the riskiest 20% of all test applicants are flagged):
  share flagged, share of defaulters caught, share of good payers wrongly flagged.
Needs the --final run of src/train.py. Results go to output/results/fairness_<set>_<version>.json.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import io, train

log = logging.getLogger(__name__)

FLAG_SHARE = 0.20
AGE_BANDS = [0, 30, 40, 50, 60, 100]
AGE_LABELS = ["under 30", "30-39", "40-49", "50-59", "60+"]


def applicant_groups(holdout):
    """Group labels for every test applicant, by gender, age band and bureau record."""
    age = -holdout["DAYS_BIRTH"].to_numpy() / 365.25
    return pd.DataFrame({
        "gender": holdout["CODE_GENDER"].astype("object").map({"F": "female", "M": "male"}),
        "age": pd.cut(age, AGE_BANDS, right=False, labels=AGE_LABELS).astype("object"),
        "bureau": np.where(holdout["IS_THIN_FILE"].to_numpy() == 1, "thin file", "has bureau record"),
    })


def group_metrics(y, pred, flagged):
    good, bad = y == 0, y == 1
    return {
        "applicants": int(len(y)),
        "default_rate": float(y.mean()),
        "mean_predicted": float(pred.mean()),
        "auc": float(roc_auc_score(y, pred)),
        "flagged": float(flagged.mean()),
        "defaulters_caught": float(flagged[bad].mean()),
        "good_payers_flagged": float(flagged[good].mean()),
        "default_rate_if_flagged": float(y[flagged].mean()),
    }


def run(feature_set, data_path, param_set="default"):
    version = train.version_name(data_path, param_set)
    _, holdout = train.split_holdout(train.load_train(data_path))
    test = pd.read_csv(train.RESULTS_DIR / f"{train.final_name(feature_set, data_path, param_set)}_pred.csv")
    assert (test["SK_ID_CURR"].to_numpy() == holdout["SK_ID_CURR"].to_numpy()).all()

    y, pred = test["TARGET"].to_numpy(), test["PRED"].to_numpy()
    cutoff = float(np.quantile(pred, 1 - FLAG_SHARE))
    flagged = pred >= cutoff
    groups = applicant_groups(holdout)
    rows = [{"dimension": "all", "group": "all applicants", **group_metrics(y, pred, flagged)}]
    for dimension in groups:
        for name in groups[dimension].dropna().unique():
            mask = (groups[dimension] == name).to_numpy()
            rows.append({"dimension": dimension, "group": name, **group_metrics(y[mask], pred[mask], flagged[mask])})
    table = pd.DataFrame(rows)
    table["order"] = table["group"].map({g: i for i, g in enumerate(AGE_LABELS)}).fillna(-1)
    table = table.sort_values(["dimension", "order", "group"]).drop(columns="order")
    log.info("cutoff %.4f flags the riskiest %.0f%%\n%s", cutoff, 100 * FLAG_SHARE,
             table.round(3).to_string(index=False))

    result = {"feature_set": feature_set, "version": version, "flag_share": FLAG_SHARE,
              "cutoff": cutoff, "excluded_unknown_gender": int(groups["gender"].isna().sum()),
              "groups": table.to_dict("records")}
    (train.RESULTS_DIR / f"fairness_{feature_set}_{version}.json").write_text(json.dumps(result, indent=2))
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
