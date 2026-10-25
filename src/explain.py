"""Explain the final model's predictions with SHAP values, on the locked test set.

Usage:
    python -m src.explain                      # feature set B, combined_fe123, default settings
    python -m src.explain --params tuned

SHAP values split each applicant's score into one contribution per feature, in log-odds:
positive pushes the predicted risk up, negative pushes it down, and the contributions plus
a baseline add up exactly to the model's score. LightGBM computes them itself
(``pred_contrib=True``, exact TreeSHAP), so no extra package is needed.
Needs the --final run of src/train.py. Writes, to output/results/:
- shap_<set>_<version>.json: mean |SHAP| per feature (all and thin-file applicants), and
  two thin-file applicants explained feature by feature (one high-risk, one low-risk)
- shap_<set>_<version>_sample.csv: SHAP values and feature values of the top features
  for a random sample of applicants, for the summary chart
"""

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import io, train

log = logging.getLogger(__name__)

N_TOP = 15
N_SAMPLE = 3000
N_EXAMPLE_FEATURES = 8
# Example applicants: the thin-file applicants at these percentiles of predicted risk.
EXAMPLES = {"high risk": 0.95, "low risk": 0.10}


def shap_values(booster, X):
    """(contributions per feature, baseline), checked against the model's own score."""
    contrib = booster.predict(X, pred_contrib=True)
    raw = booster.predict(X, raw_score=True)
    assert np.allclose(contrib.sum(axis=1), raw, atol=1e-6)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index), float(contrib[0, -1])


def _value(v):
    """Feature value as plain JSON (category names as text, missing as None)."""
    if pd.isna(v):
        return None
    return v.item() if isinstance(v, np.generic) else v


def explain_example(shap_row, x_row, pred, baseline, n=N_EXAMPLE_FEATURES):
    top = shap_row.abs().sort_values(ascending=False).index[:n]
    return {"predicted_risk": float(pred), "baseline_log_odds": baseline,
            "other_features_log_odds": float(shap_row.drop(top).sum()),
            "features": [{"feature": f, "value": _value(x_row[f]), "shap": float(shap_row[f])}
                         for f in top]}


def run(feature_set, data_path, param_set="default"):
    version = train.version_name(data_path, param_set)
    name = train.final_name(feature_set, data_path, param_set)
    booster = lgb.Booster(model_file=str(train.MODELS_DIR / f"{name}.txt"))
    _, holdout = train.split_holdout(train.load_train(data_path))
    X = holdout[booster.feature_name()]
    thin = holdout["IS_THIN_FILE"].to_numpy() == 1
    shap, baseline = shap_values(booster, X)
    pred = booster.predict(X)
    log.info("SHAP values for %d applicants x %d features", *shap.shape)

    importance = pd.DataFrame({"all": shap.abs().mean(), "thin_file": shap[thin].abs().mean()})
    importance = importance.sort_values("all", ascending=False)

    examples = {}
    thin_idx = np.where(thin)[0]
    for label, q in EXAMPLES.items():
        i = thin_idx[np.argmin(np.abs(pred[thin_idx] - np.quantile(pred[thin_idx], q)))]
        examples[label] = {"SK_ID_CURR": int(holdout["SK_ID_CURR"].iloc[i]),
                           "defaulted": int(holdout["TARGET"].iloc[i]),
                           **explain_example(shap.iloc[i], X.iloc[i], pred[i], baseline)}

    top = importance.index[:N_TOP].tolist()
    categorical = [f for f in top if X[f].dtype == "category"]
    rows = np.random.default_rng(train.SEED).choice(len(X), size=min(N_SAMPLE, len(X)), replace=False)
    # numeric values as percentiles within the test set, so every feature shares one colour
    # scale; categories have no order, so they get no value (drawn grey, like missing values)
    percentile = X[top].apply(lambda c: pd.Series(np.nan, index=c.index) if c.dtype == "category"
                              else c.rank(pct=True))
    sample = pd.concat([shap.iloc[rows][top].add_prefix("SHAP_"),
                        percentile.iloc[rows].add_prefix("PCT_")], axis=1)
    io.save_dataset(sample.reset_index(drop=True), train.RESULTS_DIR / f"shap_{feature_set}_{version}_sample.csv")

    result = {"feature_set": feature_set, "version": version, "baseline_log_odds": baseline,
              "mean_abs_shap": importance.head(40).round(5).to_dict("index"),
              "categorical_features": categorical, "examples": examples}
    (train.RESULTS_DIR / f"shap_{feature_set}_{version}.json").write_text(json.dumps(result, indent=2))
    log.info("top features by mean |SHAP|:\n%s", importance.head(N_TOP).round(4).to_string())
    for label, ex in examples.items():
        log.info("%s thin-file example: predicted %.3f, defaulted %d", label, ex["predicted_risk"], ex["defaulted"])
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
