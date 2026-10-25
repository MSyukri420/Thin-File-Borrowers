"""Train LightGBM with 5-fold cross-validation and report AUC overall and for thin-file applicants.

Usage:
    python -m src.train                      # feature set A (all features)
    python -m src.train --features B         # no bureau information
    python -m src.train --features C         # no bureau information and no outside scores
    python -m src.train --data dataset/processed/combined.pkl      # raw features only
    python -m src.train --features B --final # score the locked test set once
    python -m src.train --features B --params tuned   # settings found by src/tune.py

Only labelled (train) applicants are used. 15% of them are locked away as a final test
set (``split_holdout``); cross-validation runs on the other 85%. ``--final`` trains one
model on that 85% and scores the locked 15% once. It reports results and is never
used to choose between versions.
Every run uses the same holdout and the same stratified folds (fixed seed), so results
from different feature sets and dataset versions can be compared directly.
Results go to output/results/: a JSON summary, the out-of-fold predictions, and one
row per run appended to log.csv. Final scores go to final_*.json and final.csv, with
the test-set predictions (final_*_pred.csv) and the model (output/models/).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import io

log = logging.getLogger(__name__)

SEED = 42
N_FOLDS = 5
HOLDOUT_SIZE = 0.15
N_BOOTSTRAP = 1000
RESULTS_DIR = Path(__file__).resolve().parents[1] / "output" / "results"
MODELS_DIR = RESULTS_DIR.parent / "models"

# Not features: an ID, the answer, and labels used to split the data and the results.
NON_FEATURES = ["SK_ID_CURR", "TARGET", "IS_TRAIN", "IS_THIN_FILE"]
# Excluded by choice: gender is a legally restricted basis for lending decisions in many places.
EXCLUDED = ["CODE_GENDER"]

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 5000,       # upper limit; early stopping picks the real number
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Found by src/tune.py (Optuna, 40 trials, feature set B on combined_fe123, development set
# only), rounded. Fewer columns per tree and much stronger regularisation than the defaults.
TUNED_PARAMS = {
    **PARAMS,
    "num_leaves": 36,
    "min_child_samples": 79,
    "subsample": 0.95,
    "colsample_bytree": 0.11,
    "reg_lambda": 92.0,
    "reg_alpha": 0.26,
    "min_split_gain": 0.58,
}

PARAM_SETS = {"default": PARAMS, "tuned": TUNED_PARAMS}


def version_name(data_path, param_set="default"):
    """Name used in result files: the dataset version, plus "_tuned" for tuned settings."""
    return data_path.stem if param_set == "default" else f"{data_path.stem}_{param_set}"


def final_name(feature_set, data_path, param_set="default"):
    """Stem of the files written by ``final`` (summary, test predictions, model)."""
    return f"final_{feature_set}_{version_name(data_path, param_set)}"


def _is_bureau_info(col):
    """Columns that only exist for applicants with a credit bureau record.

    EXT_SOURCE_3 and the AMT_REQ_CREDIT_BUREAU_* enquiry counts are missing for
    ~97% / ~94% of thin-file applicants and almost never otherwise. Engineered
    EXT_SOURCE_FE_* features include EXT_SOURCE_3, so they count too.
    """
    return (col.startswith(("BUREAU_", "BB_", "AMT_REQ_CREDIT_BUREAU_", "EXT_SOURCE_FE_"))
            or col == "EXT_SOURCE_3")


FEATURE_SETS = {
    "A": lambda col: True,
    "B": lambda col: not _is_bureau_info(col),
    "C": lambda col: not _is_bureau_info(col) and not col.startswith("EXT_SOURCE_"),
}


def select_features(columns, feature_set):
    """Feature columns for ``feature_set`` ("A", "B" or "C"), in their original order."""
    keep = FEATURE_SETS[feature_set]
    return [c for c in columns if c not in NON_FEATURES + EXCLUDED and keep(c)]


def load_train(path):
    """Labelled applicants from the combined dataset built by src/preprocess.py."""
    df = pd.read_pickle(path)
    train = df[df["IS_TRAIN"]].reset_index(drop=True)
    train["TARGET"] = train["TARGET"].astype("int8")
    return train


def split_holdout(train, size=HOLDOUT_SIZE, seed=SEED):
    """Split labelled applicants into (development, locked final test set).

    Rows are sorted by SK_ID_CURR first, so the same applicants are held out on every
    run and for every dataset version built from the same raw files.
    """
    train = train.sort_values("SK_ID_CURR", ignore_index=True)
    dev_idx, holdout_idx = train_test_split(np.arange(len(train)), test_size=size,
                                            stratify=train["TARGET"], random_state=seed)
    return (train.iloc[np.sort(dev_idx)].reset_index(drop=True),
            train.iloc[np.sort(holdout_idx)].reset_index(drop=True))


def make_folds(y, n_folds=N_FOLDS, seed=SEED):
    """Fold number (0..n_folds-1) for every row; each fold keeps the overall default rate."""
    folds = np.zeros(len(y), dtype="int8")
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold
    return folds


def fit_model(X, y, train_idx, params=PARAMS):
    """Fit on ``train_idx``, early-stopping on a stratified 10% of it."""
    fit_idx, stop_idx = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx],
                                         random_state=params["random_state"])
    model = lgb.LGBMClassifier(**params)
    model.fit(X.iloc[fit_idx], y[fit_idx],
              eval_X=X.iloc[stop_idx], eval_y=y[stop_idx], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def fit_fold(X, y, folds, fold, params=PARAMS):
    """Train on every fold but ``fold``; return (rows of ``fold``, their predictions, model)."""
    valid_idx = np.where(folds == fold)[0]
    model = fit_model(X, y, np.where(folds != fold)[0], params)
    return valid_idx, model.predict_proba(X.iloc[valid_idx])[:, 1], model


def cross_validate(X, y, folds, params=PARAMS):
    """Out-of-fold predictions: each row is predicted by a model that never saw it.

    Early stopping uses 10% of each training part, never the fold being scored,
    so the fold's AUC stays an honest estimate.
    """
    oof = np.zeros(len(y))
    n_trees = []
    importance = pd.Series(0.0, index=X.columns)
    for fold in np.unique(folds):
        valid_idx, pred, model = fit_fold(X, y, folds, fold, params)
        oof[valid_idx] = pred
        n_trees.append(model.best_iteration_)
        importance += pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
        log.info("fold %d: AUC %.4f (%d trees)", fold, roc_auc_score(y[valid_idx], oof[valid_idx]),
                 model.best_iteration_)
    return oof, n_trees, importance / len(n_trees)


def evaluate(y, oof, folds, thin):
    """AUC overall and for thin-file applicants: pooled, plus the spread across folds."""
    result = {}
    for name, mask in [("overall", np.ones(len(y), bool)), ("thin_file", thin == 1),
                       ("has_bureau", thin == 0)]:
        per_fold = [roc_auc_score(y[mask & (folds == f)], oof[mask & (folds == f)])
                    for f in np.unique(folds)]
        result[name] = {
            "auc": roc_auc_score(y[mask], oof[mask]),
            "auc_fold_std": float(np.std(per_fold)),
            "applicants": int(mask.sum()),
            "default_rate": float(y[mask].mean()),
        }
    return result


def bootstrap_auc(y, pred, n=N_BOOTSTRAP, seed=SEED):
    """95% interval for AUC from resampling applicants with replacement."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], pred[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def _append_log(summary):
    """One row per run in output/results/log.csv, to compare runs side by side."""
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "data": summary["data"],
           "feature_set": summary["feature_set"], "params": summary["param_set"],
           "n_features": summary["n_features"]}
    for group in ["overall", "has_bureau", "thin_file"]:
        row[f"auc_{group}"] = round(summary[group]["auc"], 4)
        row[f"std_{group}"] = round(summary[group]["auc_fold_std"], 4)
    path = RESULTS_DIR / "log.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run(feature_set, data_path, param_set="default"):
    start = time.perf_counter()
    params = PARAM_SETS[param_set]
    dev, holdout = split_holdout(load_train(data_path))
    features = select_features(dev.columns, feature_set)
    X, y = dev[features], dev["TARGET"].to_numpy()
    thin = dev["IS_THIN_FILE"].to_numpy()
    folds = make_folds(y)
    log.info("%s, feature set %s: %d applicants (%d held out), %d features", data_path.stem,
             feature_set, len(X), len(holdout), len(features))

    oof, n_trees, importance = cross_validate(X, y, folds, params)
    summary = {
        "model": "lightgbm",
        "data": data_path.stem,
        "feature_set": feature_set,
        "param_set": param_set,
        "n_features": len(features),
        "holdout_applicants": len(holdout),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "trees_per_fold": n_trees,
        "params": params,
        **evaluate(y, oof, folds, thin),
        "top_features_by_gain": importance.sort_values(ascending=False).head(25).round(1).to_dict(),
        "minutes": round((time.perf_counter() - start) / 60, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"lightgbm_{feature_set}_{version_name(data_path, param_set)}"
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
    io.save_dataset(pd.DataFrame({"SK_ID_CURR": dev["SK_ID_CURR"], "FOLD": folds,
                                  "TARGET": y, "IS_THIN_FILE": thin, "PRED": oof}),
                    RESULTS_DIR / f"{name}_oof.csv")
    _append_log(summary)
    for group in ["overall", "thin_file", "has_bureau"]:
        r = summary[group]
        log.info("%-10s AUC %.4f (fold std %.4f), %d applicants, %.1f%% default", group,
                 r["auc"], r["auc_fold_std"], r["applicants"], 100 * r["default_rate"])
    return summary


def final(feature_set, data_path, param_set="default"):
    """Train on the whole development set and score the locked test set once."""
    start = time.perf_counter()
    params = PARAM_SETS[param_set]
    dev, holdout = split_holdout(load_train(data_path))
    features = select_features(dev.columns, feature_set)
    model = fit_model(dev[features], dev["TARGET"].to_numpy(), np.arange(len(dev)), params)

    y, thin = holdout["TARGET"].to_numpy(), holdout["IS_THIN_FILE"].to_numpy()
    pred = model.predict_proba(holdout[features])[:, 1]
    summary = {"model": "lightgbm", "data": data_path.stem, "feature_set": feature_set,
               "param_set": param_set, "n_features": len(features),
               "trees": model.best_iteration_, "seed": SEED, "params": params}
    for name, mask in [("overall", np.ones(len(y), bool)), ("thin_file", thin == 1),
                       ("has_bureau", thin == 0)]:
        summary[name] = {"auc": roc_auc_score(y[mask], pred[mask]),
                         "auc_95ci": bootstrap_auc(y[mask], pred[mask]),
                         "applicants": int(mask.sum()), "default_rate": float(y[mask].mean())}
        r = summary[name]
        log.info("%-10s test AUC %.4f (95%% CI %.4f-%.4f), %d applicants", name, r["auc"],
                 *r["auc_95ci"], r["applicants"])
    summary["minutes"] = round((time.perf_counter() - start) / 60, 1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = final_name(feature_set, data_path, param_set)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
    io.save_dataset(pd.DataFrame({"SK_ID_CURR": holdout["SK_ID_CURR"], "TARGET": y,
                                  "IS_THIN_FILE": thin, "PRED": pred}),
                    RESULTS_DIR / f"{name}_pred.csv")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(MODELS_DIR / f"{name}.txt", num_iteration=model.best_iteration_)
    row = {"data": summary["data"], "feature_set": feature_set, "params": param_set,
           "n_features": len(features)}
    for group in ["overall", "has_bureau", "thin_file"]:
        row[f"auc_{group}"] = round(summary[group]["auc"], 4)
        row[f"ci_low_{group}"], row[f"ci_high_{group}"] = (round(v, 4) for v in summary[group]["auc_95ci"])
    path = RESULTS_DIR / "final.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", choices=sorted(FEATURE_SETS), default="A")
    parser.add_argument("--data", type=Path, default=io.DATA_DIR / "processed" / "combined_fe123.pkl")
    parser.add_argument("--params", choices=sorted(PARAM_SETS), default="default",
                        help="model settings: the defaults, or those found by src/tune.py")
    parser.add_argument("--final", action="store_true", help="score the locked test set once")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    (final if args.final else run)(args.features, args.data, args.params)


if __name__ == "__main__":
    main()
