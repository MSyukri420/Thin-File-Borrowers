"""Test whether removing features helps feature set B on the engineered dataset.

Usage:
    python -m src.prune        # needs output/results/lightgbm_B_combined_fe123_oof.csv (src.train)

Two kinds of pruning, each compared fold by fold with the full feature set:
1. Group removal, chosen without looking at TARGET: building statistics, near-constant
   columns, history _SUM columns, rare category shares.
2. Importance pruning, chosen inside each fold from a model trained on that fold's
   training part only (so the scored fold never influences the choice): drop zero-gain
   features, or keep the features covering 95% / 99% of total gain.
Results go to output/results/pruning_B_combined_fe123.json.
"""

import json
import logging
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import io, train

log = logging.getLogger(__name__)

FEATURE_SET = "B"
DATA = io.DATA_DIR / "processed" / "combined_fe123.pkl"
NAME = f"{FEATURE_SET}_{DATA.stem}"
IMPORTANCE_VARIANTS = {"drop zero-gain": None, "keep 99% of gain": 0.99, "keep 95% of gain": 0.95}


def feature_groups(dev, features):
    """Groups of features that might be noise, defined without using TARGET."""
    top_share = {c: dev[c].value_counts(normalize=True, dropna=False).iloc[0] for c in features}
    return {
        "drop building stats": [c for c in features if re.search(r"_(AVG|MODE|MEDI)$", c)],
        "drop near-constant": [c for c, s in top_share.items() if s > 0.995],
        "drop history _SUM": [c for c in features
                              if c.startswith(("PREV_", "POS_", "INST_", "CC_")) and c.endswith("_SUM")],
        "drop rare category shares (<0.5%)": [c for c in features
                                              if c.endswith("_SHARE") and dev[c].mean() < 0.005],
    }


def _fit(X, y, fit_idx, stop_idx):
    model = lgb.LGBMClassifier(**train.PARAMS)
    model.fit(X.iloc[fit_idx], y[fit_idx], eval_X=X.iloc[stop_idx], eval_y=y[stop_idx],
              eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def importance_pruning(X, y, folds):
    """Out-of-fold predictions for each variant, choosing features inside every fold."""
    preds = {v: np.zeros(len(y)) for v in IMPORTANCE_VARIANTS}
    kept = {v: [] for v in IMPORTANCE_VARIANTS}
    for fold in np.unique(folds):
        train_idx, valid_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
        fit_idx, stop_idx = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx],
                                             random_state=train.SEED)
        full = _fit(X, y, fit_idx, stop_idx)
        gain = pd.Series(full.booster_.feature_importance("gain"),
                         index=X.columns).sort_values(ascending=False)
        share = gain.cumsum() / gain.sum()
        for name, cover in IMPORTANCE_VARIANTS.items():
            cols = (gain[gain > 0] if cover is None else gain[share.shift(fill_value=0) < cover]).index
            kept[name].append(len(cols))
            preds[name][valid_idx] = _fit(X[cols], y, fit_idx, stop_idx).predict_proba(
                X.iloc[valid_idx][cols])[:, 1]
        log.info("fold %d: kept %s", fold, [kept[v][-1] for v in IMPORTANCE_VARIANTS])
    return preds, {v: int(np.mean(k)) for v, k in kept.items()}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    dev, _ = train.split_holdout(train.load_train(DATA))
    features = train.select_features(dev.columns, FEATURE_SET)
    X, y = dev[features], dev["TARGET"].to_numpy()
    thin = dev["IS_THIN_FILE"].to_numpy()
    folds = train.make_folds(y)
    base = pd.read_csv(train.RESULTS_DIR / f"lightgbm_{NAME}_oof.csv")
    assert (base["SK_ID_CURR"].to_numpy() == dev["SK_ID_CURR"].to_numpy()).all()
    base_pred = base["PRED"].to_numpy()

    def fold_aucs(pred, mask):
        return np.array([roc_auc_score(y[mask & (folds == f)], pred[mask & (folds == f)])
                         for f in np.unique(folds)])

    results = []

    def report(name, pred, n_features):
        row = {"variant": name, "n_features": n_features}
        for group, mask in [("overall", np.ones(len(y), bool)), ("thin_file", thin == 1)]:
            diff = fold_aucs(pred, mask) - fold_aucs(base_pred, mask)
            row[f"auc_{group}"] = round(roc_auc_score(y[mask], pred[mask]), 4)
            row[f"gain_{group}"] = round(float(diff.mean()), 4)
            row[f"folds_better_{group}"] = int((diff > 0).sum())
        results.append(row)
        log.info("%s", row)

    report(f"baseline (all {len(features)})", base_pred, len(features))
    groups = feature_groups(dev, features)
    for name, cols in groups.items():
        keep = [c for c in features if c not in set(cols)]
        pred, _, _ = train.cross_validate(X[keep], y, folds)
        report(f"{name} (-{len(cols)})", pred, len(keep))
    preds, kept = importance_pruning(X, y, folds)
    for name in IMPORTANCE_VARIANTS:
        report(f"{name} (avg {kept[name]} kept)", preds[name], kept[name])

    out = train.RESULTS_DIR / f"pruning_{NAME}.json"
    out.write_text(json.dumps({"groups": groups, "results": results}, indent=2))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
