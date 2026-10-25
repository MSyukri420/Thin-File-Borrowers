"""Search LightGBM settings with Optuna, using cross-validation on the development set only.

Usage:
    python -m src.tune                 # 40 trials, feature set B on combined_fe123
    python -m src.tune --trials 10

Each trial trains on the same 5 folds as src/train.py and is scored by its mean fold AUC
(all applicants). Trials that are clearly worse than the median after two folds are
stopped early. The first trial is the current default settings, as a reference.
The locked test set is never loaded into a model here.
Results go to output/results/tuning_B_combined_fe123.json; copy the best settings into
TUNED_PARAMS in src/train.py.
"""

import argparse
import json
import logging

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from . import io, train

log = logging.getLogger(__name__)

FEATURE_SET = "B"
DATA = io.DATA_DIR / "processed" / "combined_fe123.pkl"
NAME = f"{FEATURE_SET}_{DATA.stem}"
TUNED = ["num_leaves", "min_child_samples", "colsample_bytree", "subsample", "reg_lambda",
         "reg_alpha", "min_split_gain"]


def suggest(trial):
    """Settings for one trial; the learning rate stays at 0.05 so trials are comparable."""
    return {
        **train.PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def make_objective(X, y, folds, thin):
    def objective(trial):
        params = suggest(trial)
        oof = np.zeros(len(y))
        aucs = []
        for fold in np.unique(folds):
            valid_idx, oof[valid_idx], model = train.fit_fold(X, y, folds, fold, params)
            aucs.append(roc_auc_score(y[valid_idx], oof[valid_idx]))
            trial.report(float(np.mean(aucs)), int(fold))
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("auc_thin_file", roc_auc_score(y[thin == 1], oof[thin == 1]))
        log.info("trial %d: mean fold AUC %.4f, thin file %.4f", trial.number, np.mean(aucs),
                 trial.user_attrs["auc_thin_file"])
        return float(np.mean(aucs))
    return objective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dev, _ = train.split_holdout(train.load_train(DATA))
    features = train.select_features(dev.columns, FEATURE_SET)
    X, y = dev[features], dev["TARGET"].to_numpy()
    folds = train.make_folds(y)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=train.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    # the defaults have no L1 penalty; 0.001, the bottom of its range, is close to none
    study.enqueue_trial({**{k: train.PARAMS.get(k, 0.0) for k in TUNED}, "reg_alpha": 1e-3})
    study.optimize(make_objective(X, y, folds, dev["IS_THIN_FILE"].to_numpy()), n_trials=args.trials)

    trials = [{"number": t.number, "state": t.state.name, "mean_fold_auc": t.value,
               "auc_thin_file": t.user_attrs.get("auc_thin_file"), "params": t.params}
              for t in study.trials]
    best = {**train.PARAMS, **study.best_params}
    out = train.RESULTS_DIR / f"tuning_{NAME}.json"
    out.write_text(json.dumps({"best_mean_fold_auc": study.best_value, "best_params": best,
                               "trials": trials}, indent=2))
    log.info("best mean fold AUC %.4f (default %.4f): %s", study.best_value,
             study.trials[0].value, study.best_params)


if __name__ == "__main__":
    main()
