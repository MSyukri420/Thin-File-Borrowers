"""Rebuild every dataset, rerun every experiment and redraw every chart.

Usage:
    python -m src.run_all       # long-running; needs the Kaggle CSVs in dataset/

Steps, in order:
1. src.preprocess: raw dataset, then with feature batches 1, 1+2 and 1+2+3.
2. src.train: 5-fold CV for feature sets A, B and C on every dataset version.
3. src.prune: pruning experiments (feature set B, combined_fe123).
4. src.train --final: score the locked test set once, raw and fully engineered.
5. src.graph: charts into docs/figures/.
"""

import logging

from . import graph, io, preprocess, prune, train

VERSIONS = {"combined": (), "combined_fe1": (1,), "combined_fe12": (1, 2),
            "combined_fe123": (1, 2, 3)}
FINAL_VERSIONS = ["combined", "combined_fe123"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    processed = io.DATA_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, batches in VERSIONS.items():
        io.save_dataset(preprocess.build_dataset(batches=batches), processed / f"{name}.pkl")
    for name in VERSIONS:
        for feature_set in train.FEATURE_SETS:
            train.run(feature_set, processed / f"{name}.pkl")
    prune.main()
    for name in FINAL_VERSIONS:
        for feature_set in train.FEATURE_SETS:
            train.final(feature_set, processed / f"{name}.pkl")
    graph.make_all()


if __name__ == "__main__":
    main()
