← [Back to README](../README.md)

# Thin-File Borrowers: technical details

## How it works

1. **Thin file**: an applicant with no record at the credit bureau (`BUREAU_COUNT == 0`). That covers 44,020 of 307,511 labelled applicants (14.3%), who default at 10.1% vs 7.7% for everyone else.
2. **Default**: Home Credit's `TARGET`, meaning payment difficulties early in the loan (8.1% of applicants).
3. **One row per applicant.** The six history tables have many rows per applicant, so each one is summarised: count, mean, min, max and sum for every number, plus the share of each category. Hand-made row-level features (for example "was this instalment late?") are computed before summarising, because the summary would lose them.
4. **Feature sets**: the same model trained on three views of the data.

| Set | Removes | Features (raw / engineered) |
|---|---|---|
| A | nothing | 556 / 627 |
| B | bureau information: `BUREAU_*`, `BB_*`, `AMT_REQ_CREDIT_BUREAU_*`, `EXT_SOURCE_3`, `EXT_SOURCE_FE_*` | 464 / 533 |
| C | B, plus every outside score (`EXT_SOURCE_*`) | 462 / 531 |

`EXT_SOURCE_1/2/3` are scores from outside sources. `EXT_SOURCE_3` behaves like a bureau score: it is missing for 97% of thin-file applicants and 7% of others, so it counts as bureau information. `EXT_SOURCE_2` is present for 99.7% of thin-file applicants. `CODE_GENDER` is excluded from every set, because gender is a legally restricted basis for lending decisions in many places.

5. **Model**: LightGBM (binary objective, learning rate 0.05, 31 leaves, at least 100 applicants per leaf, 80% row and 50% column sampling). Early stopping picks the number of trees on a 10% slice of the training data, never the data being scored.

### Hand-made features

Added in three batches, each tested against the version before it:

- **Batch 1, payment behaviour**: instalments: days late, late flag, paid ÷ due, underpaid flag, plus late-in-the-last-year versions; POS/cash loans: late flag overall and in the last year.
- **Batch 2, credit cards and past applications**: card utilisation (overall and last year), drawings ÷ limit, payment ÷ minimum due; past applications: asked ÷ granted, total cost ratio (annuity × payments ÷ credit), refused in the last 2 years, whether the most recent application was refused.
- **Batch 3, the application itself**: credit ÷ annuity (roughly the loan term), annuity ÷ income, credit ÷ goods price, employed days ÷ age, income per family member, mean outside score, bureau debt ÷ bureau credit.

## Evaluation

The labelled applicants are split once (sorted by ID, stratified, seed 42) into a **development set** (85%, 261,384 applicants) and a **locked test set** (15%, 46,127 applicants). Every choice was made with 5-fold cross-validation on the development set. The test set was scored once at the end, for every version, and used to report, not to choose.

**Locked test set** (AUC, bootstrap 95% CI in brackets; 6,534 thin-file test applicants):

| Model | All applicants | Has bureau record | Thin file |
|---|---|---|---|
| Coin flip | 0.500 | 0.500 | 0.500 |
| `EXT_SOURCE_2` alone, no model | 0.663 | 0.661 | 0.667 |
| A, raw | 0.785 [0.777–0.793] | 0.786 [0.778–0.795] | 0.767 [0.748–0.787] |
| B, raw | 0.767 [0.759–0.775] | 0.764 [0.755–0.772] | 0.776 [0.757–0.794] |
| C, raw | 0.741 [0.733–0.750] | 0.738 [0.729–0.746] | 0.754 [0.735–0.773] |
| A, engineered | **0.791** [0.783–0.798] | **0.792** [0.784–0.800] | 0.776 [0.757–0.795] |
| B, engineered | 0.772 [0.765–0.780] | 0.770 [0.761–0.777] | **0.781** [0.763–0.800] |
| C, engineered | 0.752 [0.744–0.760] | 0.750 [0.741–0.758] | 0.762 [0.743–0.780] |

The coin flip is a random ranking. The single-score baseline ranks applicants by one outside score that almost everyone has, with no training at all.

**Cross-validation on the development set** (AUC, after each feature batch):

| Version | A all | A thin | B all | B thin | C all | C thin |
|---|---|---|---|---|---|---|
| Raw | 0.781 | 0.768 | 0.760 | 0.772 | 0.737 | 0.746 |
| + Batch 1 | 0.783 | 0.772 | 0.763 | 0.775 | 0.742 | 0.751 |
| + Batch 2 | 0.785 | 0.775 | 0.765 | 0.775 | 0.743 | 0.753 |
| + Batch 3 | 0.787 | 0.777 | 0.767 | 0.779 | 0.747 | 0.756 |

All three batches together improved AUC in every fold, for every set and both groups. Batch 2 was the weakest: +0.0008 for thin-file applicants in set B, better in only 3 of 5 folds.

How to read this:

- **AUC measures ranking**: the chance that a random defaulter is scored riskier than a random non-defaulter. 0.5 is a coin flip; 1.0 is perfect. The model's probabilities were not calibrated, so read them as a ranking, not as default rates.
- **The thin-file gap is a data gap.** In set A thin-file applicants score lower (0.776 vs 0.792) because everyone else has bureau data to help. Remove it (set B) and thin-file applicants rank slightly better than the rest.
- **Which set to use**: A when bureau data exists, B for thin-file applicants. C shows what is left when outside scores are also unavailable.
- **Thin-file numbers are less certain**: about 6,500 test applicants, so the 95% interval is about ±0.02.
- **For reference**, the Kaggle winners scored 0.806 on Kaggle's own hidden test set with a large ensemble. That is a different test set, so it is a rough reference only.

### Pruning

Does removing features help? Tested on set B with engineered features (533 features), compared fold by fold with the full set:

| Removed | Features left | Change, all | Change, thin file |
|---|---|---|---|
| Building statistics (`_AVG`, `_MODE`, `_MEDI`) | 486 | −0.0001 | −0.0003 |
| Near-constant columns (>99.5% one value) | 517 | +0.0004 | +0.0009 |
| History `_SUM` columns | 467 | −0.0004 | −0.0005 |
| Rare category shares (<0.5%) | 471 | +0.0003 | −0.0003 |
| Zero importance (chosen inside each fold) | about 432 | −0.0001 | −0.0009 |
| Keep 99% of importance | about 350 | +0.0002 | −0.0015 |
| Keep 95% of importance | about 257 | −0.0001 | −0.0007 |

Every change is within ±0.0015, smaller than the fold-to-fold spread, so all features were kept. Importance-based pruning was slightly worse for thin-file applicants in all three variants.

## Data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) (Kaggle, 2018), from Home Credit, a lender to people with little or no credit history. Used under the competition [rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules). The data is not included in this repo (about 2.5 GB of CSVs).

| Table | One row per |
|---|---|
| `application_train` / `application_test` | applicant (307,511 labelled, 48,744 unlabelled) |
| `bureau` | loan reported by other lenders to the credit bureau |
| `bureau_balance` | month of each bureau loan |
| `previous_application` | earlier application to Home Credit |
| `POS_CASH_balance` | month of each earlier point-of-sale or cash loan |
| `installments_payments` | instalment of each earlier loan |
| `credit_card_balance` | month of each earlier credit card |

Cleaning:

- `365243` in any `DAYS_*` column is a placeholder for "no date", so it becomes missing.
- `XNA` category values and `Unknown` family status become missing.
- Bureau loans with impossible dates are dropped, and negative debt and credit limits are clipped to zero.
- Ratios with a zero denominator become missing.

The unlabelled Kaggle test applicants are built with the training data, but never used for modelling.

## Running and retraining

Python 3.12. Download the CSVs from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) into `dataset/`.

```bash
pip install -r requirements.txt

# build the dataset: raw columns and summaries only
python -m src.preprocess

# build with hand-made feature batches (any of 1 2 3)
python -m src.preprocess --batches 1 2 3

# 5-fold cross-validation (feature set A, B or C; default data: combined_fe123)
python -m src.train --features B
python -m src.train --features B --data dataset/processed/combined.pkl

# score the locked test set once
python -m src.train --features B --final

# pruning experiments (needs the set B cross-validation run on combined_fe123 first)
python -m src.prune

# draw every chart into docs/figures/
python -m src.graph

# everything above, end to end
python -m src.run_all

# tests (synthetic data; the real-data tests are skipped without dataset/)
python -m pytest -k "not RealData"
```

Each cross-validation run takes about 2 minutes on a laptop.

## Project layout

```
src/
  io.py               reading the Kaggle tables, saving datasets (.pkl, .parquet or .csv)
  preprocess.py       cleaning, hand-made feature batches, one row per applicant
  train.py            feature sets, locked test split, cross-validation, final scoring
  prune.py            pruning experiments
  graph.py            charts and the numbers behind them
  run_all.py          rebuild and rerun everything
tests/
  test_pipeline.py    synthetic-data tests for every step, plus checks on the real data
docs/
  figures/            charts used in the README and here

produced by the pipeline (not in the repo):
dataset/processed/    combined.pkl, combined_fe1.pkl, combined_fe12.pkl, combined_fe123.pkl
output/results/       run summaries (JSON), out-of-fold predictions, log.csv, final.csv
```

## Charts

[`docs/figures/`](figures/), with every plotted number in [`figure_data.csv`](figures/figure_data.csv):

- [`final_test.png`](figures/final_test.png): locked test-set AUC per feature set and applicant group, with 95% intervals
- [`auc_by_feature_set.png`](figures/auc_by_feature_set.png): the same, from cross-validation
- [`engineering_progression.png`](figures/engineering_progression.png): AUC after each feature batch
- [`engineering_fold_gains.png`](figures/engineering_fold_gains.png): AUC gain of each batch over the version before it, fold by fold
- [`pruning.png`](figures/pruning.png): AUC change from each pruning experiment
- [`top_features_B.png`](figures/top_features_B.png): the 20 most important features in set B
- [`roc_thin_file.png`](figures/roc_thin_file.png): ROC curves for thin-file applicants, sets A, B and C (cross-validation)
- [`default_rate_by_decile_B.png`](figures/default_rate_by_decile_B.png): actual default rate for each tenth of predicted risk

## Version history

- **v2 (current).** Hand-made feature batches, LightGBM with three feature sets, a locked test set, pruning experiments, charts and this write-up. During this work I found a flaw in my first set B: it still contained the `AMT_REQ_CREDIT_BUREAU_*` enquiry counts. They are missing for 94% of thin-file applicants and almost never for anyone else, so the model could tell who was thin-file and the "no bureau data" comparison wasn't clean. I moved them into the bureau group and reran every set B and C result. The test set was locked before any of the results above were produced.
- **v1.** Data pipeline only: loading the seven tables, cleaning, summaries to one row per applicant, and tests.
