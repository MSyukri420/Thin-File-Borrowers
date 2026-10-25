"""Charts for the experiments run by src/train.py, saved as PNGs in docs/figures/.

Usage:
    python -m src.graph                          # main dataset combined_fe123, detail charts for B
    python -m src.graph --data combined --features C

Reads output/results/ (log.csv, the per-run JSON summaries and out-of-fold predictions,
pruning_B_combined_fe123.json and the final_*.json test scores if present). A chart
whose runs are missing is skipped.
figure_data.csv holds every plotted number, as a table view of the charts.
"""

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from . import train  # noqa: E402

log = logging.getLogger(__name__)

FIGURES_DIR = train.RESULTS_DIR.parents[1] / "docs" / "figures"

# Dataset versions in the order the feature batches were added.
STEPS = {"combined": "Raw", "combined_fe1": "+ Batch 1", "combined_fe12": "+ Batch 2",
         "combined_fe123": "+ Batch 3"}
SETS = ["A", "B", "C"]
SET_LABELS = {"A": "A: all features", "B": "B: no bureau info", "C": "C: no bureau,\nno outside scores"}
GROUPS = ["overall", "has_bureau", "thin_file"]
GROUP_LABELS = {"overall": "All applicants", "has_bureau": "Has bureau record", "thin_file": "Thin file"}

# Colours (validated for colour-blind separation): one fixed hue per applicant group,
# a one-hue light->dark ramp for the ordered feature sets (more information = darker),
# and a separate pair for raw vs engineered features.
GROUP_COLORS = {"overall": "#2a78d6", "has_bureau": "#1baf7a", "thin_file": "#eb6834"}
SET_COLORS = {"A": "#104281", "B": "#2a78d6", "C": "#86b6ef"}
ENGINEERED_COLOR, RAW_COLOR = "#6250d6", "#eda100"
# Diverging pair for SHAP contributions, and a one-hue ramp for feature values.
RAISES_RISK, LOWERS_RISK = "#e34948", "#2a78d6"
VALUE_RAMP = matplotlib.colors.LinearSegmentedColormap.from_list("value", ["#cde2fb", "#104281"])
# Uncorrected vs corrected probabilities; actual vs predicted; defaulters caught vs good payers flagged.
UNCORRECTED_COLOR, CORRECTED_COLOR = RAW_COLOR, ENGINEERED_COLOR
ACTUAL_COLOR, PREDICTED_COLOR = "#2a78d6", "#eb6834"
CAUGHT_COLOR, WRONGLY_FLAGGED_COLOR = "#1baf7a", "#6250d6"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Every plotted number, collected for figure_data.csv.
_table = []


def _record(figure, **values):
    _table.append({"figure": figure, **values})


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")


def _figure_header(fig, title, subtitle):
    """Title and subtitle above a row of panels, left-aligned, without overlapping."""
    fig.subplots_adjust(top=0.78)
    fig.text(0.06, 0.97, title, fontsize=12, color=INK, va="top")
    fig.text(0.06, 0.905, subtitle, fontsize=9, color=MUTED, va="top")


def _save(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    log.info("saved %s", path)
    return path


def load_log():
    """Latest logged run with the default settings for every (dataset version, feature set)."""
    runs = pd.read_csv(train.RESULTS_DIR / "log.csv")
    runs = runs[runs["params"] == "default"]
    return runs.drop_duplicates(["data", "feature_set"], keep="last").set_index(["data", "feature_set"])


def load_oof(feature_set, data):
    path = train.RESULTS_DIR / f"lightgbm_{feature_set}_{data}_oof.csv"
    return pd.read_csv(path) if path.exists() else None


def fold_gains(feature_set, before, after, group):
    """Per-fold AUC change from dataset version ``before`` to ``after`` on the same folds."""
    a, b = load_oof(feature_set, before), load_oof(feature_set, after)
    if a is None or b is None:
        return None
    assert (a["SK_ID_CURR"] == b["SK_ID_CURR"]).all() and (a["FOLD"] == b["FOLD"]).all()
    mask = np.ones(len(a), bool) if group == "overall" else (a["IS_THIN_FILE"] == 1).to_numpy()
    gains = []
    for fold in sorted(a["FOLD"].unique()):
        m = mask & (a["FOLD"] == fold).to_numpy()
        gains.append(roc_auc_score(a["TARGET"][m], b["PRED"][m]) - roc_auc_score(a["TARGET"][m], a["PRED"][m]))
    return np.array(gains)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def plot_auc_by_feature_set(runs, data):
    """Dot plot: AUC per feature set for each applicant group, with fold spread."""
    if not all((data, s) in runs.index for s in SETS):
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(SETS))
    for i, group in enumerate(GROUPS):
        rows = runs.loc[[(data, s) for s in SETS]]
        auc, std = rows[f"auc_{group}"].to_numpy(), rows[f"std_{group}"].to_numpy()
        pos = x + (i - 1) * 0.28
        ax.errorbar(pos, auc, yerr=std, fmt="o", ms=8, color=GROUP_COLORS[group], elinewidth=1.5,
                    capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5, label=GROUP_LABELS[group])
        for p, v, sd, s in zip(pos, auc, std, SETS):
            # value to the right of the dot, clear of its own error bar
            ax.annotate(f"{v:.3f}", (p, v), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
            _record("auc_by_feature_set", data=data, feature_set=s, group=group, auc=v, fold_std=sd)
    ax.set_xticks(x, [SET_LABELS[s] for s in SETS])
    ax.set_xlim(-0.6, len(SETS) - 0.3)
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y")
    ax.legend(loc="upper right", ncols=3, bbox_to_anchor=(1, -0.18), handletextpad=0.3)
    ax.set_title("AUC by feature set and applicant group", pad=22)
    _subtitle(ax, f"LightGBM, 5-fold CV on the 85% development set · bars = fold std · data: {data}")
    return _save(fig, "auc_by_feature_set")


def plot_final_test(data):
    """Dot plot: locked test-set AUC per feature set for each group, with bootstrap 95% CI."""
    paths = [train.RESULTS_DIR / f"final_{s}_{data}.json" for s in SETS]
    if not all(p.exists() for p in paths):
        return None
    finals = [json.loads(p.read_text()) for p in paths]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(SETS))
    for i, group in enumerate(GROUPS):
        auc = np.array([f[group]["auc"] for f in finals])
        ci = np.array([f[group]["auc_95ci"] for f in finals])
        pos = x + (i - 1) * 0.28
        ax.errorbar(pos, auc, yerr=[auc - ci[:, 0], ci[:, 1] - auc], fmt="o", ms=8,
                    color=GROUP_COLORS[group], elinewidth=1.5, capsize=0, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=GROUP_LABELS[group])
        for p, v, (lo, hi), s in zip(pos, auc, ci, SETS):
            ax.annotate(f"{v:.3f}", (p, v), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
            _record("final_test", data=data, feature_set=s, group=group, auc=v, ci_low=lo, ci_high=hi)
    ax.set_xticks(x, [SET_LABELS[s] for s in SETS])
    ax.set_xlim(-0.6, len(SETS) - 0.3)
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y")
    ax.legend(loc="upper right", ncols=3, bbox_to_anchor=(1, -0.18), handletextpad=0.3)
    ax.set_title("Final score on the locked 15% test set", pad=22)
    _subtitle(ax, f"Scored once, after all choices were made · bars = bootstrap 95% CI · data: {data}")
    return _save(fig, "final_test")


def plot_engineering_progression(runs):
    """Small multiples: AUC as each feature batch is added, overall and thin file."""
    steps = [d for d in STEPS if all((d, s) in runs.index for s in SETS)]
    if len(steps) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), sharey=True)
    x = np.arange(len(steps))
    for ax, s in zip(axes, SETS):
        for group in ["overall", "thin_file"]:
            auc = np.array([runs.loc[(d, s), f"auc_{group}"] for d in steps])
            ax.plot(x, auc, marker="o", ms=7, color=GROUP_COLORS[group], label=GROUP_LABELS[group],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{auc[-1]:.3f}", (x[-1], auc[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
            ax.annotate(f"{auc[0]:.3f}", (x[0], auc[0]), xytext=(-8, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=8, color=INK_2)
            for d, v in zip(steps, auc):
                _record("engineering_progression", data=d, feature_set=s, group=group, auc=v)
        ax.set_xticks(x, [STEPS[d] for d in steps], fontsize=8)
        ax.set_xlim(-0.6, len(steps) - 0.4)
        ax.set_title(SET_LABELS[s].replace("\n", " "), fontsize=10)
        ax.grid(axis="y")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncols=2)
    _figure_header(fig, "AUC as each feature batch is added",
                   "Batch 1 = lateness & payments · 2 = card & previous applications · "
                   "3 = application ratios · same folds throughout")
    return _save(fig, "engineering_progression")


def plot_engineering_fold_gains():
    """Strip plot: per-fold AUC gain of each batch over the previous version."""
    steps = list(STEPS)
    pairs = list(zip(steps[:-1], steps[1:]))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), sharey=True)
    drew = False
    for ax, s in zip(axes, SETS):
        for j, (before, after) in enumerate(pairs):
            for i, group in enumerate(["overall", "thin_file"]):
                gains = fold_gains(s, before, after, group)
                if gains is None:
                    continue
                drew = True
                pos = j + (i - 0.5) * 0.3
                ax.scatter(np.full(len(gains), pos), gains, s=40, color=GROUP_COLORS[group],
                           edgecolor=SURFACE, linewidth=1.2, zorder=3,
                           label=GROUP_LABELS[group] if j == 0 else None)
                ax.hlines(gains.mean(), pos - 0.1, pos + 0.1, color=INK, linewidth=2, zorder=4)
                for fold, g in enumerate(gains):
                    _record("engineering_fold_gains", feature_set=s, step=STEPS[after], group=group,
                            fold=fold, auc_gain=g)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_xticks(range(len(pairs)), [STEPS[a] for _, a in pairs], fontsize=8)
        ax.set_xlim(-0.6, len(pairs) - 0.4)
        ax.set_title(SET_LABELS[s].replace("\n", " "), fontsize=10)
        ax.grid(axis="y")
    if not drew:
        plt.close(fig)
        return None
    axes[0].set_ylabel("AUC gain vs previous version")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncols=2)
    _figure_header(fig, "What each batch adds, fold by fold",
                   "Each dot = one of the 5 folds · black tick = mean · above 0 = the batch helped")
    return _save(fig, "engineering_fold_gains")


def plot_pruning(runs):
    """Dot plot: AUC change from removing features, relative to keeping them all."""
    path = train.RESULTS_DIR / "pruning_B_combined_fe123.json"
    if not path.exists():
        return None
    results = pd.DataFrame(json.loads(path.read_text())["results"])
    results = results[~results["variant"].str.startswith("baseline")].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(results))
    for i, group in enumerate(["overall", "thin_file"]):
        gain = results[f"gain_{group}"].to_numpy()
        ax.scatter(gain, y + (0.5 - i) * 0.25, s=60, color=GROUP_COLORS[group], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=GROUP_LABELS[group])
        for v, (_, row) in zip(gain, results.iterrows()):
            _record("pruning", variant=row["variant"], n_features=row["n_features"], group=group, auc_gain=v)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(y, results["variant"])
    ax.set_xlabel("AUC change vs keeping all features")
    limit = max(0.002, np.abs(results[["gain_overall", "gain_thin_file"]].to_numpy()).max() * 1.2)
    ax.set_xlim(-limit, limit)
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    ax.set_title("Removing features: no clear gain", pad=22)
    base = runs.loc[("combined_fe123", "B")] if ("combined_fe123", "B") in runs.index else None
    spread = (f" · fold std {base['std_overall']:.3f} overall, {base['std_thin_file']:.3f} thin file"
              if base is not None else "")
    _subtitle(ax, f"Feature set B on combined_fe123 · same 5 folds{spread}")
    return _save(fig, "pruning")


def plot_top_features(feature_set, data, n=20):
    """Horizontal bars: the model's most-used features, engineered ones highlighted."""
    path = train.RESULTS_DIR / f"lightgbm_{feature_set}_{data}.json"
    if not path.exists():
        return None
    gain = pd.Series(json.loads(path.read_text())["top_features_by_gain"]).head(n).iloc[::-1]
    engineered = gain.index.str.contains("_FE_")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(gain.index, gain.to_numpy(), height=0.7,
            color=np.where(engineered, ENGINEERED_COLOR, RAW_COLOR))
    for name, value, eng in zip(gain.index, gain.to_numpy(), engineered):
        _record("top_features", data=data, feature_set=feature_set, feature=name, gain=value, engineered=eng)
    ax.set_xlabel("Average gain per fold")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0, labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ENGINEERED_COLOR, RAW_COLOR)]
    ax.legend(handles, ["Engineered (_FE_)", "Original column summary"], loc="lower right")
    ax.set_title(f"Top {n} features, feature set {feature_set}", pad=22)
    _subtitle(ax, f"LightGBM gain averaged over 5 folds · data: {data}")
    return _save(fig, f"top_features_{feature_set}")


def plot_roc_thin_file(data):
    """ROC curves for thin-file applicants under each feature set."""
    oofs = {s: load_oof(s, data) for s in SETS}
    if any(o is None for o in oofs.values()):
        return None
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for s, oof in oofs.items():
        thin = oof[oof["IS_THIN_FILE"] == 1]
        fpr, tpr, _ = roc_curve(thin["TARGET"], thin["PRED"])
        auc = roc_auc_score(thin["TARGET"], thin["PRED"])
        ax.plot(fpr, tpr, color=SET_COLORS[s], label=f"{SET_LABELS[s].replace(chr(10), ' ')} (AUC {auc:.3f})")
        _record("roc_thin_file", data=data, feature_set=s, auc=auc)
    ax.set_xlabel("False positive rate (good payers flagged)")
    ax.set_ylabel("True positive rate (defaulters caught)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid()
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ROC curves, thin-file applicants", pad=22)
    _subtitle(ax, f"Out-of-fold predictions · data: {data}")
    return _save(fig, "roc_thin_file")


def plot_default_rate_by_decile(feature_set, data):
    """Bars: actual default rate in each tenth of predicted risk, overall and thin file."""
    oof = load_oof(feature_set, data)
    if oof is None:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(1, 11)
    width = 0.38
    for i, group in enumerate(["overall", "thin_file"]):
        part = oof if group == "overall" else oof[oof["IS_THIN_FILE"] == 1]
        decile = pd.qcut(part["PRED"].rank(method="first"), 10, labels=x)
        rate = part.groupby(decile, observed=True)["TARGET"].mean().to_numpy()
        # 2px surface gap between neighbouring bars
        pos = x + (i - 0.5) * (width + 0.02)
        ax.bar(pos, rate, width=width, color=GROUP_COLORS[group],
               label=f"{GROUP_LABELS[group]} (average {part['TARGET'].mean():.1%})")
        ax.annotate(f"{rate[-1]:.0%}", (pos[-1], rate[-1]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
        ax.annotate(f"{rate[0]:.1%}", (pos[0], rate[0]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
        for d, r in zip(x, rate):
            _record("default_rate_by_decile", data=data, feature_set=feature_set, group=group,
                    decile=int(d), default_rate=r)
    ax.set_xticks(x, ["1\nsafest"] + [str(d) for d in x[1:-1]] + ["10\nriskiest"])
    ax.set_xlabel("Tenth of applicants by predicted risk")
    ax.set_ylabel("Actual default rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    ax.set_title(f"Default rate by predicted risk, feature set {feature_set}", pad=22)
    _subtitle(ax, f"Out-of-fold predictions on the development set · data: {data}")
    return _save(fig, f"default_rate_by_decile_{feature_set}")


def _load_result(name):
    path = train.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def plot_tuning():
    """Dots: mean fold AUC of every finished Optuna trial, with the best so far."""
    tuning = _load_result("tuning_B_combined_fe123.json")
    if tuning is None:
        return None
    trials = pd.DataFrame(tuning["trials"])
    done = trials[trials["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(trials["mean_fold_auc"].iloc[0], color=AXIS, linewidth=1.5, linestyle="--")
    ax.annotate(f"default settings {trials['mean_fold_auc'].iloc[0]:.4f}",
                (0, trials["mean_fold_auc"].iloc[0]), xytext=(4, -12), textcoords="offset points",
                fontsize=8, color=INK_2)
    ax.scatter(done["number"], done["mean_fold_auc"], s=40, color=GROUP_COLORS["overall"],
               edgecolor=SURFACE, linewidth=1.2, zorder=3, label="finished trial")
    best = done.set_index("number")["mean_fold_auc"].cummax()
    ax.step(best.index, best.to_numpy(), where="post", color=INK, linewidth=2, label="best so far")
    ax.annotate(f"{best.iloc[-1]:.4f}", (best.index[-1], best.iloc[-1]), xytext=(6, 4),
                textcoords="offset points", fontsize=8, color=INK_2)
    for row in trials.itertuples():
        _record("tuning", step=row.number, auc=row.mean_fold_auc, variant=row.state)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean fold AUC")
    ax.grid(axis="y")
    ax.legend(loc="lower right")
    n_pruned = int((trials["state"] == "PRUNED").sum())
    ax.set_title("Hyperparameter search, feature set B", pad=22)
    _subtitle(ax, f"Optuna, {len(trials)} trials on the development set · {n_pruned} stopped early "
                  "after 2 folds (not shown)")
    return _save(fig, "tuning")


def plot_shap_summary(feature_set, version):
    """Beeswarm: SHAP value per applicant for the top features, coloured by feature value."""
    path = train.RESULTS_DIR / f"shap_{feature_set}_{version}_sample.csv"
    if not path.exists():
        return None
    sample = pd.read_csv(path)
    features = [c.removeprefix("SHAP_") for c in sample if c.startswith("SHAP_")]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    rng = np.random.default_rng(train.SEED)
    for i, f in enumerate(features):
        y = len(features) - 1 - i
        shap, pct = sample[f"SHAP_{f}"].to_numpy(), sample[f"PCT_{f}"].to_numpy()
        jitter = rng.uniform(-0.3, 0.3, len(shap))
        missing = np.isnan(pct)
        ax.scatter(shap[missing], y + jitter[missing], s=5, color=MUTED, alpha=0.5, linewidth=0)
        points = ax.scatter(shap[~missing], y + jitter[~missing], s=5, c=pct[~missing],
                            cmap=VALUE_RAMP, vmin=0, vmax=1, linewidth=0)
        _record("shap_summary", feature_set=feature_set, data=version, feature=f,
                gain=float(np.abs(shap).mean()))
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(features)), features[::-1], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("SHAP value (log-odds): left lowers predicted risk, right raises it")
    ax.grid(axis="x")
    bar = fig.colorbar(points, ax=ax, pad=0.02, fraction=0.04)
    bar.set_label("Feature value (percentile; grey = missing)", color=INK_2)
    bar.set_ticks([0, 1], labels=["low", "high"])
    bar.outline.set_visible(False)
    ax.set_title(f"What drives the predictions, feature set {feature_set}", pad=22)
    _subtitle(ax, f"Top {len(features)} features by mean |SHAP| · {len(sample):,} random test "
                  f"applicants · data: {version}")
    return _save(fig, f"shap_summary_{feature_set}")


def plot_shap_examples(feature_set, version):
    """Bars: feature contributions for two thin-file applicants, one high-risk, one low-risk."""
    result = _load_result(f"shap_{feature_set}_{version}.json")
    if result is None:
        return None
    examples = result["examples"]
    fig, axes = plt.subplots(1, len(examples), figsize=(12, 4.6))
    for ax, (label, ex) in zip(axes, examples.items()):
        rows = ex["features"] + [{"feature": "all other features", "value": None,
                                  "shap": ex["other_features_log_odds"]}]
        names = []
        for r in rows:
            v = r["value"]
            if v is None:
                shown = "missing"
            elif isinstance(v, (int, float)):
                shown = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
            else:
                shown = str(v)
            names.append(r["feature"] if r["feature"] == "all other features" else f"{r['feature']} = {shown}")
        values = np.array([r["shap"] for r in rows])
        y = np.arange(len(rows))[::-1]
        ax.barh(y, values, height=0.7, color=np.where(values > 0, RAISES_RISK, LOWERS_RISK))
        for yy, v in zip(y, values):
            ax.annotate(f"{v:+.2f}", (v, yy), xytext=(4 if v > 0 else -4, 0), textcoords="offset points",
                        ha="left" if v > 0 else "right", va="center", fontsize=8, color=INK_2)
            _record("shap_examples", feature_set=feature_set, data=version, variant=label, auc_gain=v)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(y, names, fontsize=8)
        ax.tick_params(axis="y", length=0)
        lim = np.abs(values).max() * 1.35
        ax.set_xlim(-lim, lim)
        ax.grid(axis="x")
        ax.set_title(f"{label.capitalize()} thin-file applicant: predicted risk "
                     f"{ex['predicted_risk']:.1%}", fontsize=10)
    axes[0].set_xlabel("Contribution to risk (log-odds)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RAISES_RISK, LOWERS_RISK)]
    axes[-1].legend(handles, ["raises risk", "lowers risk"], loc="lower right")
    fig.subplots_adjust(wspace=0.9)
    _figure_header(fig, "Why the model scored these applicants as it did",
                   f"SHAP contributions, top {len(rows) - 1} features each · two thin-file test "
                   f"applicants · feature set {feature_set} · data: {version}")
    return _save(fig, f"shap_examples_{feature_set}")


def plot_calibration(feature_set, version, corrected="platt_by_group"):
    """Reliability diagram: predicted vs actual default rate by tenth, before and after correction."""
    result = _load_result(f"calibration_{feature_set}_{version}.json")
    if result is None:
        return None
    bins = pd.DataFrame(result["bins"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, group in zip(axes, ["overall", "thin_file"]):
        top = bins["actual"].max() * 1.08
        ax.plot([0, top], [0, top], color=AXIS, linewidth=1)
        for method, color, name in [("raw", UNCORRECTED_COLOR, "model as trained"),
                                    (corrected, CORRECTED_COLOR, "corrected per group")]:
            part = bins[(bins["group"] == group) & (bins["method"] == method)]
            ece = result["scores"][group][method]["ece"]
            ax.plot(part["predicted"], part["actual"], marker="o", ms=7, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{name} (avg gap {ece:.1%})")
            for r in part.itertuples():
                _record("calibration", feature_set=feature_set, data=version, group=group,
                        variant=method, decile=r.bin + 1, auc=r.predicted, default_rate=r.actual)
        s = result["scores"][group]
        ax.text(0.03, 0.97, f"Average predicted {s['raw']['mean_predicted']:.1%} → "
                f"{s[corrected]['mean_predicted']:.1%}\nActual {s['raw']['actual_rate']:.1%}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_xlabel("Predicted default rate")
        ax.set_title(GROUP_LABELS[group], fontsize=10)
        ax.grid()
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Actual default rate")
    _figure_header(fig, "Do predicted probabilities match reality?",
                   f"Locked test set, 10 equal-size bins of predicted risk · on the grey line = "
                   f"perfectly calibrated · feature set {feature_set} · data: {version}")
    return _save(fig, f"calibration_{feature_set}")


def plot_fairness(feature_set, version):
    """Dot plots by group: actual vs predicted default rate, and who gets flagged at one cutoff."""
    result = _load_result(f"fairness_{feature_set}_{version}.json")
    if result is None:
        return None
    table = pd.DataFrame(result["groups"])
    table = table[table["dimension"] != "all"]
    order = {"gender": 0, "age": 1, "bureau": 2}
    table = table.sort_values("dimension", key=lambda d: d.map(order), kind="stable")
    labels = [f"{g} ({d})" if d == "age" else g for d, g in zip(table["dimension"], table["group"])]
    y = np.arange(len(table))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    panels = [
        (axes[0], [("default_rate", ACTUAL_COLOR, "actual default rate"),
                   ("mean_predicted", PREDICTED_COLOR, "average predicted")], "Default rate"),
        (axes[1], [("defaulters_caught", CAUGHT_COLOR, "defaulters flagged"),
                   ("good_payers_flagged", WRONGLY_FLAGGED_COLOR, "good payers flagged")],
         f"Share flagged as the riskiest {result['flag_share']:.0%}"),
    ]
    for ax, series, xlabel in panels:
        for col, color, name in series:
            ax.scatter(table[col], y, s=60, color=color, edgecolor=SURFACE, linewidth=1.5,
                       zorder=3, label=name)
        lo, hi = table[series[0][0]].to_numpy(), table[series[1][0]].to_numpy()
        ax.hlines(y, np.minimum(lo, hi), np.maximum(lo, hi), color=GRID, linewidth=3, zorder=2)
        for yy, a, b in zip(y, lo, hi):
            ax.annotate(f"{a:.0%}" if a >= 0.1 else f"{a:.1%}", (a, yy), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7, color=INK_2)
            ax.annotate(f"{b:.0%}" if b >= 0.1 else f"{b:.1%}", (b, yy), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=7, color=INK_2)
        for row in table.itertuples():
            _record("fairness", feature_set=feature_set, data=version, group=row.group,
                    **{col: getattr(row, col) for col, _, _ in series})
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", length=0)
    for boundary in np.cumsum(table["dimension"].value_counts(sort=False).reindex(order).to_numpy())[:-1]:
        for ax in axes:
            ax.axhline(len(table) - boundary - 0.5, color=AXIS, linewidth=1)
    _figure_header(fig, "How the model treats different groups",
                   f"Locked test set · gender is not a model input · one cutoff for everyone · "
                   f"feature set {feature_set} · data: {version}")
    return _save(fig, f"fairness_{feature_set}")


def make_all(data="combined_fe123", feature_set="B", param_set="default"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _table.clear()
    runs = load_log()
    version = data if param_set == "default" else f"{data}_{param_set}"
    charts = {
        "final_test": lambda: plot_final_test(version),
        "tuning": plot_tuning,
        "shap_summary": lambda: plot_shap_summary(feature_set, version),
        "shap_examples": lambda: plot_shap_examples(feature_set, version),
        "calibration": lambda: plot_calibration(feature_set, version),
        "fairness": lambda: plot_fairness(feature_set, version),
        "auc_by_feature_set": lambda: plot_auc_by_feature_set(runs, data),
        "engineering_progression": lambda: plot_engineering_progression(runs),
        "engineering_fold_gains": plot_engineering_fold_gains,
        "pruning": lambda: plot_pruning(runs),
        "top_features": lambda: plot_top_features(feature_set, version),
        "roc_thin_file": lambda: plot_roc_thin_file(version),
        "default_rate_by_decile": lambda: plot_default_rate_by_decile(feature_set, version),
    }
    written = []
    for name, make in charts.items():
        path = make()
        if path is None:
            log.warning("skipped %s: its runs are not in output/results/", name)
        else:
            written.append(path)
    pd.DataFrame(_table).to_csv(FIGURES_DIR / "figure_data.csv", index=False)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="combined_fe123", choices=list(STEPS),
                        help="dataset version for the single-version charts")
    parser.add_argument("--features", default="B", choices=SETS,
                        help="feature set for the single-model charts")
    parser.add_argument("--params", default="default", choices=sorted(train.PARAM_SETS),
                        help="model settings for the test-set, SHAP, calibration and fairness charts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    make_all(args.data, args.features, args.params)


if __name__ == "__main__":
    main()
