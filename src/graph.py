"""Charts for the experiments run by src/train.py, saved as PNGs in output/figures/.

Usage:
    python -m src.graph                          # main dataset combined_fe123, detail charts for B
    python -m src.graph --data combined --features C

Reads output/results/ (log.csv, the per-run JSON summaries and out-of-fold predictions,
and pruning_B_combined_fe123.json if present). A chart whose runs are missing is skipped.
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

FIGURES_DIR = train.RESULTS_DIR.parent / "figures"

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
    """Latest logged run for every (dataset version, feature set)."""
    runs = pd.read_csv(train.RESULTS_DIR / "log.csv")
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


def make_all(data="combined_fe123", feature_set="B"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _table.clear()
    runs = load_log()
    charts = {
        "auc_by_feature_set": lambda: plot_auc_by_feature_set(runs, data),
        "engineering_progression": lambda: plot_engineering_progression(runs),
        "engineering_fold_gains": plot_engineering_fold_gains,
        "pruning": lambda: plot_pruning(runs),
        "top_features": lambda: plot_top_features(feature_set, data),
        "roc_thin_file": lambda: plot_roc_thin_file(data),
        "default_rate_by_decile": lambda: plot_default_rate_by_decile(feature_set, data),
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
                        help="feature set for the top-features and decile charts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    make_all(args.data, args.features)


if __name__ == "__main__":
    main()
