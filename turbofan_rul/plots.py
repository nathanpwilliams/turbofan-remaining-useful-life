"""The three figures, all written to figures/ as PNG.

Every axis is labelled in cycles, which is the unit RUL and RMSE share, so
the numbers can be read as "wrong by about this many cycles".
"""

from pathlib import Path

import matplotlib

# Agg renders straight to file and needs no display, so run.py works headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

TRAIN_COLOR = "#2a78d6"
VALIDATION_COLOR = "#eb6834"
REFERENCE_COLOR = "#52514e"


def _style(ax):
    """Push the frame and grid behind the data.

    The marks should carry the figure; the scaffolding only needs to be
    legible enough to read a value off.
    """
    ax.grid(True, color="#d8d7d2", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4af")
    ax.tick_params(colors="#52514e", labelsize=9)
    return ax


def plot_predicted_vs_actual(y_true, y_pred, title, ax):
    """Scatter predicted against actual RUL with the perfect-prediction line.

    Points are translucent because the per-cycle data stacks thousands deep;
    the density is the signal. Deviation from the dashed line is the error,
    and its *shape* is the diagnosis -- a curve means bias, scatter means
    variance.
    """
    _style(ax)
    ax.scatter(y_true, y_pred, s=8, alpha=0.2, color=TRAIN_COLOR, linewidths=0)

    # y = x is where a perfect model would put every point.
    limit = max(max(y_true), max(y_pred))
    ax.plot([0, limit], [0, limit], "--", linewidth=2, color=REFERENCE_COLOR, label="Perfect prediction")

    # Predictions below zero are physically impossible -- an engine cannot
    # have negative life left -- so mark the boundary when any cross it.
    if min(y_pred) < 0:
        ax.axhline(0, linewidth=1.5, color="#e34948", label="Impossible (negative RUL)")

    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    return ax


def plot_clip_comparison(uncapped, capped, figures_dir="figures"):
    """Linear regression predicted-vs-actual, uncapped beside capped at 150.

    Each argument is the (y_true, y_pred) pair from the same held-out fold.
    Side by side they show what the clip did: the uncapped panel fans out
    above ~150 where the sensors no longer separate the engines, and that
    region is what the train/validation gap was made of.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8))

    plot_predicted_vs_actual(*uncapped, "Uncapped target", left)
    plot_predicted_vs_actual(*capped, "Target capped at 150 cycles", right)

    fig.suptitle("Linear regression on held-out engines", fontsize=12.5)
    fig.tight_layout()
    return save_figure(fig, "linear_regression_clip", figures_dir)


def _plot_sweep(ax, sweep, parameter, xlabel, title, chosen):
    """Draw one training/validation RMSE curve pair with std bands.

    `chosen` is the setting this project settled on; it gets a marker so the
    figure says what was picked, not just what was tried.
    """
    _style(ax)
    for column, color, label in [
        ("train_rmse", TRAIN_COLOR, "Training"),
        ("val_rmse", VALIDATION_COLOR, "Validation"),
    ]:
        mean, std = sweep[f"{column}_mean"], sweep[f"{column}_std"]
        ax.plot(sweep["x"], mean, "o-", linewidth=2, markersize=8, color=color, label=label)
        # The band is +/- 1 std across the five folds. It is wide relative to
        # the movement in the validation curve, which is why the text reports
        # the paired per-fold comparison rather than resting on these means.
        ax.fill_between(sweep["x"], mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    chosen_x = sweep.loc[sweep[parameter] == str(chosen), "x"].iloc[0]
    ax.axvline(chosen_x, color=REFERENCE_COLOR, linewidth=1, linestyle=":", zorder=0)
    ax.annotate(
        f"chosen: {chosen}",
        xy=(chosen_x, ax.get_ylim()[0]),
        xytext=(4, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=REFERENCE_COLOR,
    )

    ax.set_xticks(sweep["x"])
    ax.set_xticklabels(sweep[parameter])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return ax


def _label_validation_ends(ax, sweep):
    """Print the validation RMSE at the best and the largest setting.

    The shared y-axis is set by the forest's training curve, which drops to
    about 9 cycles, so boosting's rise past its minimum is real but too small
    to see. The numbers carry it where the pixels cannot.
    """
    best = sweep["val_rmse_mean"].idxmin()
    last = sweep.index[-1]
    for row, offset in [(best, 10), (last, 10)]:
        ax.annotate(
            f"{sweep.loc[row, 'val_rmse_mean']:.2f}",
            xy=(sweep.loc[row, "x"], sweep.loc[row, "val_rmse_mean"]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=VALIDATION_COLOR,
        )


def plot_complexity_sweeps(rf_sweep, gb_sweep, figures_dir="figures"):
    """The two complexity sweeps on a shared y-axis.

    Sharing the axis is the point of putting them together: it makes the two
    shapes directly comparable. The forest's validation curve flattens and
    stays flat, because bagging averages the individual trees' errors away.
    Boosting's turns back up, because each tree fits the last ensemble's
    residuals and eventually that means fitting noise.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    _plot_sweep(left, rf_sweep, "max_depth", "max_depth", "Random Forest", chosen=10)
    _plot_sweep(
        right, gb_sweep, "n_estimators", "n_estimators (log scale)", "Gradient Boosting", chosen=200
    )
    _label_validation_ends(right, gb_sweep)

    # Log scale because the grid spans 50 to 1000; on a linear axis the four
    # smaller values would bunch against the left edge.
    right.set_xscale("log")
    right.set_xticks(gb_sweep["x"])
    right.set_xticklabels(gb_sweep["n_estimators"])
    right.minorticks_off()

    left.set_ylabel("RMSE (cycles)")
    fig.suptitle("Complexity sweeps, 5-fold engine-grouped CV", fontsize=12.5)
    fig.tight_layout()
    return save_figure(fig, "complexity_sweeps", figures_dir)


def plot_model_comparison(cv_summary, figures_dir="figures"):
    """Bar chart of cross-validated validation RMSE, one bar per model.

    Error bars are the standard deviation across the five folds. They overlap
    for the two tree models, which is the finding: those two are not
    separable on 100 engines.
    """
    validation = cv_summary[cv_summary["split"] == "validation"]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    _style(ax)

    bars = ax.bar(
        validation["model"],
        validation["rmse_mean"],
        yerr=validation["rmse_std"],
        width=0.6,
        color=TRAIN_COLOR,
        capsize=5,
        error_kw={"ecolor": REFERENCE_COLOR, "elinewidth": 1.5},
    )

    # Label each bar directly: three values is few enough that the reader
    # should not have to measure them against the axis. Sit the label clear of
    # the error bar's top cap rather than the bar's own height.
    for bar, value, spread in zip(bars, validation["rmse_mean"], validation["rmse_std"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + spread + 1.0,
            f"{value:.2f}",
            ha="center",
            fontsize=10,
            color="#0b0b0b",
        )

    ax.set_ylabel("Validation RMSE (cycles)")
    ax.set_ylim(0, validation["rmse_mean"].max() * 1.3)
    ax.set_title("Model comparison, 5-fold engine-grouped CV\n(lower is better)", fontsize=11.5)
    fig.tight_layout()
    return save_figure(fig, "model_comparison", figures_dir)


def save_figure(fig, name, figures_dir="figures"):
    """Write a figure to figures_dir/<name>.png and close it."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
