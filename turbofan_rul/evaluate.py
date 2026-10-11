"""Metrics and the cross-validation loops.

RMSE, MAE and R2 are symmetric: they charge the same for predicting 20
cycles early as 20 cycles late. The PHM08 score is not, and that asymmetry
is what maintenance planning actually cares about.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .split import group_kfold_splits, split_xy

METRICS = ["rmse", "mae", "r2", "phm08"]


def rmse(y_true, y_pred):
    """Root mean squared error, in cycles. Lower is better."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean absolute error, in cycles.

    Read next to RMSE: a much larger RMSE means the error is concentrated in
    a few big misses rather than spread evenly.
    """
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    """Share of RUL variance the model explains. Higher is better."""
    return r2_score(y_true, y_pred)


def phm08_score(y_true, y_pred, early=13.0, late=10.0):
    """The PHM08 Data Challenge score. Lower is better.

    For d = predicted - true the penalty is exp(-d/13) - 1 when the
    prediction is early (d < 0) and exp(d/10) - 1 when it is late (d >= 0).
    The smaller divisor on the late branch makes that side climb faster, so
    overestimating remaining life -- letting an engine run past its failure
    point -- costs more than pulling it early.

    This is a sum, as in the original challenge, so it grows with the number
    of rows scored and is only comparable between runs over the same rows.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.exp(-d / early) - 1, np.exp(d / late) - 1)
    return float(penalty.sum())


def all_metrics(y_true, y_pred):
    """All four metrics for one set of predictions, keyed by name."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "phm08": phm08_score(y_true, y_pred),
    }


def cross_validate_model(model, df, features, n_splits=5):
    """Score one model across the engine-grouped folds.

    Returns one row per (fold, split) with all four metrics. Training scores
    come back alongside validation scores because the gap between them is
    what distinguishes overfitting from plain bias.

    Each fold fits a clone, so no state survives from the previous fold.
    """
    rows = []
    for fold, (train_index, val_index) in enumerate(group_kfold_splits(df, n_splits), start=1):
        train_x, train_y = split_xy(df.iloc[train_index], features)
        val_x, val_y = split_xy(df.iloc[val_index], features)

        fitted = clone(model).fit(train_x, train_y)

        for split_name, (y_true, y_pred) in {
            "train": (train_y, fitted.predict(train_x)),
            "validation": (val_y, fitted.predict(val_x)),
        }.items():
            rows.append({"fold": fold, "split": split_name, **all_metrics(y_true, y_pred)})

    return pd.DataFrame(rows)


def summarise(fold_metrics):
    """Collapse per-fold metrics into a mean and standard deviation per split.

    The standard deviation across folds is the honest error bar here: with
    only 100 engines, differences smaller than it are noise.
    """
    return (
        fold_metrics.groupby("split")[METRICS]
        .agg(["mean", "std"])
        .pipe(lambda out: out.set_axis([f"{m}_{s}" for m, s in out.columns], axis=1))
        .reset_index()
    )


def cross_validate_all(models, df, features, n_splits=5):
    """Cross-validate every model in a {name: model} mapping.

    Returns the concatenated per-split summaries with a `model` column.
    """
    summaries = []
    for name, model in models.items():
        summary = summarise(cross_validate_model(model, df, features, n_splits))
        summary.insert(0, "model", name)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def sweep_parameter(build_model, parameter, values, df, features, n_splits=5):
    """Cross-validate one model across one hyperparameter's values.

    `build_model` is called as build_model(**{parameter: value}) for each
    value. Returns training and validation RMSE, mean and std over folds, so
    the two curves can be read against each other.
    """
    rows = []
    for value in values:
        folds = cross_validate_model(build_model(**{parameter: value}), df, features, n_splits)
        summary = folds.groupby("split")["rmse"].agg(["mean", "std"])
        rows.append(
            {
                parameter: str(value),
                "train_rmse_mean": summary.loc["train", "mean"],
                "train_rmse_std": summary.loc["train", "std"],
                "val_rmse_mean": summary.loc["validation", "mean"],
                "val_rmse_std": summary.loc["validation", "std"],
            }
        )

    sweep = pd.DataFrame(rows)
    # max_depth=None means unlimited depth, which has no place on a numeric
    # axis. Give it a position just past the largest real value so it plots as
    # the end of the curve, and keep the true label in the parameter column.
    numeric = pd.to_numeric(sweep[parameter], errors="coerce")
    sweep["x"] = numeric.fillna(numeric.max() + 5)
    return sweep


def evaluate_on_test(models, train, test, features):
    """Fit each model on all 100 training engines and score the test engines.

    The test frame is one row per engine -- its last observed cycle -- scored
    against RUL_FD001.txt. Returns (metrics, predictions): a frame of one row
    per model, and a {name: predicted array} mapping for the figures.
    """
    train_x, train_y = split_xy(train, features)
    test_x, test_y = split_xy(test, features)

    rows, predictions = [], {}
    for name, model in models.items():
        prediction = clone(model).fit(train_x, train_y).predict(test_x)
        predictions[name] = prediction
        rows.append({"model": name, **all_metrics(test_y, prediction)})

    return pd.DataFrame(rows), predictions


def first_fold_predictions(model, df, features, n_splits=5):
    """Fit on the first grouped fold and predict its held-out engines.

    Used for the predicted-vs-actual figures, which need out-of-sample
    predictions from a single consistent split rather than a CV average.
    """
    train_index, val_index = next(group_kfold_splits(df, n_splits))
    train_x, train_y = split_xy(df.iloc[train_index], features)
    val_x, val_y = split_xy(df.iloc[val_index], features)

    prediction = clone(model).fit(train_x, train_y).predict(val_x)
    return val_y, prediction
