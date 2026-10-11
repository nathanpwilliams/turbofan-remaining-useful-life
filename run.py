"""Run the whole FD001 analysis: cross-validate, sweep, test, save.

    python run.py --subset FD001

Writes results/metrics_cv.csv, results/metrics_test.csv,
results/sweep_rf.csv, results/sweep_gb.csv and figures/*.png.
"""

import argparse
from pathlib import Path

from turbofan_rul import data, evaluate, models, plots


def parse_args(argv=None):
    """Command-line arguments for a full run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument(
        "--clip",
        type=int,
        default=150,
        help="cap on the RUL target in cycles; above this the sensors do not separate engines",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the complexity sweeps (--no-sweep to skip; they are the slow part)",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args(argv)


def save_table(frame, name, results_dir):
    """Write a results table to CSV and echo it to stdout."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.csv"
    frame.to_csv(path, index=False)

    print(f"\n{name}  ->  {path}")
    print(frame.to_string(index=False))
    return path


def run(args):
    """Load, cross-validate, sweep, test, save. Returns the results tables."""
    train, test, features = data.load_dataset(args.data_dir, args.subset, clip=args.clip)
    print(
        f"{args.subset}: {len(train):,} training rows from {train['ID'].nunique()} engines, "
        f"{len(test)} test engines, {len(features)} features, target clipped at {args.clip}"
    )

    # Cross-validation on the training engines. This is the number to compare
    # models on, because every model sees exactly the same folds.
    cv_metrics = evaluate.cross_validate_all(models.all_models(), train, features, args.folds)
    save_table(cv_metrics, "metrics_cv", args.results_dir)

    # Final scoring: fit on all 100 training engines, predict the last
    # observed cycle of each of the 100 unseen test engines.
    test_metrics, test_predictions = evaluate.evaluate_on_test(
        models.all_models(), train, test, features
    )
    save_table(test_metrics, "metrics_test", args.results_dir)

    # Figure 1: what clipping the target actually did to the linear baseline.
    # Both panels come from the same held-out fold, so only the target differs.
    uncapped_train, _, _ = data.load_dataset(args.data_dir, args.subset, clip=None)
    uncapped_fold = evaluate.first_fold_predictions(
        models.linear_regression(), uncapped_train, features, args.folds
    )
    capped_fold = evaluate.first_fold_predictions(
        models.linear_regression(), train, features, args.folds
    )
    plots.plot_clip_comparison(uncapped_fold, capped_fold, figures_dir=args.figures_dir)

    # Negative predicted RUL is impossible -- an engine cannot have less than
    # no life left -- and is the clearest symptom of fitting a plane to curved
    # data. Counted on the held-out fold the figure plots, which covers every
    # cycle; the 100-row test snapshot excludes the near-failure cycles where
    # the linear fit undershoots hardest.
    _, fold_predictions = capped_fold
    print(
        f"\nLinear Regression predicts negative RUL on "
        f"{(fold_predictions < 0).sum()} of {len(fold_predictions)} held-out rows "
        f"(minimum {fold_predictions.min():.1f} cycles); "
        f"on the {len(test)}-engine test snapshot the minimum is "
        f"{test_predictions['Linear Regression'].min():.1f}"
    )

    # Figure 2: the model comparison bar chart.
    plots.plot_model_comparison(cv_metrics, figures_dir=args.figures_dir)

    if args.sweep:
        # Figure 3: the complexity sweeps. Slowest step by far -- boosting
        # refits up to 1000 trees for every one of the five folds.
        rf_sweep = evaluate.sweep_parameter(
            models.random_forest, "max_depth", models.RF_DEPTH_GRID, train, features, args.folds
        )
        save_table(rf_sweep.drop(columns="x"), "sweep_rf", args.results_dir)

        gb_sweep = evaluate.sweep_parameter(
            models.gradient_boosting,
            "n_estimators",
            models.GB_ESTIMATOR_GRID,
            train,
            features,
            args.folds,
        )
        save_table(gb_sweep.drop(columns="x"), "sweep_gb", args.results_dir)

        plots.plot_complexity_sweeps(rf_sweep, gb_sweep, figures_dir=args.figures_dir)

    print(f"\nFigures written to {args.figures_dir}/")
    return cv_metrics, test_metrics


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
