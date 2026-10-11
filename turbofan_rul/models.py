"""The three regressors, with the settings the complexity sweeps chose.

No feature scaling anywhere: the trees are invariant to it, and unscaled
coefficients keep the linear baseline readable in sensor units.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

RANDOM_STATE = 27

# The grids the sweeps walk, one knob per tree model.
RF_DEPTH_GRID = [3, 5, 8, 10, 15, 20, None]
GB_ESTIMATOR_GRID = [50, 100, 200, 500, 1000]


def linear_regression():
    """Ordinary least squares, the baseline the other two have to beat.

    Deliberately unregularized: the train/validation gap is under two cycles
    once the target is clipped, so there is no variance for a penalty to
    remove. This model's error is bias, and shrinking coefficients would only
    add to it.
    """
    return LinearRegression()


def random_forest(n_estimators=100, max_depth=10):
    """Bagged trees. Depth 10 is where the validation curve flattens.

    n_jobs=-1 only affects runtime, not the fit.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def gradient_boosting(n_estimators=200, max_depth=3, learning_rate=0.1):
    """Boosted trees. 200 estimators is the minimum of the validation curve.

    Each tree fits the previous ensemble's residuals, so unlike the forest
    this one does not plateau -- past the minimum the extra trees start
    fitting noise.
    """
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
    )


def all_models():
    """The three models, keyed by display name, in increasing complexity."""
    return {
        "Linear Regression": linear_regression(),
        "Random Forest": random_forest(),
        "Gradient Boosting": gradient_boosting(),
    }
