"""Cross-validation splits that keep whole engines together.

Consecutive cycles from one engine are near-duplicates of each other, so a
split that cuts across rows puts cycle 100 in training and cycle 101 in
validation and reports a score the model has not earned. Every split here
groups on engine ID.
"""

from sklearn.model_selection import GroupKFold

from .data import TARGET


def group_kfold_splits(df, n_splits=5):
    """Yield (train_index, validation_index) pairs with engines held whole.

    GroupKFold is deterministic -- it takes no random_state, because it
    assigns groups to folds by size rather than by shuffling.
    """
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(df, groups=df["ID"])


def split_xy(df, features):
    """Split a frame into the feature matrix and the RUL target."""
    return df[features], df[TARGET]
