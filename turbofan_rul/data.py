"""Loading the C-MAPSS FD001 files and building the RUL target.

The raw files have no header and are separated by runs of spaces, so the
column names come from the dataset's readme and are applied at load time.
"""

from pathlib import Path

import pandas as pd

# Column order is fixed by the dataset readme: unit id, time in cycles,
# three operational settings, then 21 sensor measurements.
COLUMNS = ["ID", "Cycle", "Op S1", "Op S2", "Op S3"] + [f"S{i}" for i in range(1, 22)]

TARGET = "RUL"

# ID is an arbitrary engine label and Cycle is what RUL is computed from, so
# leaving either in the feature matrix would leak the answer.
LEAKY_COLUMNS = ["ID", "Cycle"]

# S6 has two unique values (21.61 on 20,225 rows, 21.6 on 406) so nunique()==1
# does not catch it, but 98% of one value carries no usable degradation signal.
# It is dropped by name alongside the genuinely constant columns.
NEAR_CONSTANT_COLUMNS = ["S6"]


def load_raw(path):
    """Read one of the raw 26-column files into a named DataFrame.

    `sep=r"\\s+"` collapses the runs of spaces that would otherwise read as
    empty columns.
    """
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_true_rul(path):
    """Read RUL_<subset>.txt into an ID-to-RUL frame.

    The file is one RUL value per line with no header, ordered by engine, so
    the row index plus one is the engine ID. Making that mapping explicit
    stops it being an unstated assumption later on.
    """
    true_rul = pd.read_csv(path, header=None, names=[TARGET])
    true_rul["ID"] = true_rul.index + 1
    return true_rul


def add_rul(df, clip=None):
    """Add the RUL target to a training frame.

    Every training engine runs to failure, so its last cycle is its failure
    cycle and RUL is the distance back from there. `clip` caps the target:
    engines far from failure have near-identical sensor readings, so high RUL
    is unlearnable and an uncapped target spends the model's capacity on it.
    """
    df = df.copy()
    # transform("max") broadcasts each engine's failure cycle back over all of
    # that engine's rows, so the subtraction stays row-wise.
    failure_cycle = df.groupby("ID")["Cycle"].transform("max")
    df[TARGET] = failure_cycle - df["Cycle"]
    if clip is not None:
        df[TARGET] = df[TARGET].clip(upper=clip)
    return df


def constant_columns(df):
    """Columns holding a single value across the whole frame.

    A sensor that never moves is either unused or broken in this subset; it
    cannot explain variation in RUL and only adds width to the matrix.
    """
    counts = df.nunique()
    return sorted(counts[counts == 1].index)


def dead_columns(df):
    """The columns to drop: detected constants plus the near-constant list."""
    dead = constant_columns(df)
    dead += [c for c in NEAR_CONSTANT_COLUMNS if c in df.columns and c not in dead]
    return sorted(dead)


def feature_columns(df):
    """The model inputs: every remaining column that is not a label or the target."""
    excluded = set(LEAKY_COLUMNS) | {TARGET}
    return [c for c in df.columns if c not in excluded]


def last_cycle_per_engine(df):
    """Keep only each engine's final observed row.

    The test engines are truncated mid-life and RUL_FD001.txt gives the true
    remaining life at that truncation point, so this is the only row per
    engine the published labels describe.
    """
    return df.groupby("ID").last().reset_index()


def load_dataset(data_dir, subset="FD001", clip=150):
    """Load train and test sets ready for modelling.

    Returns (train, test, features). `train` is every cycle of the 100
    run-to-failure engines with RUL attached; `test` is one row per truncated
    test engine -- its last observed cycle -- carrying the true RUL from
    RUL_FD001.txt. Both targets are clipped identically so the numbers are
    comparable.

    Dead columns are detected on the training set and the same list is
    dropped from the test set, so the two matrices stay aligned and nothing
    about the test data influences the feature choice.
    """
    data_dir = Path(data_dir) / subset

    train = add_rul(load_raw(data_dir / f"train_{subset}.txt"), clip=clip)
    dead = dead_columns(train)
    train = train.drop(columns=dead)

    test = load_raw(data_dir / f"test_{subset}.txt").drop(columns=dead)
    test = last_cycle_per_engine(test)

    true_rul = load_true_rul(data_dir / f"RUL_{subset}.txt")
    if clip is not None:
        true_rul[TARGET] = true_rul[TARGET].clip(upper=clip)
    test = test.merge(true_rul, on="ID", validate="one_to_one")

    return train, test, feature_columns(train)
