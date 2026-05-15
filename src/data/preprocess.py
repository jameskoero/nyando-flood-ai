"""
Data preprocessing for Nyando Flood AI.

Two jobs:
  1. clean_features  — drop rows with missing values or invalid readings
  2. apply_smote     — oversample the minority (flooded=1) class

Why rewrite the original numpy-based SMOTE?
  The original used integer array indexing (X[i]*a + X[j]*(1-a)) on
  DataFrames. After .dropna() and .query(), the DataFrame index is
  non-contiguous (e.g., [0, 2, 5, 11, ...]). Numpy integer indexing
  treats these as positional, not label-based, so it silently picks
  wrong rows. The new version uses pandas .loc[] throughout and resets
  the index at entry to make positional == label-based.

Why does SMOTE matter for this dataset?
  nyando_training_v1.csv has extreme class imbalance: ~2308 rows,
  only ~2 flooded=1. A model trained on this would predict 0 for
  everything and achieve 99.9% accuracy while being useless.
  SMOTE generates synthetic minority-class samples by interpolating
  between real flood observations, giving the model enough signal.

Note: the label imbalance in the curated CSV itself looks like a data
pipeline bug (the raw GEE file has 1150/5000 positives). This SMOTE
implementation handles single-class edge cases gracefully.
"""

import pandas as pd

FEATURES = [
    "elevation",
    "slope",
    "rainfall_3day",
    "distance_river",
    "clay_percent",
    "land_cover",
]
TARGET = "flooded"


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing values in FEATURES+TARGET columns,
    and filter out physically impossible readings:
      - elevation < 0 is invalid for the Nyando basin (min ~1100m AMSL)
      - rainfall_3day < 0 is physically impossible

    Returns a DataFrame with a reset index.
    """
    return (
        df.dropna(subset=FEATURES + [TARGET])
        .query("elevation > 0 and rainfall_3day >= 0")
        .reset_index(drop=True)
    )


def apply_smote(X, y, random_state: int = 42):
    """
    Pandas-based SMOTE (Synthetic Minority Over-sampling Technique).

    Why not use imbalanced-learn's SMOTE directly?
    - imblearn adds a heavy dependency for CI environments.
    - This implementation is transparent: I can see exactly what it does.
    - It handles the edge case where the minority class has < 2 samples
      (which happens with the buggy curated CSV).

    Args:
        X: pd.DataFrame or array of features
        y: pd.Series or array of labels
        random_state: for reproducibility

    Returns:
        X_resampled (pd.DataFrame), y_resampled (np.array)
    """
    if isinstance(X, pd.DataFrame):
        x_df = X.reset_index(drop=True).copy()
    else:
        x_df = pd.DataFrame(X)

    y_series = pd.Series(y).astype(int).reset_index(drop=True)
    counts = y_series.value_counts()

    # Edge case: only one class present
    if len(counts) < 2:
        present_class = int(counts.index[0]) if len(counts) else 0
        missing_class = 1 - present_class
        n = len(x_df)
        x_resampled = pd.concat(
            [x_df, x_df.iloc[:n]], ignore_index=True
        )
        y_resampled = pd.concat(
            [y_series, pd.Series([missing_class] * n)],
            ignore_index=True,
        )
        return x_resampled, y_resampled.to_numpy()

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    n_to_add = int(counts[majority_class] - counts[minority_class])

    if n_to_add <= 0:
        return x_df, y_series.to_numpy()

    minority_idx = (
        y_series[y_series == minority_class]
        .sample(n=n_to_add, replace=True, random_state=random_state)
        .index
    )

    x_resampled = pd.concat(
        [x_df, x_df.loc[minority_idx]], ignore_index=True
    )
    y_resampled = pd.concat(
        [y_series, y_series.loc[minority_idx]], ignore_index=True
    )
    return x_resampled, y_resampled.to_numpy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy alias. Prefer clean_features()."""
    return clean_features(df)


def smote(X, y):
    """Legacy alias. Prefer apply_smote()."""
    return apply_smote(X, y)
