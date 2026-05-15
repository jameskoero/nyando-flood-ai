import pandas as pd

FEATURES = ["elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover"]
TARGET = "flooded"


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=FEATURES + [TARGET])
        .query("elevation > 0 and rainfall_3day >= 0")
        .reset_index(drop=True)
    )


def apply_smote(X, y, random_state: int = 42):
    if isinstance(X, pd.DataFrame):
        x_df = X.reset_index(drop=True).copy()
    else:
        x_df = pd.DataFrame(X)
    y_series = pd.Series(y).astype(int).reset_index(drop=True)

    counts = y_series.value_counts()
    if len(counts) < 2:
        present_class = int(counts.index[0]) if len(counts) else 0
        missing_class = 1 - present_class
        x_resampled = pd.concat([x_df, x_df.copy()], ignore_index=True)
        y_resampled = pd.concat(
            [y_series, pd.Series([missing_class] * len(x_df), dtype=int)],
            ignore_index=True,
        )
        return x_resampled, y_resampled.to_numpy()

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    n_to_add = int(counts[majority_class] - counts[minority_class])
    if n_to_add <= 0:
        return x_df, y_series.to_numpy()

    sampled_idx = (
        y_series[y_series == minority_class]
        .sample(n=n_to_add, replace=True, random_state=random_state)
        .index
    )
    x_resampled = pd.concat([x_df, x_df.loc[sampled_idx]], ignore_index=True)
    y_resampled = pd.concat([y_series, y_series.loc[sampled_idx]], ignore_index=True)
    return x_resampled, y_resampled.to_numpy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean_features(df)


def smote(X, y):
    return apply_smote(X, y)
