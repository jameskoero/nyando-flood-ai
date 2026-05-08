"""
preprocess.py — Feature cleaning and class-balancing for Nyando Flood AI.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE

FEATURES = ["elevation","slope","rainfall_3day",
            "distance_river","clay_percent","land_cover"]
TARGET   = "flooded"


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df["elevation"] > 0]
    df = df[df["rainfall_3day"] >= 0]
    df = df.reset_index(drop=True)
    print(f"[preprocess] clean_features: {before:,} -> {len(df):,} rows")
    return df


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    before = y.value_counts().to_dict()
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    after = pd.Series(y_res).value_counts().to_dict()
    print(f"[preprocess] SMOTE: {before} -> {after}")
    return X_res, y_res


def temporal_split(df, train_years=range(2014,2023), test_years=range(2023,2025)):
    if "year" not in df.columns:
        raise KeyError("DataFrame must contain a 'year' column.")
    df_train = df[df["year"].isin(train_years)].reset_index(drop=True)
    df_test  = df[df["year"].isin(test_years)].reset_index(drop=True)
    print(f"[preprocess] Temporal split — train: {len(df_train):,} | test: {len(df_test):,}")
    return df_train, df_test
