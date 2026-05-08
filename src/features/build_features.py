"""
build_features.py — Feature engineering for Nyando Flood AI.
"""
import pandas as pd


def add_rainfall_categories(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 20, 50, 80, 120, float("inf")]
    labels = [0, 1, 2, 3, 4]
    df = df.copy()
    df["rainfall_cat"] = pd.cut(
        df["rainfall_3day"], bins=bins, labels=labels, right=False
    ).astype(int)
    return df


def add_flood_plain_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flood_plain_index"] = (
        (1.0 / df["elevation"].clip(lower=1)) *
        (1.0 / (df["slope"].clip(lower=0) + 1)) *
        (1.0 / (df["distance_river"].clip(lower=1) + 1))
    )
    return df


def add_soil_permeability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soil_permeability"] = pd.cut(
        df["clay_percent"], bins=[0,25,45,100],
        labels=[0,1,2], right=False
    ).astype(int)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rainfall_categories(df)
    df = add_flood_plain_index(df)
    df = add_soil_permeability(df)
    print(f"[features] Built {len(df.columns)} total columns")
    return df
