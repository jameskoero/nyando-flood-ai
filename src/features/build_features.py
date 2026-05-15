import pandas as pd


def add_rainfall_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rainfall_cat"] = pd.cut(
        out["rainfall_3day"],
        bins=[-float("inf"), 30, 60, 90, 120, float("inf")],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(int)
    return out


def add_flood_plain_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flood_plain_index"] = out["elevation"] / (out["slope"].clip(lower=0.1) + 1)
    return out


def add_soil_permeability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["soil_permeability"] = pd.cut(
        out["clay_percent"],
        bins=[-float("inf"), 25, 40, float("inf")],
        labels=[2, 1, 0],
        include_lowest=True,
    ).astype(int)
    return out


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_rainfall_categories(df)
    out = add_flood_plain_index(out)
    out = add_soil_permeability(out)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    return build_all_features(df)
