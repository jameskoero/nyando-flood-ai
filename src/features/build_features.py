import pandas as pd


def add_rainfall_categories(df):
    out = df.copy()
    out["rainfall_cat"] = pd.cut(
        out["rainfall_3day"],
        bins=[-float("inf"), 50, 100, 150, 200, float("inf")],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(int)
    out["rain_cat"] = out["rainfall_cat"]
    return out


def add_flood_plain_index(df):
    out = df.copy()
    out["flood_plain_index"] = out["elevation"] / (out["slope"].clip(lower=0.1) + 1)
    out["flood_plain"] = out["flood_plain_index"]
    return out


def add_soil_permeability(df):
    out = df.copy()
    out["soil_permeability"] = pd.cut(
        out["clay_percent"],
        bins=[-float("inf"), 30, 45, float("inf")],
        labels=[2, 1, 0],
        include_lowest=True,
    ).astype(int)
    out["soil_perm"] = out["soil_permeability"]
    return out


def build_all_features(df):
    out = add_rainfall_categories(df)
    out = add_flood_plain_index(out)
    out = add_soil_permeability(out)
    return out


def build(df):
    return build_all_features(df)
