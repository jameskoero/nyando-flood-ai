"""
Feature engineering for Nyando Flood AI.
"""

import pandas as pd


def add_rainfall_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin 3-day rainfall accumulation into intensity categories.

    These bins are this project's own reasoned choice, not an external
    standard — a prior version of this docstring incorrectly attributed
    them to Kenya Meteorological Department classes; KMD's actual
    published thresholds (light <5mm, moderate 5-20mm, heavy 21-50mm,
    very heavy >50mm) are for 24-hour totals with no direct project-
    specific 3-day equivalent, so no external authority is claimed here.

    Bins (mm):  (-inf, 30) = 0 dry
                [30,  60)  = 1 light
                [60,  90)  = 2 moderate
                [90, 120)  = 3 heavy
                [120, inf) = 4 extreme
    """
    out = df.copy()
    out["rainfall_cat"] = pd.cut(
        out["rainfall_3day"],
        bins=[-float("inf"), 30, 60, 90, 120, float("inf")],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(int)
    return out


def add_flood_plain_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute flood plain index = 1 / (elevation * (slope + 1)).

    Both elevation and slope in the denominator: lower elevation AND
    lower slope both increase the index, matching "higher index = higher
    flood risk" and the project's own monotonic constraints (elevation
    down, slope down -> flood probability up). A previous version had
    elevation in the numerator, which inverted this relationship — caught
    because the previous test only checked sign (>0), never direction.
    """
    out = df.copy()
    out["flood_plain_index"] = 1.0 / (
        out["elevation"] * (out["slope"].clip(lower=0.1) + 1)
    )
    return out


def add_soil_permeability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin clay percentage into soil permeability class.

    Bins:  [0, 25)  clay_percent → 2 (high permeability — drains well)
           [25, 40) clay_percent → 1 (medium)
           [40, inf)clay_percent → 0 (low permeability — waterlogging risk)
    """
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
    """Legacy alias. Prefer build_all_features()."""
    return build_all_features(df)
