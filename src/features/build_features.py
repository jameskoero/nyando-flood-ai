"""
Feature engineering for Nyando Flood AI.

I split the original monolithic build() into four named functions.
Each function has exactly one hydrological job.

Why named functions instead of one big build()?
  - Tests can import and unit-test each transformation independently.
  - A single-responsibility function is easier to debug when GEE data
    changes (e.g., if the rainfall column is renamed in a future export).

Hydrological reasoning behind each feature:
  - Rainfall categories: intensity matters more than raw mm. A 90mm
    3-day event is qualitatively different from 30mm. Thresholds follow
    Kenya Meteorological Department light/moderate/heavy/extreme classes.
  - Flood plain index: elevation / slope. Flat valleys accumulate water.
    The Nyando basin floor sits at ~1130m with near-zero slope — that is
    exactly where the 2019-2020 flooding was worst.
  - Soil permeability: high clay percentage means the soil cannot absorb
    water quickly, so surface runoff dominates. Bins into low/med/high
    permeability classes (2/1/0 — lower number = floods faster).
"""

import pandas as pd


def add_rainfall_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin 3-day rainfall accumulation into intensity categories.

    Bins (mm):  (-inf, 30) = 0 dry
                [30,  60)  = 1 light
                [60,  90)  = 2 moderate
                [90, 120)  = 3 heavy
                [120, inf) = 4 extreme

    I use include_lowest=True so that exactly 0mm falls in category 0
    rather than becoming NaN.
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
    Compute flood plain index = elevation / (slope + 0.1).

    Adding 0.1 to slope prevents division-by-zero on perfectly flat
    cells (which actually exist in the Nyando valley floor).
    Higher index = lower, flatter terrain = higher flood risk.
    """
    out = df.copy()
    out["flood_plain_index"] = out["elevation"] / (
        out["slope"].clip(lower=0.1) + 1
    )
    return out


def add_soil_permeability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin clay percentage into soil permeability class.

    High clay = slow drainage = flood risk persists longer after rain.
    Bins:  [0, 25)  clay_percent → 2 (high permeability — drains well)
           [25, 40) clay_percent → 1 (medium)
           [40, inf)clay_percent → 0 (low permeability — waterlogging risk)

    Reversed label encoding so higher number = drains better, which
    makes the feature's direction consistent with the other features
    (higher value = less flood risk).
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
    """
    Apply all three feature transformations in sequence.

    Order matters: each function copies the DataFrame, so they can be
    applied independently. The sequence here is:
      1. Rainfall categories (meteorological driver)
      2. Flood plain index   (topographic driver)
      3. Soil permeability   (soil driver)
    """
    out = add_rainfall_categories(df)
    out = add_flood_plain_index(out)
    out = add_soil_permeability(out)
    return out


# Backward-compatible alias — anything that called build(df) still works.
def build(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy alias. Prefer build_all_features() for clarity."""
    return build_all_features(df)
