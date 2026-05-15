"""
Data loading for Nyando Flood AI.

Key design decision: validate schema at load time.

Google Earth Engine exports CSVs with a UTF-8 BOM (\ufeff) prepended
to the first column name. If we don't strip it, the first column is
named "\ufeffelevation" instead of "elevation", which silently breaks
every downstream operation that references "elevation" by name.

I discovered this when the pipeline was loading data fine visually but
the schema check kept failing. Printing df.columns.tolist() showed the
invisible \ufeff character.

The _validate() function strips this at read time so the rest of the
pipeline never has to think about it.
"""

import pandas as pd
from pathlib import Path

TARGET = "flooded"

# These are the minimum columns the training CSV must have.
# lat/lon are needed for spatial cross-validation and map outputs.
FEATURES = [
    "elevation",
    "slope",
    "rainfall_3day",
    "distance_river",
    "clay_percent",
    "land_cover",
]
REQUIRED_COLUMNS = {"lon", "lat", TARGET, *FEATURES}

# Public GitHub raw URL as fallback if local CSV is missing
GITHUB_RAW = (
    "https://raw.githubusercontent.com/jameskoero/nyando-flood-ai"
    "/main/data/training/nyando_training_v1.csv"
)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip BOM characters from column names and verify required columns.

    BOM (\ufeff) is a Unicode byte-order mark that GEE prepends to the
    first column. It's invisible in most text editors but breaks string
    matching. We strip it from all columns defensively (not just the first).
    """
    # Strip BOM from all column names
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"[load_data] Missing required columns: {missing}\n"
            f"  Found: {sorted(df.columns)}"
        )
    return df


def load_training_csv() -> pd.DataFrame:
    """
    Load the canonical curated training CSV.

    Local path is preferred (faster, works offline).
    Falls back to GitHub raw URL for CI environments where the file
    may not be cached.
    """
    local = (
        Path(__file__).parent.parent.parent
        / "data" / "training" / "nyando_training_v1.csv"
    )
    if local.exists():
        return _validate(pd.read_csv(local))
    print("[load_data] Fetching from GitHub...")
    return _validate(pd.read_csv(GITHUB_RAW))


def load_raw_gee() -> pd.DataFrame:
    """
    Load the raw unfiltered GEE export for diagnostic purposes.

    This file has 5000 rows and 1150 flood positives.
    It is NOT used for model training — only for auditing the
    label generation step that produced nyando_training_v1.csv.
    """
    local = (
        Path(__file__).parent.parent.parent
        / "data" / "training" / "nyando_training_v1_raw_gee.csv"
    )
    if local.exists():
        return pd.read_csv(local)
    raise FileNotFoundError(
        "[load_data] Raw GEE CSV not found. "
        "Download from the GEE export bucket."
    )
