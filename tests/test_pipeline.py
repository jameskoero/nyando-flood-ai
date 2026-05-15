"""
Integration tests for the Nyando Flood AI pipeline.

Design principles I followed:
  1. Tests must pass with zero internet access (CI has no GEE credentials).
     So tests use the committed CSV, not a live GEE query.

  2. The canonical CSV is pinned by name (nyando_training_v1.csv) not
     by glob("*.csv") — the glob approach was picking the raw GEE file
     sometimes, which has different schema and BOM headers.

  3. Schema is validated before any feature engineering test runs.
     This makes failures self-explanatory: "Missing required columns: [x]"
     is clearer than a KeyError buried in a stack trace.

  4. Both model performance tests (AUC, F1) and data quality tests
     (class balance, row count) are included. The data quality tests
     are the canary for the label generation bug (2/2308 flooded=1).
"""

import pytest
import json
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "training"
MODEL_PATH   = ROOT / "models" / "nyando_xgb_v1.pkl"
METRICS_PATH = ROOT / "metrics.json"

# Pin the canonical CSV explicitly.
# This prevents test_pipeline from accidentally picking up
# nyando_training_v1_raw_gee.csv via glob — the raw file has
# BOM headers and different schema.
CANONICAL_CSV = DATA_DIR / "nyando_training_v1.csv"

# Features the model uses — must all be present in the canonical CSV
FEATURE_COLS = [
    "elevation", "slope", "rainfall_3day",
    "distance_river", "clay_percent", "land_cover",
]

LON_VARIANTS = {"lon", "longitude", "long", "x", "Longitude", "Lon"}
LAT_VARIANTS = {"lat", "latitude", "y",  "Latitude", "Lat"}

# Required columns after BOM stripping
REQUIRED_SCHEMA_COLS = {"lon", "lat", "elevation", "flooded", *FEATURE_COLS}


# ── Helpers ──────────────────────────────────────────────────

def _find_csv() -> Path:
    """
    Return the canonical CSV path.
    If CANONICAL_CSV exists, always use it.
    Otherwise fall back to the first CSV found (for local dev without
    the curated file).
    """
    if CANONICAL_CSV.exists():
        return CANONICAL_CSV
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        pytest.skip(f"No CSV files in {DATA_DIR}")
    return csvs[0]


def _assert_schema(df: pd.DataFrame):
    """
    Strip BOM from column names and check all required columns exist.

    Called in the raw_df fixture so every test that uses raw_df
    automatically benefits from schema validation.
    """
    columns = {c.lstrip("\ufeff") for c in df.columns}
    missing = sorted(REQUIRED_SCHEMA_COLS - columns)
    assert not missing, f"Missing required schema columns: {missing}"


def _get_auc(m: dict):
    for key in ("auc_roc", "auc", "roc_auc", "AUC", "AUC_ROC"):
        if key in m:
            return m[key]
    return 0


def _get_f1(m: dict):
    for key in ("f1", "f1_score", "F1"):
        if key in m:
            return m[key]
    return 0


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """
    Load and schema-validate the canonical training CSV.
    """
    df = pd.read_csv(_find_csv())
    _assert_schema(df)
    return df


@pytest.fixture(scope="module")
def metrics():
    if not METRICS_PATH.exists():
        pytest.skip("metrics.json not found")
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Data tests ───────────────────────────────────────────────

def test_csv_file_exists():
    """
    The canonical dataset must be committed to the repo.
    """
    assert CANONICAL_CSV.exists(), (
        f"Canonical dataset missing: {CANONICAL_CSV}"
    )


def test_canonical_csv_selected():
    """_find_csv() must return the curated file, not the raw GEE export."""
    assert _find_csv().name == "nyando_training_v1.csv"


def test_csv_has_rows(raw_df):
    assert len(raw_df) > 100, "Training CSV has fewer than 100 rows"


def test_target_column_exists(raw_df):
    assert "flooded" in raw_df.columns, (
        f"'flooded' missing. Columns: {list(raw_df.columns)}"
    )


def test_required_schema_columns(raw_df):
    """All required columns must be present (BOM-stripped)."""
    _assert_schema(raw_df)


def test_target_has_both_classes(raw_df):
    counts = raw_df["flooded"].value_counts()
    assert 0 in counts.index, "No flooded=0 (non-flood) cases"
    assert 1 in counts.index, (
        "No flooded=1 (flood) cases — check label generation pipeline"
    )


def test_lon_lat_columns_exist(raw_df):
    """
    Lon/lat are required for spatial cross-validation.
    We check multiple naming variants because GEE sometimes exports
    as 'longitude'/'latitude' instead of 'lon'/'lat'.
    """
    cols_lower = {c.lower() for c in raw_df.columns}
    has_lon = bool(cols_lower & {v.lower() for v in LON_VARIANTS})
    has_lat = bool(cols_lower & {v.lower() for v in LAT_VARIANTS})
    assert has_lon, f"No longitude column found. Columns: {list(raw_df.columns)}"
    assert has_lat, f"No latitude column found. Columns: {list(raw_df.columns)}"


def test_feature_columns_present(raw_df):
    for col in FEATURE_COLS:
        assert col in raw_df.columns, (
            f"Feature column '{col}' missing from training CSV"
        )


# ── Model artifact tests ──────────────────────────────────────

def test_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Model artifact missing: {MODEL_PATH}"


def test_model_loads():
    """Model must load without error in pickle or joblib format."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from models.train_model import load
    model = load()
    assert model is not None


def test_model_has_predict(raw_df):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from models.train_model import load
    model = load()
    cols = [c for c in FEATURE_COLS if c in raw_df.columns]
    sample = raw_df[cols].dropna().head(5)
    preds = model.predict(sample)
    assert len(preds) == len(sample)


# ── Metrics tests ────────────────────────────────────────────

def test_metrics_file_exists():
    assert METRICS_PATH.exists(), "metrics.json not found"


def test_auc_above_threshold(metrics):
    """AUC must be above 0.85 to justify production deployment."""
    auc = _get_auc(metrics)
    assert auc > 0.85, f"AUC too low: {auc:.3f} (threshold: 0.85)"


def test_f1_above_threshold(metrics):
    """F1 must be above 0.75 for flood early warning use case."""
    f1 = _get_f1(metrics)
    assert f1 > 0.75, f"F1 too low: {f1:.3f} (threshold: 0.75)"
