import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "training"
MODEL_PATH = ROOT / "models" / "nyando_xgb_v1.pkl"
METRICS    = ROOT / "metrics.json"

FEATURE_COLS = [
    "elevation",
    "slope",
    "rainfall_3day",
    "distance_river",
    "clay_percent",
    "land_cover",
]


# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw_df():
    csv_files = list(DATA_DIR.glob("*.csv"))
    assert len(csv_files) > 0, f"No CSV files found in {DATA_DIR}"
    return pd.read_csv(csv_files[0])


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def metrics():
    assert METRICS.exists(), f"metrics.json not found at {METRICS}"
    with open(METRICS) as f:
        return json.load(f)


# ── Data tests ─────────────────────────────────────────────────────────────
def test_csv_loads(raw_df):
    """CSV loads without error and has rows."""
    assert raw_df.shape[0] > 0, "CSV is empty"


def test_csv_has_minimum_rows(raw_df):
    """Dataset must have at least 500 records to be meaningful."""
    assert raw_df.shape[0] >= 500, (
        f"Too few rows: {raw_df.shape[0]}. Expected >= 500."
    )


def test_target_column_exists(raw_df):
    """'flooded' target column must be present."""
    assert "flooded" in raw_df.columns, (
        f"'flooded' column missing. Columns: {list(raw_df.columns)}"
    )


def test_target_has_positive_cases(raw_df):
    """At least one flooded == 1 case must exist in the dataset."""
    assert raw_df["flooded"].sum() > 0, (
        "No positive flood cases found in dataset."
    )


def test_feature_columns_present(raw_df):
    """All expected feature columns must exist (allows lon/longitude variants)."""
    # Allow longitude/lat column name variants
    available = set(raw_df.columns)
    for col in FEATURE_COLS:
        assert col in available, (
            f"Expected feature column '{col}' not found. "
            f"Available columns: {sorted(available)}"
        )


def test_coordinate_columns_present(raw_df):
    """Longitude and latitude columns must exist (flexible naming)."""
    lon_variants = {"lon", "longitude", "long", "x"}
    lat_variants = {"lat", "latitude", "y"}
    assert lon_variants & set(raw_df.columns), (
        f"No longitude column found. Columns: {list(raw_df.columns)}"
    )
    assert lat_variants & set(raw_df.columns), (
        f"No latitude column found. Columns: {list(raw_df.columns)}"
    )


def test_no_all_nan_feature_columns(raw_df):
    """No feature column should be entirely NaN."""
    for col in FEATURE_COLS:
        if col in raw_df.columns:
            assert raw_df[col].notna().sum() > 0, (
                f"Column '{col}' is entirely NaN."
            )


# ── Model tests ────────────────────────────────────────────────────────────
def test_model_loads(model):
    """Model pickle loads without error."""
    assert model is not None


def test_model_predicts(model, raw_df):
    """Model returns predictions for a small sample of feature rows."""
    sample = raw_df[FEATURE_COLS].dropna().head(10)
    assert len(sample) > 0, "No complete rows available for prediction test."
    preds = model.predict(sample)
    assert len(preds) == len(sample)
    assert set(preds).issubset({0, 1}), (
        f"Unexpected prediction values: {set(preds)}"
    )


def test_model_predict_proba(model, raw_df):
    """Model returns valid probabilities between 0 and 1."""
    sample = raw_df[FEATURE_COLS].dropna().head(10)
    assert len(sample) > 0
    proba = model.predict_proba(sample)
    assert proba.shape == (len(sample), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ── Metrics tests ──────────────────────────────────────────────────────────
def test_metrics_loads(metrics):
    """metrics.json loads and is a non-empty dict."""
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_auc_roc_is_strong(metrics):
    """AUC-ROC must be present (flexible key) and >= 0.90."""
    auc = metrics.get("auc_roc", metrics.get("auc", metrics.get("roc_auc", None)))
    assert auc is not None, (
        f"No AUC key found in metrics.json. Keys present: {list(metrics.keys())}"
    )
    assert auc >= 0.90, f"AUC-ROC {auc:.4f} is below expected threshold of 0.90"


def test_f1_score_present(metrics):
    """F1 score must be present and reasonable (>= 0.80)."""
    f1 = metrics.get("f1", metrics.get("f1_score", None))
    if f1 is not None:
        assert f1 >= 0.80, f"F1 {f1:.4f} is below expected threshold of 0.80"
