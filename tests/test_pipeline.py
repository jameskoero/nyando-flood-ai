"""
tests/test_pipeline.py
Nyando Flood AI — CI test suite  v2.0  (May 2026)

Design principles:
  - Zero hardcoded row counts or exact metric values that could drift
  - Flexible column-name resolution (lon/longitude, lat/latitude)
  - Flexible metrics.json key lookup (auc_roc / auc / roc_auc)
  - All paths resolved from repo root via __file__, never cwd-dependent
  - No imports from src/ — only stdlib + scientific stack
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Repo root and asset paths ──────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "training"
MODEL_PATH   = ROOT / "models" / "nyando_xgb_v1.pkl"
METRICS_PATH = ROOT / "metrics.json"

FEATURE_COLS = [
    "elevation",
    "slope",
    "rainfall_3day",
    "distance_river",
    "clay_percent",
    "land_cover",
]

LON_VARIANTS = {"lon", "longitude", "long", "x", "Longitude", "Lon"}
LAT_VARIANTS = {"lat", "latitude", "y",    "Latitude",  "Lat"}


# ── Helpers ────────────────────────────────────────────────────────────────
def _find_csv() -> Path:
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        pytest.skip(f"No CSV files in {DATA_DIR}")
    return csvs[0]


def _get_auc(m: dict):
    for key in ("auc_roc", "auc", "roc_auc", "AUC", "AUC_ROC"):
        if key in m:
            return float(m[key])
    return None


def _get_f1(m: dict):
    for key in ("f1", "f1_score", "F1", "f1score"):
        if key in m:
            return float(m[key])
    return None


# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(_find_csv())


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def metrics():
    if not METRICS_PATH.exists():
        pytest.skip(f"metrics.json not found: {METRICS_PATH}")
    with open(METRICS_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# DATA TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_csv_file_exists():
    csvs = list(DATA_DIR.glob("*.csv"))
    assert len(csvs) > 0, f"No CSV files in {DATA_DIR}"


def test_csv_has_rows(raw_df):
    assert raw_df.shape[0] > 0, "CSV is empty"


def test_csv_minimum_size(raw_df):
    assert raw_df.shape[0] >= 100, \
        f"Only {raw_df.shape[0]} rows — dataset too small"


def test_target_column_exists(raw_df):
    assert "flooded" in raw_df.columns, \
        f"'flooded' missing. Columns: {list(raw_df.columns)}"


def test_target_has_both_classes(raw_df):
    counts = raw_df["flooded"].value_counts()
    assert 0 in counts.index, "No flooded=0 (non-flood) cases"
    assert 1 in counts.index, "No flooded=1 (flood) cases"


def test_feature_columns_present(raw_df):
    missing = [c for c in FEATURE_COLS if c not in raw_df.columns]
    assert not missing, \
        f"Missing features: {missing}\nAvailable: {sorted(raw_df.columns)}"


def test_coordinate_columns_present(raw_df):
    cols = set(raw_df.columns)
    assert cols & LON_VARIANTS, \
        f"No longitude column. Expected one of {LON_VARIANTS}"
    assert cols & LAT_VARIANTS, \
        f"No latitude column. Expected one of {LAT_VARIANTS}"


def test_features_not_all_nan(raw_df):
    for col in FEATURE_COLS:
        if col in raw_df.columns:
            assert raw_df[col].notna().sum() > 0, \
                f"Column '{col}' is entirely NaN"


def test_no_total_duplicate_dataset(raw_df):
    dup_ratio = raw_df.duplicated().sum() / len(raw_df)
    assert dup_ratio < 0.95, \
        f"{dup_ratio:.1%} duplicate rows — data integrity issue"


# ══════════════════════════════════════════════════════════════════════════
# MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"


def test_model_loads(model):
    assert model is not None


def test_model_has_predict(model):
    assert hasattr(model, "predict"), "Model missing .predict()"


def test_model_has_predict_proba(model):
    assert hasattr(model, "predict_proba"), "Model missing .predict_proba()"


def test_model_predicts_binary(model, raw_df):
    sample = raw_df[FEATURE_COLS].dropna().head(20)
    assert len(sample) > 0, "No complete rows for prediction"
    preds = model.predict(sample)
    assert len(preds) == len(sample)
    assert set(preds).issubset({0, 1}), \
        f"Unexpected prediction values: {set(preds)}"


def test_model_predict_proba_valid(model, raw_df):
    sample = raw_df[FEATURE_COLS].dropna().head(20)
    proba = model.predict_proba(sample)
    assert proba.shape == (len(sample), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), \
        "Row probabilities do not sum to 1"
    assert np.all(proba >= 0) and np.all(proba <= 1), \
        "Probabilities outside [0, 1]"


def test_model_not_constant_predictor(model, raw_df):
    sample = raw_df[FEATURE_COLS].dropna().head(50)
    if len(sample) < 5:
        pytest.skip("Too few rows to test for constant prediction")
    preds = model.predict(sample)
    assert len(set(preds)) > 1, \
        "Model predicts same class for all samples — degenerate model"


# ══════════════════════════════════════════════════════════════════════════
# METRICS TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_metrics_file_exists():
    assert METRICS_PATH.exists(), f"metrics.json not found: {METRICS_PATH}"


def test_metrics_is_valid_json(metrics):
    assert isinstance(metrics, dict) and len(metrics) > 0, \
        "metrics.json is empty or not a dict"


def test_auc_key_present(metrics):
    auc = _get_auc(metrics)
    assert auc is not None, \
        f"No AUC key found. Keys: {list(metrics.keys())}"


def test_auc_is_strong(metrics):
    auc = _get_auc(metrics)
    if auc is None:
        pytest.skip("No AUC key")
    assert auc >= 0.90, f"AUC {auc:.4f} below 0.90 benchmark"


def test_f1_is_reasonable(metrics):
    f1 = _get_f1(metrics)
    if f1 is None:
        pytest.skip("No F1 key in metrics.json")
    assert f1 >= 0.80, f"F1 {f1:.4f} below 0.80 benchmark"


def test_metrics_has_model_name(metrics):
    assert "model" in metrics, \
        f"'model' key missing. Keys: {list(metrics.keys())}"
