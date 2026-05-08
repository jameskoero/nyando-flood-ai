"""test_preprocess.py — pytest tests for src/data/preprocess.py"""
import pandas as pd
import pytest
from src.data.preprocess import clean_features, apply_smote, FEATURES, TARGET


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "elevation":      [1142.5, -1.0,  1138.2, 1145.0,  0.0],
        "slope":          [2.3,     2.3,   1.8,    3.1,    1.5],
        "rainfall_3day":  [87.4,   91.2,  -5.0,   74.6,   60.0],
        "distance_river": [320.0, 180.0,  850.0,   95.0,  400.0],
        "clay_percent":   [42.1,   45.3,   38.7,   48.2,   39.0],
        "land_cover":     [40,     40,     50,     40,     30],
        "flooded":        [1,       0,      0,      1,      0],
    })


def test_clean_removes_negative_elevation(sample_df):
    assert (clean_features(sample_df)["elevation"] > 0).all()

def test_clean_removes_zero_elevation(sample_df):
    assert 0.0 not in clean_features(sample_df)["elevation"].values

def test_clean_removes_negative_rainfall(sample_df):
    assert (clean_features(sample_df)["rainfall_3day"] >= 0).all()

def test_clean_returns_dataframe(sample_df):
    assert isinstance(clean_features(sample_df), pd.DataFrame)

def test_clean_resets_index(sample_df):
    result = clean_features(sample_df)
    assert list(result.index) == list(range(len(result)))

def test_clean_keeps_valid_rows(sample_df):
    assert len(clean_features(sample_df)) > 0

def test_smote_returns_equal_classes(sample_df):
    clean = clean_features(sample_df)
    X_res, y_res = apply_smote(clean[FEATURES], clean[TARGET])
    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1]

def test_smote_preserves_feature_columns(sample_df):
    clean = clean_features(sample_df)
    X_res, _ = apply_smote(clean[FEATURES], clean[TARGET])
    assert list(X_res.columns) == FEATURES
