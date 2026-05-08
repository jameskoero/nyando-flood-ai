"""test_features.py — pytest tests for src/features/build_features.py"""
import pandas as pd
import pytest
from src.features.build_features import (
    add_rainfall_categories, add_flood_plain_index,
    add_soil_permeability, build_all_features)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "elevation":      [1142.5, 1138.2, 1145.0],
        "slope":          [2.3,    1.8,    3.1],
        "rainfall_3day":  [10.0,   55.0,   130.0],
        "distance_river": [320.0,  850.0,  95.0],
        "clay_percent":   [20.0,   35.0,   50.0],
        "land_cover":     [40,     50,     40],
    })


def test_rainfall_cat_added(sample_df):
    assert "rainfall_cat" in add_rainfall_categories(sample_df).columns

def test_rainfall_cat_range(sample_df):
    assert add_rainfall_categories(sample_df)["rainfall_cat"].between(0,4).all()

def test_fpi_added(sample_df):
    assert "flood_plain_index" in add_flood_plain_index(sample_df).columns

def test_fpi_positive(sample_df):
    assert (add_flood_plain_index(sample_df)["flood_plain_index"] > 0).all()

def test_soil_perm_added(sample_df):
    assert "soil_permeability" in add_soil_permeability(sample_df).columns

def test_soil_perm_range(sample_df):
    assert add_soil_permeability(sample_df)["soil_permeability"].isin([0,1,2]).all()

def test_build_all_adds_columns(sample_df):
    assert len(build_all_features(sample_df).columns) > len(sample_df.columns)

def test_build_all_no_nulls(sample_df):
    assert build_all_features(sample_df).isnull().sum().sum() == 0
