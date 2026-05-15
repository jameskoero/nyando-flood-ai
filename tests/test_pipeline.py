"""Nyando Flood AI — pytest suite (15 tests, real GEE data)"""
import json,joblib,numpy as np,pandas as pd,pytest
from pathlib import Path
ROOT=Path(__file__).parent.parent
FEATURES=["elevation","slope","rainfall_3day","distance_river","clay_percent","land_cover"]

@pytest.fixture(scope="module")
def data(): return pd.read_csv(ROOT/"data/training/nyando_training_v1.csv")
@pytest.fixture(scope="module")
def raw(): return pd.read_csv(ROOT/"data/training/nyando_training_v1_raw_gee.csv")
@pytest.fixture(scope="module")
def model(): return joblib.load(ROOT/"models/nyando_xgb_v1.pkl")
@pytest.fixture(scope="module")
def metrics():
    with open(ROOT/"metrics.json") as f: return json.load(f)

# Real GEE data integrity tests
def test_real_gee_rows(raw):        assert raw.shape[0]==2308,"Expected 2308 real GEE observations"
def test_real_coordinates(raw):     assert raw.lon.between(34.0,36.0).all() and raw.lat.between(-1.0,1.0).all()
def test_real_sar_labels(raw):      assert raw.flooded.sum()==2,"Expected exactly 2 SAR-confirmed flood pixels"
def test_real_elevation(raw):       assert 1100<=raw.elevation.min() and raw.elevation.max()<=2600
def test_real_rainfall(raw):        assert 80<=raw.rainfall_3day.min() and raw.rainfall_3day.max()<=165
def test_no_nulls(raw):             assert raw.isnull().sum().sum()==0

# Training data tests
def test_training_shape(data):      assert data.shape[0]==2308 and data.shape[1]>=10
def test_training_columns(data):    assert all(c in data.columns for c in FEATURES+["flooded","lon","lat"])
def test_flood_rate(data):          assert 0.15<=data.flooded.mean()<=0.30,"Expect ~22% calibrated flood rate"
def test_has_susceptibility(data):  assert "susceptibility" in data.columns

# Model tests
def test_model_loads(model):        assert hasattr(model,"predict_proba")
def test_sar_point_high(model):
    # Both real SAR flood points should score HIGH
    p=model.predict_proba([[1137,.7,112.0,847,42,30]])[0,1]; assert p>.5,f"SAR flood point scored {p:.3f}"
def test_highland_low(model):
    p=model.predict_proba([[2169,13.9,115,1904,40,10]])[0,1]; assert p<.5,f"Highland scored {p:.3f}"
def test_auc(metrics):              assert metrics["auc_roc"]>=0.85
def test_real_data_documented(metrics): assert "real_data_note" in metrics,"Provenance missing"
