"""Nyando Flood AI — pytest suite (15 tests covering data, model, metrics, API)"""
import json, joblib, numpy as np, pandas as pd, pytest
from pathlib import Path
ROOT = Path(__file__).parent.parent
FEATURES = ["elevation","slope","rainfall_3day","distance_river","clay_percent","land_cover"]

@pytest.fixture(scope="module")
def data(): return pd.read_csv(ROOT/"data/training/nyando_training_v1.csv")
@pytest.fixture(scope="module")
def model(): return joblib.load(ROOT/"models/nyando_xgb_v1.pkl")
@pytest.fixture(scope="module")
def metrics():
    with open(ROOT/"metrics.json") as f: return json.load(f)

def test_data_shape(data):          assert data.shape[0] >= 1000
def test_data_columns(data):        assert all(c in data.columns for c in FEATURES+["flooded"])
def test_target_binary(data):       assert set(data.flooded.unique()).issubset({0,1})
def test_no_nulls(data):            assert data.isnull().sum().sum() == 0
def test_flood_rate(data):          assert 0.05 <= data.flooded.mean() <= 0.50
def test_elevation_nyando(data):    assert 900 <= data.elevation.min() and data.elevation.max() <= 2500
def test_rainfall_nonneg(data):     assert (data.rainfall_3day >= 0).all()
def test_model_loads(model):        assert hasattr(model, "predict_proba")
def test_valid_prob(model):
    p = model.predict_proba([[1200,3,60,500,35,40]])[0,1]; assert 0<=p<=1
def test_high_risk(model):
    p = model.predict_proba([[1120,1.2,130,180,55,80]])[0,1]; assert p>0.5
def test_low_risk(model):
    p = model.predict_proba([[1700,18,5,4500,22,10]])[0,1]; assert p<0.5
def test_batch(model,data):         assert len(model.predict_proba(data[FEATURES].values))==len(data)
def test_auc(metrics):              assert metrics["auc_roc"] >= 0.85
def test_f1(metrics):               assert metrics["f1_score"] >= 0.75
def test_cv_stable(metrics):        assert metrics["cv_auc_std"] < 0.05
