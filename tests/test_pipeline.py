"""Nyando Flood AI — pytest suite"""
import json, joblib, numpy as np, pandas as pd, pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURES = [
    "elevation", "slope", "rainfall_3day",
    "distance_river", "clay_percent", "land_cover"
]


@pytest.fixture(scope="module")
def data():
    path = ROOT / "data" / "training" / "nyando_training_v1.csv"
    assert path.exists(), f"CSV not found: {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def model():
    path = ROOT / "models" / "nyando_xgb_v1.pkl"
    assert path.exists(), f"Model not found: {path}"
    return joblib.load(path)


@pytest.fixture(scope="module")
def metrics():
    path = ROOT / "metrics.json"
    assert path.exists(), f"metrics.json not found: {path}"
    with open(path) as f:
        return json.load(f)


# ── Data tests ───────────────────────────────────────────────────
def test_data_shape(data):
    assert data.shape[0] >= 100

def test_data_columns(data):
    for col in FEATURES + ["flooded"]:
        assert col in data.columns, f"Missing column: {col}"

def test_target_binary(data):
    assert set(data.flooded.unique()).issubset({0, 1})

def test_no_nulls(data):
    assert data.isnull().sum().sum() == 0

def test_flood_rate(data):
    assert 0.01 <= data.flooded.mean() <= 0.99


# ── Model tests ──────────────────────────────────────────────────
def test_model_loads(model):
    assert hasattr(model, "predict_proba")

def test_prediction_is_probability(model):
    x = np.array([[1200.0, 3.0, 60.0, 500.0, 35.0, 40]])
    p = float(model.predict_proba(x)[0, 1])
    assert 0.0 <= p <= 1.0

def test_batch_prediction(model, data):
    probs = model.predict_proba(data[FEATURES].values)
    assert probs.shape == (len(data), 2)
    assert (probs >= 0).all() and (probs <= 1).all()


# ── Metrics tests ────────────────────────────────────────────────
def test_auc_above_threshold(metrics):
    auc = metrics.get("auc_roc", metrics.get("auc", 0))
    assert float(auc) >= 0.85, f"AUC too low: {auc}"

def test_f1_above_threshold(metrics):
    f1 = metrics.get("f1_score", metrics.get("f1", 0))
    assert float(f1) >= 0.75, f"F1 too low: {f1}"
