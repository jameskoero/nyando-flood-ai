"""test_model.py — pytest tests for models and geo_utils."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from src.utils.geo_utils import classify_risk


@pytest.fixture
def tiny_model():
    X = pd.DataFrame(np.random.rand(200, 6), columns=[
        "elevation","slope","rainfall_3day",
        "distance_river","clay_percent","land_cover"])
    y = pd.Series(np.random.randint(0, 2, 200))
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X


def test_predict_proba_range(tiny_model):
    model, X = tiny_model
    probs = model.predict_proba(X)[:,1]
    assert probs.min() >= 0.0 and probs.max() <= 1.0

def test_predict_returns_binary(tiny_model):
    model, X = tiny_model
    assert set(model.predict(X)).issubset({0, 1})

def test_predict_shape(tiny_model):
    model, X = tiny_model
    assert len(model.predict(X)) == len(X)

def test_risk_low():        assert classify_risk(0.20)["risk_class"] == "LOW"
def test_risk_medium():     assert classify_risk(0.50)["risk_class"] == "MEDIUM"
def test_risk_high():       assert classify_risk(0.70)["risk_class"] == "HIGH"
def test_risk_critical():   assert classify_risk(0.90)["risk_class"] == "CRITICAL"

def test_risk_boundary_medium():
    assert classify_risk(0.35)["risk_class"] == "MEDIUM"

def test_risk_boundary_critical():
    assert classify_risk(0.80)["risk_class"] == "CRITICAL"

def test_risk_returns_label():
    result = classify_risk(0.65)
    assert "risk_label" in result and isinstance(result["risk_label"], str)

def test_risk_returns_score():
    assert classify_risk(0.72)["score"] == round(0.72, 4)
