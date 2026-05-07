"""
Nyando Basin Flood Risk Prediction API
FastAPI backend — deploy on Render.com (free tier)
Author: Bishop James Koero | github.com/jameskoero
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nyando Flood Risk API",
    description=(
        "Ward-level flood susceptibility prediction for Nyando River Basin, Kenya. "
        "XGBoost model trained on CHIRPS rainfall, NASA DEM, and Sentinel-1 SAR data. "
        "100% open data — Kenya DPA 2019 & GDPR compliant."
    ),
    version="1.0.0",
    contact={"name": "Bishop James Koero", "url": "https://github.com/jameskoero"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/nyando_xgb_v1.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"[WARN] Model file not found at {MODEL_PATH} — /predict will return 503")

FEATURES = ["elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover"]

# ── Risk classifier ───────────────────────────────────────────────────────────
def classify_risk(score: float) -> dict:
    if score < 0.35:
        return {"risk_class": "LOW",      "risk_label": "Minimal flood risk"}
    elif score < 0.60:
        return {"risk_class": "MEDIUM",   "risk_label": "Monitor closely — conditions changing"}
    elif score < 0.80:
        return {"risk_class": "HIGH",     "risk_label": "Prepare evacuation routes now"}
    else:
        return {"risk_class": "CRITICAL", "risk_label": "Immediate action required"}

# ── Request / Response models ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    elevation:      float = Field(..., example=1142.5, description="Terrain elevation (m)")
    slope:          float = Field(..., example=2.3,    description="Slope angle (degrees)")
    rainfall_3day:  float = Field(..., example=87.4,   description="3-day accumulated rainfall (mm)")
    distance_river: float = Field(..., example=320.0,  description="Distance to nearest river (m)")
    clay_percent:   float = Field(..., example=42.1,   description="Soil clay fraction 0–5cm (%)")
    land_cover:     int   = Field(..., example=40,     description="ESA WorldCover land cover class")
    ward:           Optional[str] = Field(None, example="Nyando Central")

class ShapContribution(BaseModel):
    feature:      str
    contribution: float

class PredictResponse(BaseModel):
    risk_score:    float
    risk_class:    str
    risk_label:    str
    shap_top3:     list[ShapContribution]
    ward:          Optional[str]
    model_version: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "model": "nyando_xgb_v1" if model else "not_loaded",
        "version": "1.0.0",
        "features": FEATURES,
    }

@app.get("/", tags=["System"])
def root():
    return {
        "message": "Nyando Flood Risk API v1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "author": "Bishop James Koero — github.com/jameskoero",
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH env variable.")

    X = pd.DataFrame([{f: getattr(req, f) for f in FEATURES}])

    # Risk score
    risk_score = float(model.predict_proba(X)[0][1])
    risk_info  = classify_risk(risk_score)

    # SHAP explanation
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)[0]
        shap_pairs  = sorted(
            zip(FEATURES, shap_values),
            key=lambda kv: abs(kv[1]),
            reverse=True
        )
        shap_top3 = [
            ShapContribution(feature=f, contribution=round(float(v), 4))
            for f, v in shap_pairs[:3]
        ]
    except Exception:
        shap_top3 = []

    return PredictResponse(
        risk_score=round(risk_score, 4),
        risk_class=risk_info["risk_class"],
        risk_label=risk_info["risk_label"],
        shap_top3=shap_top3,
        ward=req.ward,
        model_version="1.0.0",
    )

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(rows: list[PredictRequest]):
    """Predict risk scores for multiple wards in one call."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = []
    for req in rows:
        X = pd.DataFrame([{f: getattr(req, f) for f in FEATURES}])
        score = float(model.predict_proba(X)[0][1])
        info  = classify_risk(score)
        results.append({
            "ward":       req.ward,
            "risk_score": round(score, 4),
            "risk_class": info["risk_class"],
            "risk_label": info["risk_label"],
        })
    return {"predictions": results, "count": len(results)}
