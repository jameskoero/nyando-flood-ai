"""Nyando Flood AI FastAPI Backend v1.2"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib, numpy as np, time
from pathlib import Path

app = FastAPI(
    title="Nyando Flood AI API",
    version="1.2.0",
    description="Ward-level flood risk - Nyando Basin, Kenya (Real GEE data)"
)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
t0 = time.time()

@app.on_event("startup")
async def load_model():
    global model
    base = Path(__file__).parent
    candidates = [
        base / "models" / "nyando_xgb_v1.pkl",
        Path(__file__).parent / "models" / "nyando_xgb_v1.pkl"),
        base / "nyando_xgb_v1.pkl",
    ]
    for p in candidates:
        print(f"Trying: {p}  exists={p.exists()}")
        if p.exists():
            try:
                model = joblib.load(p)
                print(f"Model loaded from {p}")
                return
            except Exception as e:
                print(f"Failed: {e}")
    print("All paths failed")

FEATURES = ["elevation","slope","rainfall_3day",
            "distance_river","clay_percent","land_cover"]
LABELS = {
    "LOW":      "Minimal flood risk - normal activities OK",
    "MEDIUM":   "Monitor closely - avoid riverbanks",
    "HIGH":     "Prepare evacuation routes now",
    "CRITICAL": "EVACUATE immediately - alert authorities"
}

def classify(s):
    if s < 0.35: return "LOW"
    if s < 0.60: return "MEDIUM"
    if s < 0.80: return "HIGH"
    return "CRITICAL"

class Req(BaseModel):
    elevation:      float = Field(..., ge=800,  le=3000)
    slope:          float = Field(..., ge=0,    le=45)
    rainfall_3day:  float = Field(..., ge=0,    le=250)
    distance_river: float = Field(..., ge=0,    le=6000)
    clay_percent:   float = Field(..., ge=0,    le=100)
    land_cover:     int   = Field(..., ge=0,    le=100)
    ward:           Optional[str] = Field(default="Unknown")

def _predict(r):
    X = np.array([[r.elevation, r.slope, r.rainfall_3day,
                   r.distance_river, r.clay_percent, r.land_cover]])
    s = float(model.predict_proba(X)[0, 1])
    c = classify(s)
    top3 = sorted(zip(FEATURES, model.feature_importances_),
                  key=lambda x: -x[1])[:3]
    return {"risk_score": round(s,4), "risk_class": c,
            "risk_label": LABELS[c], "flood_risk": 1 if s>=0.5 else 0,
            "top3_features": [{"feature":f,"importance":round(float(i),4)}
                              for f,i in top3],
            "ward": r.ward, "model_version": "1.2.0"}

@app.get("/")
async def root():
    return {"api":"Nyando Flood AI v1.2","docs":"/docs",
            "data_source":"Real GEE satellite data","status":"live"}

@app.get("/health")
async def health():
    return {"status":"ok","model":"nyando_xgb_v1","version":"1.2.0",
            "model_loaded": model is not None,
            "uptime_s": round(time.time()-t0)}

@app.post("/predict")
async def predict(r: Req):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return _predict(r)

@app.post("/predict/batch")
async def batch(reqs: List[Req]):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if len(reqs) > 100:
        raise HTTPException(400, "Max 100 per batch")
    return {"predictions":[_predict(r) for r in reqs],"count":len(reqs)}

@app.get("/metrics")
async def get_metrics():
    return {"auc_roc":0.9717,"f1_score":0.9022,"cv_score":0.9727,
            "cv_std":0.004,"recall":0.922,"precision":0.889,
            "training_samples":2308,
            "model_type":"GradientBoostingClassifier","version":"1.2.0"}
