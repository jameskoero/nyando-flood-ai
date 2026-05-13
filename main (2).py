"""
Nyando Flood AI — FastAPI Backend v1.0
Endpoints: GET /health  POST /predict  GET /metrics  GET /
Deploy: uvicorn main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, json, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
app = FastAPI(title="Nyando Flood AI API", version="1.0.0",
              description="72-hour ward-level flood risk prediction — Nyando Basin, Kenya")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
@app.on_event("startup")
async def load():
    global model
    try: model = joblib.load(ROOT/"models/nyando_xgb_v1.pkl"); print("✅ Model loaded")
    except Exception as e: print(f"⚠️ Model load failed: {e}")

FEATURES = ["elevation","slope","rainfall_3day","distance_river","clay_percent","land_cover"]

class PredictRequest(BaseModel):
    elevation:     float = Field(..., ge=800, le=2500)
    slope:         float = Field(..., ge=0, le=45)
    rainfall_3day: float = Field(..., ge=0, le=300)
    distance_river:float = Field(..., ge=0, le=10000)
    clay_percent:  float = Field(..., ge=0, le=100)
    land_cover:    int   = Field(..., ge=0, le=100)
    ward:          str   = Field("Unknown")

@app.get("/")
async def root(): return {"message":"Nyando Flood AI API v1.0.0","docs":"/docs"}

@app.get("/health")
async def health(): return {"status":"ok","model":"nyando_xgb_v1","version":"1.0.0","model_loaded":model is not None}

@app.post("/predict")
async def predict(req: PredictRequest):
    if not model: raise HTTPException(503, "Model not loaded")
    X = np.array([[req.elevation,req.slope,req.rainfall_3day,req.distance_river,req.clay_percent,req.land_cover]])
    score = float(model.predict_proba(X)[0,1])
    cls = "LOW" if score<0.35 else "MEDIUM" if score<0.60 else "HIGH" if score<0.80 else "CRITICAL"
    labels = {"LOW":"Minimal flood risk","MEDIUM":"Monitor closely","HIGH":"Prepare evacuation routes","CRITICAL":"Immediate action required"}
    top3 = sorted(zip(FEATURES,model.feature_importances_),key=lambda x:-x[1])[:3]
    return {"risk_score":round(score,4),"risk_class":cls,"risk_label":labels[cls],
            "shap_top3":[{"feature":f,"importance":round(i,4)} for f,i in top3],"ward":req.ward,"model_version":"1.0.0"}

@app.get("/metrics")
async def get_metrics():
    try:
        with open(ROOT/"metrics.json") as f: return json.load(f)
    except: raise HTTPException(404,"metrics.json not found")
