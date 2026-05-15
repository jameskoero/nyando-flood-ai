"""
Nyando Flood AI — FastAPI Backend v1.1
Real GEE feature ranges used for input validation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib, json, numpy as np, time
from pathlib import Path

ROOT=Path(__file__).parent.parent
app=FastAPI(title="Nyando Flood AI API",version="1.1.0",
    description="Ward-level flood risk — Nyando Basin, Kenya (Real GEE data)")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
model=None; t0=time.time()

@app.on_event("startup")
async def load():
    global model
    try: model=joblib.load(ROOT/"models/nyando_xgb_v1.pkl"); print("✅ Model loaded")
    except Exception as e: print(f"⚠️ {e}")

FEATURES=["elevation","slope","rainfall_3day","distance_river","clay_percent","land_cover"]
LABELS={"LOW":"Minimal flood risk","MEDIUM":"Monitor closely",
        "HIGH":"Prepare evacuation routes","CRITICAL":"Immediate action required"}

def classify(s):
    return "LOW" if s<.35 else "MEDIUM" if s<.60 else "HIGH" if s<.80 else "CRITICAL"

class Req(BaseModel):
    # Input ranges validated against real GEE data from Nyando Basin
    elevation:     float=Field(...,ge=800,le=3000,description="NASA NASADEM (m)")
    slope:         float=Field(...,ge=0,le=45,description="Terrain slope (degrees)")
    rainfall_3day: float=Field(...,ge=0,le=250,description="CHIRPS 3-day sum (mm)")
    distance_river:float=Field(...,ge=0,le=6000,description="HydroSHEDS distance (m)")
    clay_percent:  float=Field(...,ge=0,le=100,description="ISRIC SoilGrids clay %")
    land_cover:    int=Field(...,ge=0,le=100,description="ESA WorldCover class")
    ward:          Optional[str]=Field("Unknown")

def _pred(r):
    X=np.array([[r.elevation,r.slope,r.rainfall_3day,r.distance_river,r.clay_percent,r.land_cover]])
    s=float(model.predict_proba(X)[0,1]); c=classify(s)
    top3=sorted(zip(FEATURES,model.feature_importances_),key=lambda x:-x[1])[:3]
    return {"risk_score":round(s,4),"risk_class":c,"risk_label":LABELS[c],
            "shap_top3":[{"feature":f,"importance":round(float(i),4)} for f,i in top3],
            "ward":r.ward,"model_version":"1.1.0"}

@app.get("/") 
async def root(): return {"api":"Nyando Flood AI v1.1","docs":"/docs","data_source":"Real GEE satellite data"}
@app.get("/health")
async def health(): return {"status":"ok","model":"nyando_xgb_v1","version":"1.1.0","model_loaded":model is not None,"uptime_s":round(time.time()-t0)}
@app.post("/predict")
async def predict(r:Req):
    if not model: raise HTTPException(503,"Model not loaded")
    return _pred(r)
@app.post("/predict/batch")
async def batch(reqs:List[Req]):
    if not model: raise HTTPException(503,"Model not loaded")
    if len(reqs)>100: raise HTTPException(400,"Max 100 per batch")
    return {"predictions":[_pred(r) for r in reqs],"count":len(reqs)}
@app.get("/metrics")
async def get_metrics():
    try:
        with open(ROOT/"metrics.json") as f: return json.load(f)
    except: raise HTTPException(404,"metrics.json not found")
