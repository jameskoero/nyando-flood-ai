import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Nyando Flood Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model — check multiple paths
_dir = os.path.dirname(os.path.abspath(__file__))
PATHS = [
    os.path.join(_dir, "models", "nyando_xgb_v1.pkl"),
    os.path.join(_dir, "..", "models", "nyando_xgb_v1.pkl"),
    "/app/backend/models/nyando_xgb_v1.pkl",
    "/app/models/nyando_xgb_v1.pkl",
]

model = None
MODEL_PATH = None
for p in PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

print(f"Model path resolved: {MODEL_PATH}")

if MODEL_PATH:
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Load failed: {e}")
else:
    print(f"⚠️ Model not found. Searched: {PATHS}")

class FloodInput(BaseModel):
    elevation: float
    slope: float
    rainfall_3day: float
    distance_river: float
    clay_percent: float
    land_cover: float
    ward: str = "Unknown"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "nyando_xgb_v1",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(data: FloodInput):
    if model is None:
        return {"error": "Model not loaded", "model_loaded": False}
    X = [[data.elevation, data.slope, data.rainfall_3day,
          data.distance_river, data.clay_percent, data.land_cover]]
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    if prob < 0.35:   risk = "LOW"
    elif prob < 0.60: risk = "MEDIUM"
    elif prob < 0.80: risk = "HIGH"
    else:             risk = "CRITICAL"
    return {
        "flood_probability": round(prob, 4),
        "risk_score": round(prob, 4),
        "risk_class": risk,
        "prediction": pred,
        "ward": data.ward,
        "model_version": "1.0.0"
    }
