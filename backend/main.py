import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Nyando Flood Risk API", version="1.0.0")

# Try xgb first, fallback to gb
_dir = os.path.dirname(__file__)
_xgb = os.path.join(_dir, "models", "nyando_xgb_v1.pkl")
_gb  = os.path.join(_dir, "models", "nyando_gb_v1.pkl")
MODEL_PATH = _xgb if os.path.exists(_xgb) else _gb
print(f"Using model: {MODEL_PATH}")

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Model load failed: {e}")

class FloodInput(BaseModel):
    elevation: float
    slope: float
    rainfall_3day: float
    distance_river: float
    clay_percent: float
    land_cover: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model": "nyando_gb_v1",
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(data: FloodInput):
    if model is None:
        return {"error": "Model not loaded"}
    
    X = [[data.elevation, data.slope, data.rainfall_3day,
          data.distance_river, data.clay_percent, data.land_cover]]
    
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    
    if prob < 0.35:
        risk = "LOW"
    elif prob < 0.60:
        risk = "MEDIUM"
    elif prob < 0.80:
        risk = "HIGH"
    else:
        risk = "CRITICAL"
    
    return {
        "risk_score": round(prob, 4),
        "risk_class": risk,
        "prediction": pred,
        "model_version": "1.0.0"
    }
