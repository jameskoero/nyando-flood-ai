import os
import json
import time
import hashlib
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Nyando Flood Risk API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_dir = os.path.dirname(os.path.abspath(__file__))
PATHS = [
    os.path.join(_dir, "models", "nyando_xgb_v1.pkl"),
    os.path.join(_dir, "..", "models", "nyando_xgb_v1.pkl"),
    "/app/backend/models/nyando_xgb_v1.pkl",
    "/app/models/nyando_xgb_v1.pkl",
]

model = None
MODEL_PATH = None
MODEL_SHA256 = None
PROVENANCE = {}

for p in PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

print(f"Model path resolved: {MODEL_PATH}")

if MODEL_PATH:
    try:
        model = joblib.load(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()
        print(f"Model loaded from {MODEL_PATH}")
        print(f"SHA-256: {MODEL_SHA256}")
    except Exception as e:
        print(f"Load failed: {e}")
else:
    print(f"Model not found. Searched: {PATHS}")

_provenance_path = os.path.join(os.path.dirname(MODEL_PATH), "PROVENANCE.json") if MODEL_PATH else None
if _provenance_path and os.path.exists(_provenance_path):
    with open(_provenance_path) as f:
        PROVENANCE = json.load(f)
    if PROVENANCE.get("model_sha256") != MODEL_SHA256:
        print("PROVENANCE.json's recorded model hash does not match the loaded model file.")
else:
    print("No PROVENANCE.json found next to the model.")

_metrics_cache = {"data": None, "loaded_at": 0}
_METRICS_TTL = 900

def _load_metrics():
    now = time.time()
    if _metrics_cache["data"] is not None and (now - _metrics_cache["loaded_at"]) < _METRICS_TTL:
        return _metrics_cache["data"]
    metrics_path = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json") if MODEL_PATH else None
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
    else:
        data = {"error": "metrics.json not found"}
    _metrics_cache["data"] = data
    _metrics_cache["loaded_at"] = now
    return data

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
        "model_loaded": model is not None,
        "model_file": os.path.basename(MODEL_PATH) if MODEL_PATH else None,
        "model_sha256": MODEL_SHA256,
        "training_data_file": PROVENANCE.get("training_data_file"),
        "training_data_sha256": PROVENANCE.get("training_data_sha256"),
        "version": "1.0.0",
    }

@app.get("/metrics")
def metrics():
    data = _load_metrics()
    data["model_sha256"] = MODEL_SHA256
    data["cache_ttl_seconds"] = _METRICS_TTL
    return data

@app.post("/predict")
@limiter.limit("10/minute")
def predict(request: Request, data: FloodInput):
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
