from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib, numpy as np, time

app = FastAPI(title="Nyando Flood API", version="1.2.0")

FEATURES = ["elevation","slope","rainfall_3day",
            "distance_river","clay_percent","land_cover"]

MODEL_PATH = Path(__file__).parent / "models" / "nyando_xgb_v1.pkl"
model = None
start_time = time.time()

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR loading model: {e}")

class Req(BaseModel):
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
        "uptime_s": round(time.time() - start_time)
    }

@app.post("/predict")
def predict(r: Req):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    X = np.array([[getattr(r, f) for f in FEATURES]])
    score = float(model.predict_proba(X)[0, 1])
    if score < 0.35:   cls = "LOW"
    elif score < 0.60: cls = "MEDIUM"
    elif score < 0.80: cls = "HIGH"
    else:              cls = "CRITICAL"
    imp = model.feature_importances_
    top3 = sorted(zip(FEATURES, imp), key=lambda x: -x[1])[:3]
    return {
        "risk_score": round(score, 4),
        "risk_class": cls,
        "ward": r.ward,
        "shap_top3": [{"feature": f, "contribution": round(v, 4)}
                      for f, v in top3],
        "model_version": "1.2.0",
        "model_type": "GradientBoostingClassifier"
    }
