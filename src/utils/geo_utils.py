"""
geo_utils.py — Geospatial helpers for Nyando Flood AI.
"""
import math
from typing import List

NYANDO_BBOX = [34.7, -0.4, 35.3, 0.1]

RISK_THRESHOLDS = {
    "LOW":      (0.00, 0.35),
    "MEDIUM":   (0.35, 0.60),
    "HIGH":     (0.60, 0.80),
    "CRITICAL": (0.80, 1.01),
}

RISK_LABELS = {
    "LOW":      "Minimal flood risk",
    "MEDIUM":   "Monitor closely — conditions changing",
    "HIGH":     "Prepare evacuation routes now",
    "CRITICAL": "Immediate action required",
}


def classify_risk(score: float) -> dict:
    for cls, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return {"risk_class": cls,
                    "risk_label": RISK_LABELS[cls],
                    "score": round(score, 4)}
    return {"risk_class": "CRITICAL",
            "risk_label": RISK_LABELS["CRITICAL"],
            "score": round(score, 4)}


def bbox_to_ee_geometry(bbox: List[float] = None) -> str:
    b = bbox or NYANDO_BBOX
    return f"ee.Geometry.Rectangle([{b[0]}, {b[1]}, {b[2]}, {b[3]}])"


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
