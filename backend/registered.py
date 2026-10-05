"""The registered Phase C model served through ONNX Runtime (docs/PHASE_C_PROTOCOL.md Section 17): a hash-verified session, scoring and stored metrics."""
import hashlib
import importlib
import json
import os

from backend.integrity import read_verified

LAND_COVER_CLASSES = (10, 20, 30, 40, 50, 60, 80, 90)
FEATURES = ("elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover")
OUTPUT = "probabilities"


class Registered:
    def __init__(self):
        self.session, self.reason, self.file, self.sha256, self.entry, self.meta = None, "not loaded", None, None, {}, {}

    def describe(self):
        if self.session is None:
            return {"loaded": False, "reason": self.reason}
        return {"loaded": True, "file": self.file, "sha256": self.sha256, "trained_on": self.entry.get("trained_on"), "arm": self.meta.get("model"),
                "protocol": self.meta.get("protocol"), "registration_run": self.meta.get("registration_run")}

    def score(self, values):
        import numpy as np
        feed = {n: np.array([[np.nan if values.get(n) is None else values[n]]], dtype=np.int64 if n == "land_cover" else np.float64) for n in FEATURES}
        return float(self.session.run([OUTPUT], feed)[0][0][1])


def load_registered(root, ort_loader=None):
    """Verify the active manifest entry's file, then build the session from the verified bytes. Never raises: a failure leaves session None and says why."""
    r = Registered()
    try:
        ort = ort_loader() if ort_loader else importlib.import_module("onnxruntime")
    except Exception as e:
        r.reason = "onnxruntime is not available (%s)" % type(e).__name__
        return r
    try:
        with open(os.path.join(root, "models", "MANIFEST.json"), encoding="utf-8") as f:
            active = [(k, v) for k, v in json.load(f).items() if v.get("status") == "active"]
        if len(active) != 1:
            raise ValueError("expected exactly one active model in models/MANIFEST.json, found %d" % len(active))
        rel, entry = active[0]
        blob = read_verified(os.path.join(root, rel), root)
        session = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
        meta = dict(session.get_modelmeta().custom_metadata_map)
        inputs = {i.name: i.type for i in session.get_inputs()}
        want = {n: ("tensor(int64)" if n == "land_cover" else "tensor(double)") for n in FEATURES}
        if inputs != want:
            raise ValueError("the model inputs %s differ from the API contract %s" % (inputs, want))
        if meta.get("training_data_sha256") != entry.get("trained_on"):
            raise ValueError("the model's embedded training-data hash differs from models/MANIFEST.json")
        r.session, r.file, r.entry, r.meta, r.sha256, r.reason = session, os.path.basename(rel), entry, meta, hashlib.sha256(blob).hexdigest(), "loaded"
    except Exception as e:
        r.reason = "%s: %s" % (type(e).__name__, str(e)[:200])
    return r


def metrics_payload(root):
    """Stored evaluation of the registered model from docs/PHASE_C_RESULTS.json (read from MLflow when the closure record was built); None if the file is absent."""
    path = os.path.join(root, "docs", "PHASE_C_RESULTS.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    reg, rob = res["runs"]["registration"], res["runs"]["robustness"]
    m, r = reg["metrics"], rob["metrics"]

    def stat(src, p):
        return {k: src["%s/%s" % (p, k)] for k in ("mean_diff", "ci_low", "ci_high")}

    t = "temporal/con_minus_elevation/ci_low"
    return {"registered_model": res["registered_model"]["file"], "model_sha256": res["registered_model"]["sha256"], "arm": res["registered_model"]["arm"],
            "evaluation": "leave-one-event-out with shared locations removed; paired per-event bootstrap, 97.5% intervals",
            "per_event_mean_auc": {"mappable": m["mappable/con/per_event_mean"], "full": m["full/con/per_event_mean"]},
            "versus_elevation_only": {"mappable": stat(m, "mappable/con/vs_elevation"), "full": stat(m, "full/con/vs_elevation")},
            "beyond_elevation_only": {"selection_split": m["mappable/con/vs_elevation/ci_low"] > 0, "buffered_1000m": r["buffer/con_minus_elevation/ci_low"] > 0,
                                      "out_of_time": (r[t] > 0) if t in r else None},
            "training_data_sha256": reg["training_data_sha256"], "runs": {"registration": reg["id"], "robustness": rob["id"]},
            "source": "docs/PHASE_C_RESULTS.json", "record": "docs/PHASE_C_CLOSURE.md"}
