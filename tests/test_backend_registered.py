"""Serving the registered model (docs/PHASE_C_PROTOCOL.md Section 17; register rows D9 and D33): the endpoints against the committed artifact and the stored results."""
import csv
import json
import os
from pathlib import Path

import pytest

if os.environ.get("GITHUB_ACTIONS") == "true":
    import onnxruntime as ort
else:
    try:
        import onnxruntime as ort
    except Exception as e:
        pytest.skip("onnxruntime is not usable here: %s" % type(e).__name__, allow_module_level=True)

import numpy as np
from fastapi.testclient import TestClient

from backend.main import REGISTERED, app
from backend.registered import FEATURES, LAND_COVER_CLASSES

ROOT = Path(__file__).resolve().parent.parent
ACTIVE = [(k, v) for k, v in json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8")).items() if v.get("status") == "active"]


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _reset():
    try:
        app.state.limiter.reset()
    except AttributeError:
        app.state.limiter._storage.reset()


@pytest.fixture(autouse=True)
def fresh_limits():
    _reset()
    yield
    _reset()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _good(n):
    data = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    rel = [k for k, v in data.items() if isinstance(v, dict) and "label_source" in v][0]
    with open(ROOT / rel, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows[:: max(1, len(rows) // 60)]:
        if r["land_cover"] != "" and int(float(r["land_cover"])) in LAND_COVER_CLASSES and all(r[k] != "" for k in ("elevation", "slope", "rainfall_3day", "distance_river")):
            p = {k: float(r[k]) for k in ("elevation", "slope", "rainfall_3day", "distance_river")}
            p["land_cover"] = int(float(r["land_cover"]))
            if r["clay_percent"] != "":
                p["clay_percent"] = float(r["clay_percent"])
            out.append(p)
    return out[:n]


def _direct(sess, p):
    feed = {n: np.array([[np.nan if p.get(n) is None else p[n]]], dtype=np.int64 if n == "land_cover" else np.float64) for n in FEATURES}
    return float(sess.run(["probabilities"], feed)[0][0][1])


def test_health_reports_both_models_and_the_registered_hash(client):
    h = client.get("/health").json()
    reg = h["registered_model"]
    assert h["model_loaded"] is True and h["model_sha256"]
    assert reg["loaded"] is True and reg["sha256"] == ACTIVE[0][1]["sha256"] and reg["trained_on"] == ACTIVE[0][1]["trained_on"]


def test_score_equals_the_committed_artifact_on_real_rows(client):
    sess = ort.InferenceSession(str(ROOT / ACTIVE[0][0]), providers=["CPUExecutionProvider"])
    rows = _good(8)
    assert len(rows) >= 5
    for p in rows:
        resp = client.post("/v2/score", json=p)
        assert resp.status_code == 200, resp.text
        b = resp.json()
        assert abs(b["score"] - _direct(sess, p)) < 1e-4 and 0.0 <= b["score"] <= 1.0
        assert "flood_probability" not in b and "risk_class" not in b and "not a flood probability" in b["score_meaning"]
        assert b["model"]["sha256"] == ACTIVE[0][1]["sha256"] and "not flood probabilities" in b["claim_limit"]


def test_missing_clay_is_scored_with_a_warning(client):
    p = dict(_good(1)[0])
    p.pop("clay_percent", None)
    resp = client.post("/v2/score", json=p)
    assert resp.status_code == 200 and any("clay_percent" in w for w in resp.json()["warnings"])


@pytest.mark.parametrize("change", [{"land_cover": 70}, {"rainfall_3day": -1}, {"distance_river": -5}, {"clay_percent": 101}, {"elevation": None}])
def test_score_rejects_invalid_input(client, change):
    p = dict(_good(1)[0])
    p.update(change)
    resp = client.post("/v2/score", json=p)
    assert resp.status_code == 422 and "detail" in resp.json()


def test_legacy_predict_still_works_and_stays_labelled(client):
    body = next(p for p in _good(60) if "clay_percent" in p)
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200 and "Legacy model, not validated" in resp.json()["notice"]


def test_metrics_v2_matches_the_stored_results(client):
    res = json.loads((ROOT / "docs" / "PHASE_C_RESULTS.json").read_text(encoding="utf-8"))
    r = client.get("/v2/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["model_sha256"] == res["registered_model"]["sha256"] == REGISTERED.sha256
    assert m["per_event_mean_auc"]["mappable"] == res["runs"]["registration"]["metrics"]["mappable/con/per_event_mean"]
    assert m["runs"]["registration"] == res["runs"]["registration"]["id"] and isinstance(m["beyond_elevation_only"]["selection_split"], bool)


def test_the_api_land_cover_classes_equal_the_model_graph():
    if os.environ.get("GITHUB_ACTIONS") == "true":
        import onnx
    else:
        onnx = pytest.importorskip("onnx")
    model = onnx.load(str(ROOT / ACTIVE[0][0]))
    meta = {p.key: p.value for p in model.metadata_props}
    if "land_cover_classes" in meta:
        classes = [int(c) for c in meta["land_cover_classes"].split(",")]
    else:
        classes = [int(i.name.split("_")[1]) for i in model.graph.initializer if i.name.startswith("category_")]
    assert sorted(classes) == sorted(LAND_COVER_CLASSES)


def test_protocol_section_17_exists():
    text = (ROOT / "docs" / "PHASE_C_PROTOCOL.md").read_text(encoding="utf-8")
    assert "## 17. Serving the registered model" in text and "/v2/score" in text
