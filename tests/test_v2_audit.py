# Regression tests for the Sep 29 2026 audit.
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
BODY = {"elevation": 1142.5, "slope": 2.3, "rainfall_3day": 87.4, "distance_river": 320.0,
        "clay_percent": 42.1, "land_cover": 1, "ward": "Ahero"}


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    yield


@pytest.fixture(scope="module")
def api():
    spec = importlib.util.spec_from_file_location("nyando_api_main", REPO / "backend" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metrics_unavailable_is_503_and_not_cached(api, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", str(tmp_path / "m.pkl"))
    monkeypatch.setitem(api._metrics_cache, "data", None)
    monkeypatch.setitem(api._metrics_cache, "loaded_at", 0)
    c = TestClient(api.app)
    r = c.get("/metrics")
    assert r.status_code == 503 and r.json().get("available") is False
    (tmp_path / "metrics.json").write_text(json.dumps({"note": "x"}))
    r2 = c.get("/metrics")
    assert r2.status_code == 200 and r2.json()["note"] == "x"


def test_predict_labels_the_legacy_model(api):
    api.limiter.reset()
    r = TestClient(api.app).post("/predict", json=BODY)
    assert r.status_code == 200
    assert "not validated" in r.json().get("notice", "").lower()


@pytest.mark.parametrize("old,new", [("1142.5", "NaN"), ("320.0", "Infinity")])
def test_predict_rejects_non_finite_numbers(api, old, new):
    api.limiter.reset()
    bad = json.dumps(BODY).replace(old, new)
    r = TestClient(api.app).post("/predict", content=bad, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_predict_rate_limit_is_enforced_in_app(api):
    api.limiter.reset()
    c = TestClient(api.app)
    codes = [c.post("/predict", json=BODY).status_code for _ in range(12)]
    assert codes[0] == 200 and 429 in codes, codes


def test_readme_api_section_matches_the_code():
    src = (REPO / "backend" / "main.py").read_text()
    readme = (REPO / "README.md").read_text()
    for t in ("0.35", "0.60", "0.80"):
        assert ("prob < " + t) in src and t in readme, t
    assert "CRITICAL" in readme
    assert "risk_label" not in readme or "risk_label" in src


def _cv(df, drop_seen_locations):
    cols = ["elevation", "slope", "rainfall_3day", "distance_river"]
    z = df.dropna(subset=cols).reset_index(drop=True)
    X = z[cols].values
    y = z["flooded"].astype(int).values
    g = z["event_id"].values
    loc = (z.lon.round(5).astype(str) + "_" + z.lat.round(5).astype(str)).values
    p = np.full(len(y), np.nan)
    per = []
    for e in np.unique(g):
        te = g == e
        tr = ~te
        if drop_seen_locations:
            tr = tr & ~np.isin(loc, loc[te])
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) == 2:
            per.append(roc_auc_score(y[te], p[te]))
    ok = ~np.isnan(p)
    return roc_auc_score(y[ok], p[ok]), float(np.mean(per))


def test_readme_reports_the_three_cv_figures():
    df = pd.read_csv(REPO / "data" / "training" / "nyando_training_v2_multidate.csv")
    pooled, per_event = _cv(df, False)
    located, _ = _cv(df, True)
    assert abs(pooled - 0.8412) < 0.0005
    readme = (REPO / "README.md").read_text()
    for name, v in (("pooled", pooled), ("per-event mean", per_event), ("location-excluded", located)):
        assert format(v, ".3f") in readme, name + " " + format(v, ".3f") + " missing from README"


def test_predict_returns_503_when_the_model_is_not_loaded(api, monkeypatch):
    api.limiter.reset()
    monkeypatch.setattr(api, "model", None)
    r = TestClient(api.app).post("/predict", json=BODY)
    assert r.status_code == 503 and r.json().get("model_loaded") is False


def test_ward_length_is_capped(api):
    api.limiter.reset()
    c = TestClient(api.app)
    assert c.post("/predict", json=dict(BODY, ward="A" * 64)).status_code == 200
    assert c.post("/predict", json=dict(BODY, ward="A" * 65)).status_code == 422


def test_env_file_variants_are_gitignored():
    import subprocess
    for f in (".env", ".env.local", ".env.production", "backend/.env.local"):
        assert subprocess.run(["git", "check-ignore", "-q", f], cwd=REPO).returncode == 0, f


def test_openapi_schema_is_not_served(api):
    api.limiter.reset()
    r = TestClient(api.app).get('/openapi.json')
    assert r.status_code == 404


def test_predict_has_a_global_cap_across_client_addresses(api):
    api.limiter.reset()
    codes = []
    for i in range(api.GLOBAL_PREDICT_LIMIT + 10):
        c = TestClient(api.app, client=('198.51.100.' + str(i % 250 + 1), 50000))
        codes.append(c.post('/predict', json=BODY).status_code)
    assert codes.count(200) == api.GLOBAL_PREDICT_LIMIT, codes
    assert set(codes) <= {200, 429}, codes
