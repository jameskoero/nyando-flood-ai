"""Register row D51, option C: the legacy POST /predict is gone by default (410) and runs only with LEGACY_PREDICT=on; the gate answers before the body is validated, calls no model, uses no rate limit and logs each attempt without naming the caller."""
import logging

from fastapi.testclient import TestClient

from backend import main as api


class _Boom:
    def predict_proba(self, X):
        raise AssertionError("the legacy model must not be called when the route is off")

    predict = predict_proba


def _client(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv("LEGACY_PREDICT", raising=False)
    else:
        monkeypatch.setenv("LEGACY_PREDICT", value)
    api.limiter.reset()
    return TestClient(api.app)


def test_the_legacy_route_is_gone_by_default_and_points_to_v2(monkeypatch):
    r = _client(monkeypatch).post("/predict", json={})
    assert r.status_code == 410
    detail = r.json()["detail"]
    assert detail["error"] == "gone" and detail["use"] == ["/v2/score", "/v2/advisory"]
    assert r.headers["link"].startswith("</v2/score>") and 'rel="successor-version"' in r.headers["link"]
    assert "x-content-type-options" in r.headers


def test_the_gate_answers_before_the_body_is_validated_and_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(api, "model", _Boom())
    c = _client(monkeypatch)
    assert c.post("/predict", json={"elevation": "not a number"}).status_code == 410
    assert c.post("/predict", content=b"{}", headers={"content-type": "application/json"}).status_code == 410


def test_a_gone_route_does_not_use_the_rate_limit(monkeypatch):
    c = _client(monkeypatch)
    codes = {c.post("/predict", json={}).status_code for _ in range(api.GLOBAL_PREDICT_LIMIT + 8)}
    assert codes == {410}


def test_only_the_value_on_turns_the_route_back_on(monkeypatch):
    for value in ("off", "true", "1", ""):
        assert _client(monkeypatch, value).post("/predict", json={}).status_code == 410, value
    for value in ("on", " ON "):
        assert _client(monkeypatch, value).post("/predict", json={}).status_code == 422, value


def test_every_attempt_is_logged_and_the_log_names_no_caller(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="nyando.legacy")
    _client(monkeypatch).post("/predict", json={}, headers={"cf-connecting-ip": "203.0.113.9"})
    lines = [r.getMessage() for r in caplog.records if r.name == "nyando.legacy"]
    assert lines == ["legacy_predict_attempt enabled=False"]


def test_the_v2_routes_are_not_gated(monkeypatch):
    c = _client(monkeypatch)
    assert c.post("/v2/score", json={}).status_code == 422
    assert c.post("/v2/advisory", json={}).status_code == 422
