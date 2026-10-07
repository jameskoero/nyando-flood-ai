"""Phase C results and closure record (docs/PHASE_C_PROTOCOL.md, Section 16.7; docs/ROADMAP_DEVIATIONS.md D12 and D29)."""
import json
import re
from pathlib import Path

import pytest

from src.models.closure import SHUFFLE_NULL, frame_result, inside_null, registered_arm, relied_upon, render, verdict
from src.models.registration import registered

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def results():
    return json.loads((ROOT / "docs" / "PHASE_C_RESULTS.json").read_text(encoding="utf-8"))


def test_closure_record_is_the_rendering_of_the_results(results):
    d20 = ROOT / "docs" / "D20_RESULTS.json"
    assert (ROOT / "docs" / "PHASE_C_CLOSURE.md").read_text(encoding="utf-8") == render(results, json.loads(d20.read_text(encoding="utf-8")) if d20.exists() else None)


def test_results_are_traceable_to_logged_runs(results):
    data = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    train = [v["sha256"] for v in data.values() if isinstance(v, dict) and "label_source" in v]
    assert len(train) == 1
    for name, run in results["runs"].items():
        assert re.fullmatch(r"[0-9a-f]{32}", run["id"]), name
        assert re.fullmatch(r"[0-9a-f]{7,40}", run["commit"]), name
        assert run["training_data_sha256"] == train[0] and run["status"] == "FINISHED", name
    models = json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    e = models.get(results["registered_model"]["file"])
    assert e is not None and e["sha256"] == results["registered_model"]["sha256"] and e["status"] in ("active", "retired")


def test_decision_code_reproduces_the_registered_arm(results):
    assert registered_arm(results) == results["runs"]["registration"]["registered_arm"]
    assert registered(frame_result(results, "mappable"), frame_result(results, "full")) == registered_arm(results)


def test_registered_model_has_no_monotonicity_violations_in_the_results(results):
    m = results["runs"]["registration"]["metrics"]
    assert all(m["violations/con/%s" % f] == 0 for f in ("rainfall_3day", "elevation", "distance_river", "slope")) and m["eligible"] == 1.0


def test_registered_artifact_is_logged_to_the_registration_run(results):
    rb = results["artifact_readback"]
    row = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D12 |")][0]
    closed = row.strip().strip("|").split("|")[-1].strip() == "closed"
    assert closed == bool(rb["present"])
    if rb["present"]:
        assert rb["run"] == results["runs"]["registration"]["id"]
        assert any(p.endswith(Path(results["registered_model"]["file"]).name) for p in rb["listed"])


def test_load_check_environment_is_pinned():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "numpy==1.26.4" in req and "scikit-learn==1.6.1" in req
    assert "requirements-onnx.txt" in ci and re.search(r"python-version:\s*[\"']?3\.11", ci)


def test_readings_follow_the_protocol_on_arithmetic_vectors():
    """Arithmetic test vectors for the reading logic only; they are not data and no result is claimed from them."""
    assert relied_upon(0.001) and not relied_upon(0.0) and not relied_upon(-0.02)
    lo, hi = SHUFFLE_NULL["mappable"]
    assert inside_null("mappable", (lo + hi) / 2) and not inside_null("mappable", hi + 0.01) and not inside_null("mappable", lo - 0.01)
    assert verdict({"a": False, "b": False}) == "stands" and verdict({"a": False, "b": True}) == "unresolved"
