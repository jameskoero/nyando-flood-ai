"""tests/test_dashboard_evidence.py -- the dashboard's research figures come from one generated file.

frontend/public/evidence.json is built by scripts/build_evidence.py from the stored result files, the
manifests, the training file and the ward polygons. These tests fail if the committed file drifts from a
fresh build, if a headline figure differs from its source, or if the limits are dropped. Standard library
only; no Earth Engine, no network, no secrets.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DASH = REPO / "frontend" / "src" / "dashboard"


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this test never touches Earth Engine."""
    yield


def _builder():
    spec = importlib.util.spec_from_file_location("build_evidence", REPO / "scripts" / "build_evidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evidence():
    return json.loads((REPO / "frontend" / "public" / "evidence.json").read_text(encoding="utf-8"))


def _load(rel):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def test_the_committed_evidence_file_is_a_fresh_build():
    b = _builder()
    fresh = b.render(b.build(REPO))
    assert (REPO / b.OUT).read_text(encoding="utf-8") == fresh, "run: python scripts/build_evidence.py"


def test_headline_figures_equal_their_stored_sources():
    ev = _evidence()
    e = _load("docs/PHASE_E_RESULTS.json")
    p = _load("docs/PROMOTION_RESULTS.json")["stats"]
    reg = _load("docs/REGISTRATION.json")
    assert ev["performance"]["per_event_mean_auc"] == reg["metrics"]["per_event_mean_auc"]
    for name, key in (("pooled", "S6/pooled_mappable"), ("buffered", "buffered"), ("temporal", "temporal")):
        for f in ("mean_diff", "ci_low", "ci_high", "wins", "losses", "ties"):
            assert ev["comparison"][name][f] == p[key][f], (name, f)
    assert ev["levels"]["cutpoints"] == e["cutpoints"]
    assert ev["levels"]["rates"]["HIGH"]["share"] == e["rates"]["HIGH"]["share"]
    assert ev["model"]["sha256"] == e["model_sha256"]
    assert _load("models/MANIFEST.json")[ev["model"]["file"]]["sha256"] == e["model_sha256"]


def test_the_limits_stay_visible_in_the_evidence():
    ev = _evidence()
    awasi = ev["wards"]["items"]["Awasi/Onjiko"]
    assert awasi["estimable"] is False and "auc" not in awasi, "an unavailable AUC must never become zero"
    assert ev["disclaimer"].startswith("Observed flooded shares") and "not basin-wide flood probabilities" in ev["disclaimer"]
    text = " ".join(ev["limits"] + ev["not_done"]).lower()
    for must in ("not flood probabilities", "block b", "not promoted", "buffer", "exclusion mask", "not an official warning"):
        assert must in text, must


def test_the_map_uses_the_repository_ward_polygons():
    ev = _evidence()
    gj = _load("data/external/nyando_wards.geojson")
    assert set(ev["geo"]["wards"]) == {f["properties"]["ward"] for f in gj["features"]}
    assert set(ev["wards"]["items"]) == set(ev["geo"]["wards"])


def test_the_dashboard_features_are_inputs_of_the_score_api():
    ev = _evidence()
    tree = ast.parse((REPO / "backend" / "main.py").read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ScoreInput")
    fields = {s.target.id for s in cls.body if isinstance(s, ast.AnnAssign)}
    keys = [f["key"] for f in ev["features"]]
    assert len(keys) == 6 and set(keys) <= fields, (keys, fields)
    assert sorted(f["key"] for f in ev["features"] if f["direction"] != "unconstrained") == [
        "distance_river", "elevation", "rainfall_3day", "slope"]


def test_the_dashboard_code_holds_no_secrets_or_legacy_endpoints():
    files = [f for ext in ("*.js", "*.jsx") for f in DASH.rglob(ext)]
    assert files
    for f in files:
        s = f.read_text(encoding="utf-8")
        for bad in ("/predict", "nyando-flood-api", "API_KEY", "GEMINI", "GROQ", "localStorage"):
            assert bad not in s, (f.name, bad)


def test_main_renders_the_dashboard_and_the_scoring_page_is_untouched_inside_it():
    main = (REPO / "frontend" / "src" / "main.jsx").read_text(encoding="utf-8")
    assert "import Dashboard from './dashboard/Dashboard.jsx'" in main and "<Dashboard />" in main
    assert 'import("../App.jsx")' in (DASH / "Dashboard.jsx").read_text(encoding="utf-8")
