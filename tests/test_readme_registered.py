"""The README states the registered model, the D11 result, the live-data job and the free-tier decision, with figures read from the stored records (register rows D11, D43, D44, D45)."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
REG = json.loads((ROOT / "docs" / "REGISTRATION.json").read_text(encoding="utf-8"))
OSM = json.loads((ROOT / "docs" / "D11_OSM_CHECK.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def iv(s):
    return "%+.4f [%.4f, %.4f]" % (s["mean_diff"], s["ci_low"], s["ci_high"])


def test_the_readme_states_the_registered_model_and_the_free_tier_decision():
    a, v = REG["metrics"]["per_event_mean_auc"], REG["metrics"]["versus_previous_model"]
    for k in ("existing_events_mappable", "block_a_mappable", "existing_events_full", "block_a_full"):
        assert "%.3f" % a[k] in README, k
    for k in ("pooled/mappable", "pooled/full", "buffered", "temporal"):
        assert iv(v[k]) in README, k
    assert "register row D45" in README and "free tier" in README
    assert "`live-data` job" in README and "register row D44" in README
    rows = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D45 |")]
    assert len(rows) == 1 and "tests/test_readme_registered.py::" in rows[0]


def test_the_readme_states_the_d11_result_and_ticks_it_only_when_complete():
    sm = OSM["summary"]
    assert "found %d of the %d zero-distance rows inside OSM water or wetland areas and %d with an OSM waterway line within %d m" % (sm["inside_water_or_wetland"], sm["rows"], sm["with_waterway_within_50m"], OSM["radius_m"]) in README
    assert ("- [x] A (remainder)" in README) == (OSM["status"] == "complete")
