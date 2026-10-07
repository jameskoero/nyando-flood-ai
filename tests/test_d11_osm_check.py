"""Register row D11: the OSM comparison on file covers the zero-distance rows, the summary follows the rows, and the register, closure record and README state it exactly."""
from pathlib import Path
import json
import pytest
from src.models.cv import load_training_frame

ROOT = Path(__file__).resolve().parent.parent
REC = json.loads((ROOT / "docs" / "D11_OSM_CHECK.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_check_covers_every_zero_distance_row():
    df, tsha = load_training_frame(ROOT)
    z = df[df.distance_river == 0]
    assert REC["training_data_sha256"] == tsha
    assert {(round(float(a), 6), round(float(b), 6)) for a, b in zip(z.lon, z.lat)} == {(r["lon"], r["lat"]) for r in REC["rows"]}
    assert REC["status"] == ("complete" if all(r["status"] == "ok" for r in REC["rows"]) else "incomplete")


def test_the_summary_follows_the_rows_and_a_waterway_entry_has_a_waterway_tag():
    rows, sm = REC["rows"], REC["summary"]
    ok = [r for r in rows if r["status"] == "ok"]
    assert sm["rows"] == len(rows) and sm["answered"] == len(ok) and sm["failed"] == len(rows) - len(ok)
    assert sm["with_waterway_within_50m"] == sum(1 for r in ok if r["waterway_within_50m"])
    assert sm["other_feature_within_50m"] == sum(1 for r in ok if r["other_feature_within_50m"])
    assert sm["inside_water_or_wetland"] == sum(1 for r in ok if r["inside_water_or_wetland"])
    assert all("waterway" in x for r in ok for x in r["waterway_within_50m"])
    assert all("waterway" not in x for r in ok for x in r["other_feature_within_50m"])


def test_the_register_and_closure_record_state_the_result():
    sm, done = REC["summary"], REC["status"] == "complete"
    row = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D11 |")]
    assert len(row) == 1 and "docs/D11_OSM_CHECK.json" in row[0] and REC["snapshot_utc"] in row[0]
    assert "OSM answered %d of %d rows" % (sm["answered"], sm["rows"]) in row[0] and "%d have a waterway line within" % sm["with_waterway_within_50m"] in row[0]
    assert row[0].rstrip().endswith("| closed |" if done else "| open: partial |")
    closure = (ROOT / "docs" / "PHASE_CLOSURE.md").read_text(encoding="utf-8")
    assert "docs/D11_OSM_CHECK.json" in closure and "has not been done" not in closure


def test_the_readme_notice_does_not_call_the_live_rebuild_in_progress():
    notice = [l for l in (ROOT / "README.md").read_text(encoding="utf-8").splitlines() if l.startswith("> **Correction notice")]
    assert len(notice) == 1 and "is in progress" not in notice[0] and "Do not cite the retracted figures" in notice[0]
# END OF FILE
