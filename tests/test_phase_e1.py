"""Phase E E1 (docs/PHASE_E_PROTOCOL.md Sections 4 and 8; register row D49): the shared-location exclusion, the rebuild of the figures from per-event counts and the three gates on known values, plus the run script and the workflow."""
from pathlib import Path
import pytest
from src.advisory import e1
from src.advisory import rules as R

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _rows(shares):
    lv, y, ev = [], [], []
    for e in "abc":
        for level, k in zip(R.LEVELS, shares):
            lv += [level] * 40
            y += [1] * k + [0] * (40 - k)
            ev += [e] * 40
    return lv, y, ev


def test_block_a_rows_at_training_locations_are_not_kept():
    assert e1.keep_mask(["1,2", "3,4", "5,6", "3,4"], ["3,4", "9,9"]) == [True, False, True, False]


def test_the_per_event_counts_rebuild_the_same_figures():
    lv, y, ev = _rows((4, 20, 36))
    s = e1.summarize(lv, y, ev)
    lv2, y2, ev2 = e1.rebuild(s["per_event"])
    assert e1.summarize(lv2, y2, ev2) == s and s["rates"]["HIGH"] == {"rows": 120, "flood": 108, "share": 0.9}


def test_a_clear_ranking_is_validated_and_a_flat_one_is_not():
    good = e1.summarize(*_rows((4, 20, 36)))
    assert good["gates"] == {"V1": True, "V2": True, "V3": True} and good["outcome"] == R.VALIDATED
    assert good["bootstrap"]["diff"] == pytest.approx(0.8) and good["bootstrap"]["ci_low"] == pytest.approx(0.8)
    flat = e1.summarize(*_rows((20, 20, 20)))
    assert not flat["gates"]["V2"] and not flat["gates"]["V3"] and flat["outcome"] == R.NOT_VALIDATED


def test_an_empty_level_fails_the_gates_instead_of_crashing():
    s = e1.summarize(["LOW"] * 150 + ["MEDIUM"] * 150, [0, 1] * 150, ["a", "b", "c"] * 100)
    assert s["bootstrap"]["ci_low"] is None and s["gates"]["V1"] is False and s["outcome"] == R.NOT_VALIDATED


def test_the_run_script_and_the_workflow_state_the_protocol():
    run = (ROOT / "scripts" / "run_phase_e1.py").read_text(encoding="utf-8")
    for s in ("load_registered", "rules.cutpoints", "keep_mask", "location_keys", "loeo_splits(M, drop_shared_locations=True)", "RESULT_JSON", "--log-mlflow", "DAGSHUB_TOKEN", "restrict_to_mappable"):
        assert s in run, s
    wf = (ROOT / ".github" / "workflows" / "e1.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in wf and "contents: read" in wf and ": write" not in wf and "timeout-minutes" in wf
    assert "DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}" in wf and "python scripts/run_phase_e1.py --log-mlflow" in wf and '"3.11"' in wf
