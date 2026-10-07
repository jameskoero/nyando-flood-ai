"""Phase D protocol (docs/PHASE_D_PROTOCOL.md; register row D46): the text states the rules in src/models/phase_d.py, the gates behave on arithmetic vectors, Block B stays sealed and production has no torch."""
import json
from pathlib import Path
import pytest
from src.models import phase_d as D

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "docs" / "PHASE_D_PROTOCOL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_protocol_states_the_frozen_rules():
    assert D.COMPARATOR in TEXT and D.PROMOTED in TEXT and D.NOT_PROMOTED in TEXT
    for h in D.HIDDEN:
        assert str(h) in TEXT and "%d parameters" % D.param_count(h) in TEXT
    for lam in D.LAMBDAS:
        assert "penalty weight %g" % lam in TEXT
    for k, v in D.FIXED.items():
        assert "%s %s" % (k, v) in TEXT
    assert "%d inner event-grouped folds" % D.INNER_FOLDS in TEXT and "Under %d scorable events" % D.MIN_BLOCK_B_EVENTS in TEXT
    assert "at most %g" % D.PARITY_TOLERANCE in TEXT and "No synthetic rows" in TEXT and "nyando_mlp_" in TEXT
    row = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D46 |")]
    assert len(row) == 1 and "tests/test_phase_d.py::" in row[0]


def test_a_few_thousand_parameters_and_the_grid_is_ordered_simplest_first():
    assert [D.param_count(h) for h in D.HIDDEN] == [1025, 3073] and len(D.GRID) == 6
    assert D.GRID[0] == {"hidden": (32, 16), "lam": 0.0} and D.GRID[-1] == {"hidden": (64, 32), "lam": 10.0}


def test_the_constraints_and_land_cover_classes_equal_the_ones_hgbcon_uses():
    from src.models.boosters import LAND_COVER_CLASSES
    from src.models.cv import CONSTRAINTS
    assert D.CONSTRAINTS == CONSTRAINTS and list(D.LAND_COVER_CLASSES) == list(LAND_COVER_CLASSES)


def test_statistical_gates_on_arithmetic_vectors():
    win, tie, lose = {"mean_diff": 0.02, "ci_low": 0.005}, {"mean_diff": 0.02, "ci_low": -0.001}, {"mean_diff": -0.01, "ci_low": -0.03}
    assert D.gates(20, win, win, win, 0.01) == {"G1": True, "G2": True, "G3": True}
    assert D.gates(20, win, tie, win, 0.01)["G1"] is False and D.gates(20, win, win, lose, 0.01)["G2"] is False and D.gates(20, win, win, win, 0.0)["G3"] is False
    assert not any(D.gates(D.MIN_BLOCK_B_EVENTS - 1, win, win, win, 0.01).values()) and all(D.gates(D.MIN_BLOCK_B_EVENTS, win, win, win, 0.01).values())


def test_export_gates_and_the_decision():
    zero = {"rainfall_3day": 0, "elevation": 0, "distance_river": 0, "slope": 0}
    ok = D.export_gates(1e-6, 1e-7, zero, True)
    assert all(ok.values()) and D.export_gates(2e-5, 0, zero, True)["X1"] is False and D.export_gates(0, 0, dict(zero, slope=1), True)["X3"] is False
    good = {"G1": True, "G2": True, "G3": True}
    assert D.decision(good, ok) == D.PROMOTED and D.decision(good) is None
    assert D.decision(dict(good, G2=False), ok) == D.NOT_PROMOTED and D.decision(good, dict(ok, X3=False)) == D.NOT_PROMOTED


def test_block_b_stays_sealed_until_the_freeze_file_is_merged():
    names = [p.name for p in (ROOT / "data" / "confirmatory").glob("*") if "block_b" in p.name.lower()]
    names += [k for k in json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8")) if "block_b" in k.lower()]
    assert D.block_b_allowed((ROOT / "docs" / "PHASE_D_FREEZE.json").exists(), names)
    assert D.block_b_allowed(False, []) and not D.block_b_allowed(False, ["x_block_b.csv"]) and D.block_b_allowed(True, ["x_block_b.csv"])


def test_production_files_do_not_mention_torch():
    for f in ("requirements.txt", "requirements-onnx.txt", "constraints-ci.txt", "Dockerfile"):
        assert "torch" not in (ROOT / f).read_text(encoding="utf-8").lower(), f
