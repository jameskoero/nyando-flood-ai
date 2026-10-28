"""Environment parity (register row D49; roadmap Section 2, train-check): the CI environment must reproduce the per-event AUCs that Phase D stored for the fixed-parameter hgb:con on the full frame, so a later drift in the libraries or the data shows up as a failing test."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE_MICRO = 2   # the stored values are integers in millionths of AUC


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def compare(stored, computed):
    """The largest per-event difference in millionths; the two event sets must be identical."""
    assert set(stored) == set(computed), "event sets differ: %s" % sorted(set(stored) ^ set(computed))
    return max(abs(int(stored[e]) - int(computed[e])) for e in stored)


def test_compare_reports_the_largest_difference_and_rejects_different_events():
    assert compare({"a": 10, "b": 20}, {"a": 11, "b": 17}) == 3
    with pytest.raises(AssertionError):
        compare({"a": 1}, {"b": 1})


def test_ci_reproduces_the_stored_phase_d_hgb_per_event_aucs_on_the_full_frame():
    from src.models.boosters import make_estimator
    from src.models.cv import FEATURE_SETS, load_training_frame, loeo_splits, out_of_fold_scores
    from src.models.phase_d_stats import aucs
    stored = json.loads((ROOT / "docs" / "PHASE_D_RESULTS.json").read_text(encoding="utf-8"))
    df, sha = load_training_frame()
    assert sha == stored["training_data_sha256"]
    six = list(FEATURE_SETS["six"])
    F = df.reset_index(drop=True)
    scores = out_of_fold_scores(make_estimator("hgb", six, True)(stored["modal_points"]["hgb"]), F, six, loeo_splits(F, drop_shared_locations=True))
    worst = compare(stored["auc_micro"]["full"]["hgb:con"], aucs(F, scores))
    env = stored["environment"]
    assert worst <= TOLERANCE_MICRO, "CI differs from the Phase D run by up to %d millionths of AUC on one event (Phase D: python %s, numpy %s, scikit-learn %s)" % (worst, env["python"], env["numpy"], env["scikit-learn"])
