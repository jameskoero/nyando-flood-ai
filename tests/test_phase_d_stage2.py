"""Phase D Stage 2 setup (docs/PHASE_D_PROTOCOL.md Section 9; register row D46): the procedure is stated with the rules in code, the selected-point rule and the statistics behave on arithmetic cases, and the cached nested selection equals boosters.nested_scores."""
from pathlib import Path
import numpy as np
import pytest
from src.models import phase_d as D
from src.models.boosters import make_estimator, nested_scores
from src.models.cv import EVENT, FEATURE_SETS, MIN_CLASS_N, load_training_frame
from src.models.phase_d_cv import nested_cached
from src.models.phase_d_stats import stat

ROOT = Path(__file__).resolve().parent.parent
SEC = (ROOT / "docs" / "PHASE_D_PROTOCOL.md").read_text(encoding="utf-8").split("## 9. ")[1]
SIX = list(FEATURE_SETS["six"])


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_procedure_is_stated_with_the_rules_in_code():
    assert D.MIN_CLASS_N == 1 and "MIN_CLASS_N = %d" % D.MIN_CLASS_N in SEC
    assert "%d resamples" % D.N_BOOT in SEC and "alpha %g" % D.ALPHA in SEC and "most outer folds" in SEC and "Block B decides" in SEC and "never a gate" in SEC
    assert all("%g" % want in SEC and "tolerance %g" % tol in SEC for want, tol in D.REPRODUCE.values())


def test_the_selected_point_is_the_most_chosen_and_ties_go_to_the_simpler_point():
    a, b, c = D.GRID[0], D.GRID[1], D.GRID[-1]
    assert D.modal_point([b, b, a], D.GRID) == b and D.modal_point([b, a], D.GRID) == a and D.modal_point([c, b, a], D.GRID) == a
    assert D.modal_point([{"hidden": [64, 32], "lam": 10.0}] * 2 + [a], D.GRID) == c  # points read back from JSON hold lists


def test_the_statistics_use_the_stored_millionths_and_are_deterministic():
    ref = {"e%d" % i: 500000 + 1000 * i for i in range(20)}
    mlp = {e: v + 10000 for e, v in ref.items()}
    s = stat(ref, mlp)
    assert s["n_events"] == 20 and s["wins"] == 20 and s["mean_diff"] == pytest.approx(0.01) and s["ci_low"] > 0 and s == stat(ref, mlp)


def test_the_cached_nested_selection_equals_nested_scores_and_resumes(tmp_path):
    df, _ = load_training_frame(ROOT)
    d6 = df[df[EVENT].isin(sorted(set(df[EVENT]))[:6])].reset_index(drop=True)
    make, space = make_estimator("hgb", SIX, True), {"max_depth": (2, 3), "max_iter": (20,), "min_samples_leaf": (20, 40)}
    a, ca = nested_scores(make, space, d6, SIX)
    cache = tmp_path / "c.json"
    assert nested_cached(make, space, d6, SIX, cache, budget_s=-1) is None
    b, cb = nested_cached(make, space, d6, SIX, cache)

    def boom(p):
        raise AssertionError("a cached fold was recomputed")
    c, cc = nested_cached(boom, space, d6, SIX, cache)
    assert np.allclose(a, b, equal_nan=True) and ca == cb and np.allclose(b, c, equal_nan=True) and cb == cc
