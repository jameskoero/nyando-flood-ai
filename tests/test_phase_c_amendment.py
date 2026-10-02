"""Protocol r3 (docs/PHASE_C_PROTOCOL.md, Sections 9 and 10): numbers recomputed from the training file."""
import numpy as np
import pytest

from src.models.baseline import build_logistic
from src.models.cv import (FEATURE_SETS, LABEL, SEED, elevation_floor_mask, land_cover_lookup_scores,
                           load_training_frame, loeo_splits, null_distribution, null_interval, out_of_fold_scores,
                           per_group_auc, pooled_auc, reference_scores, selection_metric_rule,
                           stratified_pooled_auc, EVENT)


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def frame():
    df, _ = load_training_frame()
    return df


def test_reference_arms_match_the_protocol(frame):
    ref = reference_scores(frame)
    expected = {"elevation": (0.8239, 0.9077), "hand": (0.7871, 0.8061),
                "distance_river": (0.7988, 0.8771), "slope": (0.7276, 0.7243)}
    for name, (p, e) in expected.items():
        assert ref[name][0] == pytest.approx(p, abs=1e-4), name
        assert ref[name][1] == pytest.approx(e, abs=1e-4), name
        assert ref[name][2] == 23


def test_land_cover_lookup_matches_the_protocol(frame):
    sc = land_cover_lookup_scores(frame)
    pe = per_group_auc(frame, sc, EVENT, 1)
    assert pooled_auc(frame[LABEL], sc) == pytest.approx(0.7844, abs=1e-4)
    assert float(np.mean(list(pe.values()))) == pytest.approx(0.8687, abs=1e-4)


def test_floor_rows_match_the_protocol(frame):
    floor = elevation_floor_mask(frame)
    assert int(floor.sum()) == 549
    assert int(frame.loc[floor, LABEL].sum()) == 529
    assert int((1 - frame.loc[floor, LABEL]).sum()) == 20
    assert (frame.loc[floor, "hand"] == 0).all()
    assert set(frame.loc[floor, "ward"]) == {"Kabonyo/Kanyagwal"}


def test_rows_outside_the_water_defined_classes_match_the_protocol(frame):
    keep = ~frame["land_cover"].isin([10, 80, 90])
    assert int(keep.sum()) == 3195
    assert int(frame.loc[keep, LABEL].sum()) == 1010


def test_stratified_elevation_auc_matches_the_protocol(frame):
    strata = stratified_pooled_auc(frame, -frame["elevation"].to_numpy())
    expected = {"land_cover_10": 0.930, "land_cover_30": 0.814, "land_cover_40": 0.453,
                "land_cover_80": 0.594, "land_cover_90": 0.776}
    for name, v in expected.items():
        assert strata[name] == pytest.approx(v, abs=6e-4), name
    assert "not_elevation_floor" in strata
    for name in ("elevation_floor", "land_cover_20", "land_cover_50", "land_cover_60"):
        assert name not in strata, name + " has too few rows of one class to be scored"


def test_six_arm_selection_split_reproduces_the_logged_baseline(frame):
    """Tolerance 5e-4: this also checks the logged Colab value under whatever Python this test runs on."""
    feats = FEATURE_SETS["six"]
    sc = out_of_fold_scores(build_logistic(feats), frame, feats, loeo_splits(frame, drop_shared_locations=True))
    pe = per_group_auc(frame, sc, EVENT, 1)
    assert pooled_auc(frame[LABEL], sc) == pytest.approx(0.8988, abs=5e-4)
    assert float(np.mean(list(pe.values()))) == pytest.approx(0.9391, abs=5e-4)


def test_first_null_draw_reproduces_the_logged_null(frame):
    feats = FEATURE_SETS["six"]
    pooled, per_event = null_distribution(build_logistic(feats), frame, feats, 1, base_seed=SEED)
    assert pooled[0] == pytest.approx(0.4573, abs=5e-4)
    assert per_event[0] == pytest.approx(0.5053, abs=5e-4)


def test_selection_rule_logic_uses_arithmetic_test_vectors():
    """Arithmetic test vectors for the decision logic only. They are not data and no result is claimed from them."""
    centred = np.linspace(0.45, 0.55, 101)
    below = np.linspace(0.40, 0.46, 101)
    assert null_interval(centred)["excludes_half"] is False
    assert null_interval(below)["excludes_half"] is True
    assert selection_metric_rule(centred, centred) == "pooled_auc"
    assert selection_metric_rule(below, centred) == "per_event_mean"
    assert selection_metric_rule(below, below) == "unresolved"
