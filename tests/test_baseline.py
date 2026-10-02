"""Phase C baseline: offline checks on the real training file (see docs/PHASE_C_PROTOCOL.md)."""
import numpy as np
import pytest

from src.models.baseline import (build_logistic, coefficient_signs, evaluate_arm, run_all,
                                 sensitivity_frames)
from src.models.cv import CONSTRAINTS, EVENT, FEATURE_SETS, LABEL, load_training_frame


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def frame():
    df, _ = load_training_frame()
    return df


@pytest.fixture(scope="module")
def subset(frame):
    g = frame.groupby(EVENT)[LABEL].agg(["size", "sum"])
    both = sorted(g[(g["sum"] > 0) & (g["sum"] < g["size"])].index)[:6]
    return frame[frame[EVENT].isin(both)].reset_index(drop=True)


def test_pipeline_fits_blank_clay_and_nominal_land_cover(frame):
    feats = FEATURE_SETS["six"]
    assert frame["clay_percent"].isna().any()
    fitted = build_logistic(feats).fit(frame[feats], frame[LABEL])
    p = fitted.predict_proba(frame[feats])
    assert p.shape == (len(frame), 2) and np.isfinite(p).all()


def test_unseen_land_cover_class_is_ignored(frame):
    feats = FEATURE_SETS["six"]
    train = frame[frame["land_cover"] != 80]
    held = frame[frame["land_cover"] == 80]
    assert len(held) > 0
    p = build_logistic(feats).fit(train[feats], train[LABEL]).predict_proba(held[feats])
    assert np.isfinite(p).all()


def test_blank_indicator_exists_even_when_training_has_no_blanks(frame):
    feats = FEATURE_SETS["six_no_land_cover"]
    complete = frame[frame["clay_percent"].notna()]
    full = build_logistic(feats).fit(frame[feats], frame[LABEL])
    part = build_logistic(feats).fit(complete[feats], complete[LABEL])
    probe = frame[feats].iloc[:5]
    assert full.named_steps["prep"].transform(probe).shape[1] == part.named_steps["prep"].transform(probe).shape[1]


def test_coefficient_signs_cover_the_four_constrained_features(frame):
    """Signs are reported, not enforced: correlated terrain features can flip a sign in a joint fit."""
    feats = FEATURE_SETS["six"]
    fitted = build_logistic(feats).fit(frame[feats], frame[LABEL])
    signs = coefficient_signs(fitted, feats)
    assert set(signs) == set(CONSTRAINTS)
    assert set(signs.values()) <= {-1, 0, 1}


def test_evaluation_is_deterministic(subset):
    feats = FEATURE_SETS["bare4"]
    a = evaluate_arm(subset, feats)
    b = evaluate_arm(subset, feats)
    for split in ("loeo", "sel", "null_sel"):
        assert a[split]["pooled_auc"] == b[split]["pooled_auc"]


def test_evaluate_arm_returns_every_split(subset):
    r = evaluate_arm(subset, FEATURE_SETS["bare4"])
    for split in ("loeo", "sel", "null_sel"):
        assert 0.0 < r[split]["pooled_auc"] < 1.0
        assert np.isfinite(r[split]["per_event_mean"])


def test_run_all_metrics_are_finite_and_named(subset):
    metrics, _ = run_all(subset)
    assert metrics and all(np.isfinite(v) for v in metrics.values())
    for arm in FEATURE_SETS:
        assert arm + "/sel/pooled_auc" in metrics
    assert "sign/elevation" in metrics
    assert any(k.startswith("six/sel/event/") for k in metrics)


def test_sensitivity_frames_remove_what_they_say(frame):
    floor = frame["elevation"] == frame["elevation"].min()
    assert int(floor.sum()) == 549
    assert int((frame["distance_river"] == 0).sum()) == 15
    frames = sensitivity_frames(frame)
    assert len(frames["no_floor_rows"]) == len(frame) - 549
    assert len(frames["no_zero_distance"]) == len(frame) - 15
    assert len(frames["clay_present"]) == len(frame) - int(frame["clay_percent"].isna().sum())
