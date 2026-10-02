"""Phase C boosters (docs/PHASE_C_PROTOCOL.md, Section 13): code grids equal the protocol's, splits are clean, estimators run."""
from pathlib import Path

import numpy as np
import pytest

from src.models.boosters import (HGB_FIXED, HGB_SPACE, INNER_FOLDS, XGB_FIXED, XGB_SPACE, describe, describe_fixed,
                                 expand, inner_splits, make_estimator, nested_scores)
from src.models.cv import (EVENT, FEATURE_SETS, LABEL, land_cover_lookup_scores, load_layer_flags, load_training_frame,
                           location_keys, per_group_auc, pooled_auc, restrict_to_mappable)

ROOT = Path(__file__).resolve().parent.parent
FEATS = FEATURE_SETS["six"]
PARAMS = {"hgb": {"max_depth": 2, "max_iter": 20, "min_samples_leaf": 20},
          "xgb": {"max_depth": 2, "n_estimators": 20, "min_child_weight": 1}}
SMALL = {"max_depth": (2, 3), "max_iter": (20,), "min_samples_leaf": (20,)}


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


def test_grids_in_the_code_equal_the_protocol():
    text = (ROOT / "docs" / "PHASE_C_PROTOCOL.md").read_text(encoding="utf-8")
    for s in (describe(HGB_SPACE), describe(XGB_SPACE), describe_fixed(HGB_FIXED), describe_fixed(XGB_FIXED)):
        assert s in text, s


def test_each_grid_has_eight_points_simplest_first():
    for space in (HGB_SPACE, XGB_SPACE):
        grid = expand(space)
        assert len(grid) == 8 and len({tuple(g.items()) for g in grid}) == 8
        assert list(grid[0].values()) == [min(v) for v in space.values()]
        assert list(grid[-1].values()) == [max(v) for v in space.values()]


def test_inner_splits_are_event_grouped_and_location_clean(frame):
    train = np.flatnonzero((frame[EVENT] != "2020-04").to_numpy())
    splits = inner_splits(frame, train)
    assert len(splits) == INNER_FOLDS
    keys, ev = location_keys(frame).to_numpy(), frame[EVENT].to_numpy()
    seen = []
    for _, tr, te in splits:
        assert set(tr) <= set(train) and set(te) <= set(train) and not set(tr) & set(te)
        assert not set(keys[tr]) & set(keys[te]) and not set(ev[tr]) & set(ev[te])
        seen += sorted(set(ev[te]))
    assert sorted(seen) == sorted(set(ev) - {"2020-04"})


def test_estimators_fit_on_real_rows_and_predict_finite(frame):
    assert frame["clay_percent"].isna().any()
    X, y = frame[FEATS], frame[LABEL].to_numpy()
    for kind in ("hgb", "xgb"):
        for constrained in (True, False):
            p = make_estimator(kind, FEATS, constrained)(PARAMS[kind]).fit(X, y).predict_proba(X)
            assert p.shape == (len(X), 2) and np.isfinite(p).all(), (kind, constrained)


def test_a_land_cover_class_absent_from_training_still_predicts(frame):
    train, held = frame[frame["land_cover"] != 50], frame[frame["land_cover"] == 50]
    assert len(held) > 0
    for kind in ("hgb", "xgb"):
        model = make_estimator(kind, FEATS, True)(PARAMS[kind]).fit(train[FEATS], train[LABEL].to_numpy())
        assert np.isfinite(model.predict_proba(held[FEATS])).all(), kind


def test_nested_scores_are_deterministic_and_choose_from_the_grid(subset):
    make = make_estimator("hgb", FEATS, True)
    s1, c1 = nested_scores(make, SMALL, subset, FEATS)
    s2, c2 = nested_scores(make, SMALL, subset, FEATS)
    assert np.isfinite(s1).all() and np.array_equal(s1, s2) and c1 == c2
    assert len(c1) == subset[EVENT].nunique() and all(c in expand(SMALL) for c in c1)


def test_mappable_land_cover_lookup_matches_the_protocol(frame):
    mapp = restrict_to_mappable(frame, load_layer_flags(frame))
    sc = land_cover_lookup_scores(mapp)
    pe = per_group_auc(mapp, sc, EVENT, 1)
    assert pooled_auc(mapp[LABEL], sc) == pytest.approx(0.7890, abs=5e-4)
    assert float(np.mean(list(pe.values()))) == pytest.approx(0.8843, abs=5e-4)


def test_parallel_nested_scores_equal_the_serial_ones(subset):
    make = make_estimator("hgb", FEATS, True)
    s1, c1 = nested_scores(make, SMALL, subset, FEATS, n_jobs=1)
    s2, c2 = nested_scores(make, SMALL, subset, FEATS, n_jobs=2)
    assert np.array_equal(s1, s2) and c1 == c2
