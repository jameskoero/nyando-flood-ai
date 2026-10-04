"""Phase C robustness battery (docs/PHASE_C_PROTOCOL.md, Section 16): subsets, splits and statistics on real rows and arithmetic vectors."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.constrained import build_constrained_logistic
from src.models.cv import (EVENT, FEATURE_SETS, LABEL, load_layer_flags, load_training_frame, loeo_splits, location_keys,
                           out_of_fold_scores, per_group_auc, restrict_to_mappable)
from src.models.robustness import (BUFFER_M, CUTOFF_YEAR, EARTH_RADIUS_M, FLOOR_ELEVATION, MIN_TEST_EVENTS, N_REPEATS, beats,
                                   buffered_splits, coordinate_columns, paired, permutation_audit, prior_only_scores, reversal,
                                   straddling_events, straddling_events, subset_masks, temporal_holdout, temporal_split)

ROOT = Path(__file__).resolve().parent.parent
SIX = FEATURE_SETS["six"]


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def frame():
    df, _ = load_training_frame()
    return df


@pytest.fixture(scope="module")
def mappable(frame):
    return restrict_to_mappable(frame, load_layer_flags(frame))


@pytest.fixture(scope="module")
def subset(frame):
    g = frame.groupby(EVENT)[LABEL].agg(["size", "sum"])
    both = sorted(g[(g["sum"] > 0) & (g["sum"] < g["size"])].index)[:6]
    return frame[frame[EVENT].isin(both)].reset_index(drop=True)


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def test_sensitivity_subsets_have_the_documented_sizes(frame):
    masks = subset_masks(frame)
    assert int((~masks["no_floor_rows"]).sum()) == 549
    assert int((~masks["no_zero_distance"]).sum()) == 15
    assert 0 < int(masks["clay_present"].sum()) < len(frame)


def test_buffered_split_removes_every_neighbour_and_keeps_the_test_folds(mappable):
    lon, lat = coordinate_columns(mappable)
    la, lo = mappable[lat].to_numpy(float), mappable[lon].to_numpy(float)
    base, buf = loeo_splits(mappable, drop_shared_locations=True), buffered_splits(mappable)
    assert [b[0] for b in buf] == [b[0] for b in base]
    removed = 0
    for (_, tr0, te0), (_, tr1, te1) in zip(base, buf):
        assert np.array_equal(te0, te1) and set(tr1) <= set(tr0)
        removed += len(tr0) - len(tr1)
    assert removed > 0
    for (_, tr0, te0), (_, tr1, te1) in list(zip(base, buf))[:4]:
        d = _haversine_m(la[tr1][:, None], lo[tr1][:, None], la[te1][None, :], lo[te1][None, :])
        assert float(d.min()) > BUFFER_M * 0.999
        gone = np.setdiff1d(tr0, tr1)
        if len(gone):
            dg = _haversine_m(la[gone][:, None], lo[gone][:, None], la[te1][None, :], lo[te1][None, :])
            assert float(dg.min(axis=1).max()) <= BUFFER_M * 1.001


def test_temporal_split_is_clean(frame):
    tr, te = temporal_split(frame)
    year = pd.to_datetime(frame["event_date"]).dt.year.to_numpy()
    keys = location_keys(frame).to_numpy()
    assert len(tr) > 0 and len(te) > 0
    assert year[tr].max() <= CUTOFF_YEAR < year[te].min()
    assert not set(keys[tr]) & set(keys[te]) and not set(frame[EVENT].iloc[tr]) & set(frame[EVENT].iloc[te])
    assert not set(straddling_events(frame)) & (set(frame[EVENT].iloc[tr]) | set(frame[EVENT].iloc[te]))


def test_temporal_test_events_are_counted_consistently(mappable):
    r = temporal_holdout(mappable, {"con": (build_constrained_logistic, SIX)})
    _, te = temporal_split(mappable)
    g = mappable.iloc[te].groupby(EVENT)[LABEL].agg(["min", "max"])
    assert set(r["events"]["con"]) == set(g.index[g["min"] != g["max"]])


def test_permutation_audit_is_deterministic_and_consistent_with_the_fold_scores(subset):
    r1 = permutation_audit(build_constrained_logistic, subset, SIX, n_repeats=2)
    r2 = permutation_audit(build_constrained_logistic, subset, SIX, n_repeats=2)
    assert r1["base"] == r2["base"] and r1["within_drop"] == r2["within_drop"] and r1["pooled_drop"] == r2["pooled_drop"]
    ref = per_group_auc(subset, out_of_fold_scores(build_constrained_logistic(SIX), subset, SIX, loeo_splits(subset, drop_shared_locations=True)), EVENT, 1)
    assert set(r1["base"]) == set(ref) and all(abs(r1["base"][e] - ref[e]) < 1e-9 for e in ref)
    assert set(r1["within_drop"]) == set(SIX) and all(len(v) == 2 for v in r1["pooled_drop"].values())


def test_prior_only_scores_are_constant_inside_each_event(subset):
    sc = prior_only_scores(subset)
    assert np.isfinite(sc).all()
    for _, rows in subset.groupby(EVENT).indices.items():
        assert float(np.ptp(sc[rows])) == 0.0
    assert np.allclose(list(per_group_auc(subset, sc, EVENT, 1).values()), 0.5)


def test_paired_beats_and_reversal_on_arithmetic_vectors():
    """Arithmetic test vectors for the decision logic only; they are not data and no result is claimed from them."""
    champ = {"e%d" % i: 0.90 + 0.001 * i for i in range(12)}
    up, down = {k: v + 0.05 for k, v in champ.items()}, {k: v - 0.05 for k, v in champ.items()}
    assert beats(champ, up)[0] is True and beats(champ, down)[0] is False and beats(champ, dict(champ))[0] is False
    rev, detail = reversal(champ, {"hgb_con": down, "xgb_con": up})
    assert rev is True and detail["xgb_con"]["beats"] is True and detail["hgb_con"]["beats"] is False
    assert reversal(champ, {"hgb_con": down})[0] is False
    p = paired(champ, up)
    assert abs(p["mean_diff"] - 0.05) < 1e-9 and p["n"] == 12


def test_protocol_section_16_states_the_rules():
    text = (ROOT / "docs" / "PHASE_C_PROTOCOL.md").read_text(encoding="utf-8")
    assert "## 16. Robustness battery" in text
    for tag in ("16.0", "16.1", "16.2", "16.3", "16.4", "16.5", "16.6", "16.7"):
        assert tag in text, tag
    for s in ("cutoff year %d" % CUTOFF_YEAR, "%d repeats" % N_REPEATS, "%d scorable test events" % MIN_TEST_EVENTS,
              "%s m" % format(int(BUFFER_M), ","), str(FLOOR_ELEVATION), "549 on the full frame", "15 zero-distance rows"):
        assert s in text, s
