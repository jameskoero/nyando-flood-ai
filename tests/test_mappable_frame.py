"""Protocol r4 (docs/PHASE_C_PROTOCOL.md, Sections 11 and 12): numbers recomputed from the committed layer flags."""
import numpy as np
import pytest

from src.models.baseline import build_logistic
from src.models.cv import (EVENT, FEATURE_SETS, LABEL, load_layer_flags, load_training_frame, loeo_splits,
                           out_of_fold_scores, per_group_auc, pooled_auc, reference_scores, restrict_to_mappable)


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def frame():
    df, _ = load_training_frame()
    return df


@pytest.fixture(scope="module")
def flags(frame):
    return load_layer_flags(frame)


@pytest.fixture(scope="module")
def mappable(frame, flags):
    return restrict_to_mappable(frame, flags)


def test_flags_cover_every_training_row(frame, flags):
    assert len(flags) == len(frame) == 4420
    assert (flags["row"].to_numpy() == np.arange(len(frame))).all()
    assert set(flags["exclusion_mask"].unique()) == {0, 1}


def test_no_flood_row_lies_in_the_exclusion_or_reference_water_mask(frame, flags):
    cases = frame[LABEL].to_numpy() == 1
    assert int(cases.sum()) == 1970
    assert int(flags.loc[cases, "exclusion_mask"].sum()) == 0
    assert int(flags.loc[cases, "reference_water_mask"].sum()) == 0


def test_controls_in_the_masks_match_the_protocol(frame, flags):
    ctrl = frame[LABEL].to_numpy() == 0
    assert int(flags.loc[ctrl, "exclusion_mask"].sum()) == 945
    assert int(flags.loc[ctrl, "reference_water_mask"].sum()) == 8
    lc = frame.loc[ctrl, "land_cover"]
    assert flags.loc[ctrl, "exclusion_mask"].groupby(lc).sum().to_dict() == {10: 96, 20: 319, 30: 440, 40: 17, 50: 7, 60: 1, 80: 21, 90: 44}
    assert lc.value_counts().to_dict() == {10: 123, 20: 399, 30: 1523, 40: 252, 50: 9, 60: 2, 80: 32, 90: 110}


def test_mappable_frame_matches_the_protocol(mappable):
    assert len(mappable) == 3475
    assert int(mappable[LABEL].sum()) == 1970
    assert int((mappable[LABEL] == 0).sum()) == 1505
    g = mappable.groupby(EVENT)[LABEL].agg(["size", "sum"])
    assert int(((g["sum"] > 0) & (g["sum"] < g["size"])).sum()) == 23


def test_reference_arms_on_the_mappable_frame_match_the_protocol(mappable):
    ref = reference_scores(mappable)
    expected = {"elevation": (0.7945, 0.8975), "hand": (0.7301, 0.7456),
                "distance_river": (0.7827, 0.8749), "slope": (0.6654, 0.6726)}
    for name, (p, e) in expected.items():
        assert ref[name][0] == pytest.approx(p, abs=1e-4), name
        assert ref[name][1] == pytest.approx(e, abs=1e-4), name
        assert ref[name][2] == 23


EXPECTED = {("bare4", "full"): (0.8264, 0.9104), ("bare4", "mappable"): (0.8053, 0.9030),
            ("six", "full"): (0.8988, 0.9391), ("six", "mappable"): (0.8841, 0.9383),
            ("six_no_land_cover", "full"): (0.8361, 0.9108), ("six_no_land_cover", "mappable"): (0.8105, 0.9067)}


def test_arms_on_both_frames_match_the_protocol(frame, mappable):
    frames = {"full": frame, "mappable": mappable}
    for (arm, name), (pooled, per_event) in EXPECTED.items():
        fr, feats = frames[name], FEATURE_SETS[arm]
        sc = out_of_fold_scores(build_logistic(feats), fr, feats, loeo_splits(fr, drop_shared_locations=True))
        pe = per_group_auc(fr, sc, EVENT, 1)
        assert pooled_auc(fr[LABEL], sc) == pytest.approx(pooled, abs=5e-4), (arm, name)
        assert float(np.mean(list(pe.values()))) == pytest.approx(per_event, abs=5e-4), (arm, name)


def test_restricting_to_the_mappable_frame_never_removes_a_flood_row(frame, flags):
    kept = restrict_to_mappable(frame, flags)
    assert int(kept[LABEL].sum()) == int(frame[LABEL].sum())
