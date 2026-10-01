"""Tests for the Phase C evaluation harness and for the numbers quoted in docs/PHASE_C_PROTOCOL.md.

They run on the committed training file only: no Earth Engine and no model. The AUC tests use the
real labels as scores (a perfect and a reversed ranking), and the bootstrap tests use constant
inputs, so no invented data is involved.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models import cv

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "PHASE_C_PROTOCOL.md"


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this module never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def frame():
    df, _ = cv.load_training_frame()
    return df


def test_frame_is_the_manifest_file():
    df, digest = cv.load_training_frame()
    man = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    entry = next(v for v in man.values() if isinstance(v, dict) and "label_source" in v)
    assert digest == entry["sha256"]
    assert len(df) == entry["row_count"] == 4420
    assert df.index.equals(pd.RangeIndex(len(df)))
    assert set(cv.FEATURE_SETS["seven"]) <= set(df.columns)


def test_bare4_matches_the_data_gate():
    spec = importlib.util.spec_from_file_location("gate", ROOT / "tests" / "test_data_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert cv.BARE_4 == gate.BARE_4


def test_declared_constraints_are_the_roadmap_four():
    assert cv.CONSTRAINTS == {"rainfall_3day": 1, "elevation": -1, "distance_river": -1, "slope": -1}
    assert set(cv.CONSTRAINTS) <= set(cv.FEATURE_SETS["bare4"])


def test_loeo_holds_out_each_event_once(frame):
    splits = cv.loeo_splits(frame)
    assert len(splits) == frame[cv.EVENT].nunique() == 35
    seen = np.concatenate([test for _, _, test in splits])
    assert len(seen) == len(frame) == len(set(seen))
    for e, train, test in splits:
        assert set(frame.loc[test, cv.EVENT]) == {e}
        assert e not in set(frame.loc[train, cv.EVENT])
        assert len(train) + len(test) == len(frame)


def test_location_drop_leaves_no_shared_location(frame):
    keys = cv.location_keys(frame)
    plain = {e: train for e, train, _ in cv.loeo_splits(frame)}
    dropped = 0
    for e, train, test in cv.loeo_splits(frame, drop_shared_locations=True):
        assert not set(keys.iloc[train]) & set(keys.iloc[test])
        assert set(train) <= set(plain[e])
        dropped += len(plain[e]) - len(train)
    assert dropped > 0


def test_lowo_tests_the_four_wards_with_both_classes(frame):
    splits = cv.lowo_splits(frame)
    assert [w for w, _, _ in splits] == ["Ahero", "East Kano/Wawidhi", "Kabonyo/Kanyagwal", "Kobura"]
    for w, train, test in splits:
        assert set(frame.loc[test, cv.WARD]) == {w}
        assert w not in set(frame.loc[train, cv.WARD])


def test_auc_helpers_on_the_real_labels(frame):
    y = frame[cv.LABEL].to_numpy()
    assert cv.pooled_auc(y, y.astype(float)) == 1.0
    assert cv.pooled_auc(y, 1.0 - y) == 0.0
    per = cv.per_group_auc(frame, y.astype(float), cv.EVENT)
    assert len(per) == 23 and all(v == 1.0 for v in per.values())
    s = y.astype(float)
    s[:100] = np.nan
    assert 0.0 <= cv.pooled_auc(y, s) <= 1.0
    summary = cv.summarize_scores(frame, y.astype(float))
    assert summary["pooled_auc"] == 1.0 and summary["per_event_mean"] == 1.0
    assert sorted(summary["per_ward"]) == ["Ahero", "East Kano/Wawidhi", "Kabonyo/Kanyagwal", "Kobura"]


def test_shuffle_keeps_every_event_count(frame):
    shuffled = cv.shuffle_labels_within_events(frame)
    assert frame.groupby(cv.EVENT)[cv.LABEL].sum().equals(shuffled.groupby(cv.EVENT)[cv.LABEL].sum())
    assert (shuffled[cv.LABEL] != frame[cv.LABEL]).any()
    assert shuffled.drop(columns=cv.LABEL).equals(frame.drop(columns=cv.LABEL))


def test_bootstrap_is_deterministic_and_handles_identity_cases():
    base = {"e" + str(i): 0.5 for i in range(23)}
    same = cv.paired_event_bootstrap(base, base)
    assert same["mean_diff"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0 and same["ties"] == 23
    up = {k: v + 0.1 for k, v in base.items()}
    r1, r2 = cv.paired_event_bootstrap(base, up), cv.paired_event_bootstrap(base, up)
    assert r1 == r2 and r1["ci_low"] > 0 and r1["wins"] == 23


def _between_event_share(df, col):
    s = df[[col, cv.EVENT]].dropna()
    total = ((s[col] - s[col].mean()) ** 2).sum()
    group_mean = s.groupby(cv.EVENT)[col].transform("mean")
    return ((group_mean - s[col].mean()) ** 2).sum() / total


def test_protocol_numbers_match_the_data(frame):
    doc = DOC.read_text(encoding="utf-8")
    n = len(frame)
    g = frame.groupby(cv.EVENT)[cv.LABEL].agg(n="size", f="sum")
    both = g[(g.f > 0) & (g.f < g.n)].index
    expected = [
        f"{len(g)} events",
        f"{len(both)} events with both classes",
        f"{int((g.f == 0).sum())} events with no flood rows",
        f"{frame[cv.LABEL].mean():.1%} flood",
    ]
    assert int((g.f == g.n).sum()) == 0
    peaks = frame[frame.sample_mode == "event_peak"]
    pg = peaks.groupby(cv.EVENT)[cv.LABEL].agg(["size", "sum"])
    assert len(pg) == 5 and (pg["size"] == 500).all() and (pg["sum"] == 250).all()
    expected += ["250 flood and 250 control", f"{len(peaks):,} of {n:,} rows ({len(peaks) / n:.1%})"]

    keys = cv.location_keys(frame)
    shared = int((frame.assign(_k=keys).groupby("_k")[cv.EVENT].transform("nunique") > 1).sum())
    expected.append(f"{shared} rows ({shared / n:.1%})")

    blank = frame.clay_percent.isna()
    kk = int((blank & (frame[cv.WARD] == "Kabonyo/Kanyagwal")).sum())
    expected.append(f"`clay_percent` is blank on {int(blank.sum())} rows, {kk} of them in Kabonyo/Kanyagwal")
    floor = frame.elevation == frame.elevation.min()
    assert frame.loc[floor, cv.WARD].nunique() == 1
    expected.append(f"{int(floor.sum())} rows ({floor.mean():.1%}) sit at the DEM elevation floor")

    expected.append(f"between-event variance share of `rainfall_3day` is {_between_event_share(frame, 'rainfall_3day'):.3f}")
    cands = [c for c in frame.columns if c.startswith("rain_")]
    assert len(cands) == 6
    assert all(f"{_between_event_share(frame, c):.3f}" == "1.000" for c in cands)
    expected.append("between-event share 1.000")

    aw = frame[frame[cv.WARD] == "Awasi/Onjiko"]
    expected.append(f"Awasi/Onjiko has {int(aw[cv.LABEL].sum())} flood rows in {len(aw)}")
    lc = frame.groupby("land_cover")[cv.LABEL].agg(["size", "mean"])
    for k in (20, 80):
        expected.append(f"class {k} ({int(lc.loc[k, 'size'])} rows, flood share {lc.loc[k, 'mean']:.3f})")

    sub = frame[frame[cv.EVENT].isin(both)]
    direction = {"rainfall_3day": 1, "elevation": -1, "distance_river": -1, "slope": -1, "hand": -1}
    declared = {"rainfall_3day": "+", "elevation": "-", "distance_river": "-", "slope": "-", "hand": "none"}
    for col, d in direction.items():
        per = np.array([roc_auc_score(s[cv.LABEL], s[col]) for _, s in sub.groupby(cv.EVENT) if s[col].nunique() > 1])
        agree = int(((per - 0.5) * d > 0).sum())
        expected.append(f"| {col} | {declared[col]} | {per.mean():.3f} | {agree} of {len(per)} |")

    missing = [s for s in expected if s not in doc]
    assert not missing, "protocol text does not match the data for: " + "; ".join(missing)
