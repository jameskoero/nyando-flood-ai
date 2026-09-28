"""tests/test_data_gate.py -- the A-Gate as CI-enforced pytest (Roadmap Sections 1 and 2).

Gates every training file listed in data/MANIFEST.json that carries a `label_source`.
Pure pandas / numpy / scipy / scikit-learn: no Earth Engine, no network, no secrets, so it also
runs on fork PRs. Two deviations from the roadmap text, each justified by the real data:

  * "clay_percent flat between classes" is kept only as an expected-failure report. Labels come
    from GFM and never from clay, so a class difference reflects floodplain soils, not circularity
    (in the old circular file clay was flat BECAUSE its labels came from a formula that ignored it).
    It is replaced by: no single feature separates the classes (AUC < 0.90) plus provenance checks.
  * A repeated value above 3% needs a documented physical reason AND the top value must actually
    satisfy that reason in the data, so a stale or invented reason fails the gate.
"""

import hashlib
import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "data" / "MANIFEST.json"
WARDS_PATH = REPO / "data" / "external" / "nyando_wards.geojson"

RAIN_COLS = ["rain_aoi_3d", "rain_aoi_7d", "rain_aoi_14d", "rain_basin_3d", "rain_basin_7d", "rain_basin_14d"]
BARE_4 = ["elevation", "slope", "rainfall_3day", "distance_river"]
SINGLE_FEATURES = ["elevation", "slope", "hand", "distance_river", "rainfall_3day", "clay_percent"] + RAIN_COLS
REPEAT_CHECKED = SINGLE_FEATURES + ["land_cover"]
REQUIRED_COLUMNS = {"event_id", "sample_set", "sample_mode", "sample_type", "event_date", "gfm_item_id",
                    "lon", "lat", "ward", "flooded", "elevation", "slope", "hand", "distance_river",
                    "river_adjacent_verified", "rainfall_3day", "clay_percent", "land_cover"} | set(RAIN_COLS)
MAY_BE_BLANK = {"clay_percent"}     # SoilGrids has no value at seasonal-water fringes; never filled

# feature -> (reason, check that the most repeated value really satisfies the reason)
DOCUMENTED = {
    "hand": ("HAND is exactly 0 on MERIT drainage-network cells and many points sit in the channel and floodplain",
             lambda top, s, df: top == 0.0),
    "slope": ("the Kano plain is flat: slope is exactly 0 on flat 90 m DEM cells",
              lambda top, s, df: top == 0.0),
    "elevation": ("many lake-level plain points share the DEM's lowest value; all such rows must have HAND 0",
                  lambda top, s, df: top == s.min() and bool((df.loc[s == top, "hand"] == 0).all())),
    "rainfall_3day": ("CHIRPS pixels are about 5.5 km wide so points share values; the top value is a dry window (0 mm)",
                      lambda top, s, df: top == 0.0),
    "land_cover": ("categorical ESA WorldCover class, not a continuous measurement",
                   lambda top, s, df: True),
}
for _c in RAIN_COLS:
    DOCUMENTED[_c] = ("date-level feature: one value per scene date by construction (checked separately)",
                      lambda top, s, df: True)


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this gate never touches Earth Engine."""
    yield


def _manifest():
    assert MANIFEST_PATH.exists(), "data/MANIFEST.json is missing"
    return json.loads(MANIFEST_PATH.read_text())


def _entries():
    return {k: v for k, v in _manifest().items() if isinstance(v, dict) and "label_source" in v}


@lru_cache(maxsize=None)
def _frame(rel):
    return pd.read_csv(REPO / rel)


def _each():
    entries = _entries()
    assert entries, "data/MANIFEST.json lists no training file with a label_source"
    for rel, entry in entries.items():
        yield rel, entry, _frame(rel)


def _ward_bounds():
    gj = json.loads(WARDS_PATH.read_text())
    xs, ys, names = [], [], set()

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for x in c:
                walk(x)

    for f in gj["features"]:
        names.add(f["properties"].get("ward"))
        walk(f["geometry"]["coordinates"])
    return min(xs), min(ys), max(xs), max(ys), names


def test_manifest_lists_a_training_file():
    assert _entries(), "no training file in data/MANIFEST.json"


def test_file_integrity_and_format():
    for rel, entry, df in _each():
        raw = (REPO / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], f"{rel}: sha256 differs from the manifest"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{rel}: has a BOM (re-saved by a spreadsheet app?)"
        assert b"\r\n" not in raw, f"{rel}: has Windows line endings (re-saved by a spreadsheet app?)"
        assert len(df) == entry["row_count"], f"{rel}: row_count differs from the manifest"


def test_required_columns_and_missing_values():
    for rel, entry, df in _each():
        missing = REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"{rel}: missing columns {sorted(missing)}"
        blank = {c for c in df.columns if df[c].isna().any()}
        assert blank <= MAY_BE_BLANK, f"{rel}: unexpected blank values in {sorted(blank - MAY_BE_BLANK)}"
        assert set(df.flooded.unique()) <= {0, 1}
        assert ((df.sample_type == "flood_case") == (df.flooded == 1)).all(), f"{rel}: sample_type disagrees with flooded"


def test_row_count_and_flood_rate():
    for rel, entry, df in _each():
        assert len(df) > 1000, f"{rel}: only {len(df)} rows"
        rate = df.flooded.mean()
        assert 0.10 <= rate <= 0.90, f"{rel}: flood rate {rate:.1%} is not double-digit or is implausible"


def test_every_row_has_a_real_event_date_matching_its_gfm_scene():
    for rel, entry, df in _each():
        assert df.event_date.str.match(r"^\d{4}-\d{2}-\d{2}$").all(), f"{rel}: event_date is not ISO"
        d = pd.to_datetime(df.event_date)
        assert d.min() >= pd.Timestamp("2015-01-01") and d.max() <= pd.Timestamp(date.today()), f"{rel}: date out of range"
        assert df.gfm_item_id.str.startswith("ENSEMBLE_FLOOD_").all(), f"{rel}: a row has no GFM scene id"
        for iso, ids in zip(df.event_date, df.gfm_item_id):
            found = re.findall(r"ENSEMBLE_FLOOD_(\d{8})T", ids)
            assert found and all(f"{x[:4]}-{x[4:6]}-{x[6:]}" == iso for x in found), f"{rel}: date {iso} != scene {ids}"


def test_labels_come_from_gfm():
    for rel, entry, df in _each():
        assert "Copernicus GFM" in entry["label_source"], f"{rel}: manifest label_source is not GFM"


def test_points_lie_inside_the_ward_area():
    x0, y0, x1, y1, names = _ward_bounds()
    for rel, entry, df in _each():
        tol = 0.001
        assert df.lon.between(x0 - tol, x1 + tol).all() and df.lat.between(y0 - tol, y1 + tol).all(), f"{rel}: point outside the wards"
        assert df.ward.notna().all(), f"{rel}: a row has no ward"
        assert set(df.ward) <= names, f"{rel}: unknown ward names {sorted(set(df.ward) - names)}"


def test_bare_4_feature_logistic_regression_is_not_circular():
    for rel, entry, df in _each():
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        p = cross_val_predict(pipe, df[BARE_4].values, df.flooded.values, cv=LeaveOneGroupOut(),
                              groups=df.event_id.values, method="predict_proba")[:, 1]
        auc = roc_auc_score(df.flooded, p)
        assert auc < 0.90, f"{rel}: leave-one-event-out AUC {auc:.3f} >= 0.90; the labels look derived from these features"


def test_no_single_feature_separates_the_classes():
    for rel, entry, df in _each():
        for c in SINGLE_FEATURES:
            d = df[[c, "flooded"]].dropna()
            a = roc_auc_score(d.flooded, d[c])
            assert max(a, 1 - a) < 0.90, f"{rel}: {c} alone separates the classes (AUC {max(a, 1 - a):.3f})"


def test_repeated_values_have_a_documented_physical_reason():
    for rel, entry, df in _each():
        for c in REPEAT_CHECKED:
            s = df[c].dropna()
            vc = s.round(6).value_counts(normalize=True)
            top, share = vc.index[0], vc.iloc[0]
            if share <= 0.03:
                continue
            assert c in DOCUMENTED, f"{rel}: {c} repeats {top} in {share:.1%} of rows with no documented reason"
            reason, ok = DOCUMENTED[c]
            assert ok(top, s, df), f"{rel}: {c} top value {top} ({share:.1%}) does not satisfy its documented reason: {reason}"
        for c in RAIN_COLS:
            assert df.groupby("sample_set")[c].nunique().max() == 1, f"{rel}: {c} varies within a scene date"


def test_river_adjacent_flag_is_consistent():
    for rel, entry, df in _each():
        assert df.river_adjacent_verified.dtype == bool, f"{rel}: river_adjacent_verified is not boolean"
        zero = df.distance_river == 0
        assert df.loc[zero, "river_adjacent_verified"].all(), f"{rel}: a zero-distance row is not flagged"
        assert not df.loc[~zero, "river_adjacent_verified"].any(), f"{rel}: a flagged row has distance > 0"


def test_event_scenes_have_balanced_case_control_samples():
    for rel, entry, df in _each():
        for sid, g in df[df.sample_mode == "event_peak"].groupby("sample_set"):
            share = g.flooded.mean()
            assert 0.30 <= share <= 0.70, f"{rel}: event {sid} flood share {share:.2f} is outside 0.30-0.70"


def test_manifest_and_sets_table_agree_with_the_data():
    for rel, entry, df in _each():
        assert entry["sample_sets"] == df.sample_set.nunique()
        assert entry["sample_types"] == {k: int(v) for k, v in df.sample_type.value_counts().items()}
        assert round(float(df.flooded.mean()), 4) == entry["flood_rate"]
        sets = pd.read_csv(REPO / entry["sets_table"])
        assert set(sets.sample_set) == set(df.sample_set) and int(sets.n_kept.sum()) == len(df), f"{rel}: sets table disagrees"
        assert int(sets.n_clay_blank.sum()) == int(df.clay_percent.isna().sum()), f"{rel}: blank-clay count disagrees"
        if entry["dropped_points"] > 0:
            log = pd.read_csv(REPO / entry["dropped_log"])
            assert len(log) == entry["dropped_points"], f"{rel}: dropped log disagrees with the manifest"


@pytest.mark.xfail(strict=False, reason=(
    "Original roadmap rule 'clay_percent flat between classes'. Expected to differ on real GFM labels: "
    "floodplain soils are clay-rich and labels never use clay. Kept as a visible report, not a blocker."))
def test_clay_percent_is_flat_between_classes_original_rule():
    for rel, entry, df in _each():
        d = df.dropna(subset=["clay_percent"])
        p = stats.mannwhitneyu(d.clay_percent[d.flooded == 0], d.clay_percent[d.flooded == 1]).pvalue
        assert p > 0.01, f"{rel}: clay differs between classes (Mann-Whitney p = {p:.1e})"
