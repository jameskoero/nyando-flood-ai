"""
Smoke test for case-control sampling against the confirmed April 2020
event (58 GFM items already verified for this window earlier in Phase A),
sampled against the real committed ward geometry.
"""
from datetime import date

import geopandas as gpd

from src.data.case_control_sampler import sample_case_control_points


def _load_nyando_aoi():
    wards = gpd.read_file("data/external/nyando_wards.geojson")
    aoi = wards.union_all()
    return aoi, str(wards.crs)


def test_sampling_returns_both_classes():
    aoi, aoi_crs = _load_nyando_aoi()
    points = sample_case_control_points(
        aoi=aoi, aoi_crs=aoi_crs, event_date=date(2020, 4, 15), n_per_class=5,
    )

    assert len(points) > 0, "expected sample points for the confirmed April 2020 event"

    flooded = [p for p in points if p.flooded]
    non_flooded = [p for p in points if not p.flooded]
    assert len(flooded) > 0, "expected at least one flooded point"
    assert len(non_flooded) > 0, "expected at least one non-flooded point"

    for p in points:
        assert p.gfm_item_id, "every point must trace back to a real GFM STAC item"
