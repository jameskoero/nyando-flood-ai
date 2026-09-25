"""
Smoke test for GFMClient against a known-good event (see roadmap Section 1:
STAC coverage already confirmed 58 items for this window). Not a substitute
for the full A-Gate suite — that runs against the assembled dataset, not
this client in isolation.
"""
from datetime import date
from shapely.geometry import box

from src.data.gfm_client import GFMClient

NYANDO_AOI = box(34.70, -0.40, 35.40, 0.10)  # from gee_extract_nyando.py


def test_search_finds_known_event():
    client = GFMClient()
    items = client.search_items(
        bbox=list(NYANDO_AOI.bounds), date_window="2020-04-01/2020-04-30"
    )
    assert len(items) > 0, "expected real GFM coverage for the confirmed April 2020 window"


def test_peak_extent_returns_provenance():
    client = GFMClient()
    result = client.get_peak_flood_extent(
        aoi=NYANDO_AOI, aoi_crs="EPSG:4326", target_date=date(2020, 4, 15)
    )
    assert result is not None
    assert result.contributing_item_id
    assert 0.0 <= result.flood_fraction <= 1.0
