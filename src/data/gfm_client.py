"""
gfm_client.py — Copernicus Global Flood Monitoring (GFM) data access layer.

Thin, dependency-light interface to the EODC STAC catalogue for retrieving
Sentinel-1-derived flood extent products over an area of interest and time
window. Replaces hand-derived SAR thresholding (V1) with an independently
peer-reviewed, ensemble flood-detection product as the primary label source
for Nyando Flood AI's V2 rebuild.

References
----------
Chini, M. et al. (2017)                 — LIST flood-mapping algorithm
Martinis, S. et al. (2015)               — DLR flood-mapping algorithm
Bauer-Marschallinger, B. et al. (2022)   — TU Wien flood-mapping algorithm
Salamon, P. et al. (2021)                — CEMS/GFM system overview

Notes
-----
Uses the OPEN STAC catalogue (stac.eodc.eu/api/v1) — no authentication
required for search or asset read. Do NOT use the authenticated REST API
(api.gfm.eodc.eu) here; it needs a registered account and adds nothing
for read-only archive access.

Pixel encoding (0 = no flood, 1 = flood, 255 = no-data) was confirmed
empirically against a real ensemble_flood_extent asset before being
hardcoded below — see data/MANIFEST.json provenance notes for that
verification session. Never assume a vendor's encoding without checking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import rioxarray
import xarray as xr
from pystac import Item
from pystac_client import Client
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

GFM_STAC_URL = "https://stac.eodc.eu/api/v1"
GFM_COLLECTION = "GFM"

FLOOD_VALUE = 1
NO_FLOOD_VALUE = 0
NODATA_VALUE = 255


@dataclass(frozen=True)
class FloodExtentResult:
    """Result of a peak-flood-extent query for one event window."""

    flood_mask: xr.DataArray        # boolean, True = flooded pixel
    valid_mask: xr.DataArray        # boolean, True = not no-data
    contributing_item_id: str       # GFM STAC item id with the peak extent
    scene_date: date                # actual Sentinel-1 acquisition date used
    requested_date: date            # event date originally asked for
    is_fallback: bool               # True if scene_date != requested_date

    @property
    def flood_pixel_count(self) -> int:
        return int(self.flood_mask.sum())

    @property
    def valid_pixel_count(self) -> int:
        return max(int(self.valid_mask.sum()), 1)

    @property
    def flood_fraction(self) -> float:
        return self.flood_pixel_count / self.valid_pixel_count


class GFMClient:
    """Client over the EODC GFM STAC catalogue."""

    def __init__(self, stac_url: str = GFM_STAC_URL) -> None:
        self._catalog = Client.open(stac_url)

    def search_items(self, bbox: list[float], date_window: str) -> list[Item]:
        """Search GFM items intersecting bbox within an ISO date_window, e.g. '2020-04-01/2020-04-30'."""
        search = self._catalog.search(
            collections=[GFM_COLLECTION], bbox=bbox, datetime=date_window
        )
        items = list(search.items())
        logger.info("GFM search %s over %s: %d item(s)", date_window, bbox, len(items))
        return items

    def get_peak_flood_extent(
        self,
        aoi: BaseGeometry,
        aoi_crs: str,
        target_date: date,
        window_days: int = 15,
    ) -> Optional[FloodExtentResult]:
        """
        Return the peak observed flood extent clipped to `aoi`, searching
        `window_days` on either side of `target_date` and falling back to
        the nearest available Sentinel-1 pass if the exact date has none.

        Returns None on a genuine data gap — never fabricates a result.
        """
        start = (target_date - timedelta(days=window_days)).isoformat()
        end = (target_date + timedelta(days=window_days)).isoformat()
        bbox = list(aoi.bounds)

        items = self.search_items(bbox, f"{start}/{end}")
        if not items:
            logger.warning(
                "No GFM coverage for %s ± %dd over AOI bounds %s",
                target_date, window_days, bbox,
            )
            return None

        best_flood_px = -1
        best_result: Optional[FloodExtentResult] = None

        for item in items:
            da = rioxarray.open_rasterio(
                item.assets["ensemble_flood_extent"].href, masked=False
            )
            clipped = da.rio.clip([aoi], aoi_crs, drop=True, from_disk=True).squeeze()

            valid = clipped != NODATA_VALUE
            flooded = (clipped == FLOOD_VALUE) & valid
            n_flood = int(flooded.sum())

            if n_flood > best_flood_px:
                best_flood_px = n_flood
                scene_dt = item.datetime.date() if item.datetime else target_date
                best_result = FloodExtentResult(
                    flood_mask=flooded,
                    valid_mask=valid,
                    contributing_item_id=item.id,
                    scene_date=scene_dt,
                    requested_date=target_date,
                    is_fallback=(scene_dt != target_date),
                )

        return best_result
