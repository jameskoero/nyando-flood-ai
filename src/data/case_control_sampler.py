"""
case_control_sampler.py — Case-control point sampling from GFM flood extent.

For each event window, draws N points from inside the GFM ensemble flood
extent ("case") and N points from outside it ("control"), clipped to the real
Nyando ward AOI. This replaces blind random sampling. The labels come from GFM's
flood extent, not from this project's own covariates.

A control means "not detected as flooded by GFM", not "known dry": in the V2
training file 945 of 2,450 controls (38.6%) lie inside GFM's exclusion mask, where
GFM resets flood pixels to no flood (data/derived/gfm_layer_flags.csv and
docs/PHASE_C_PROTOCOL.md, Section 12).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
from shapely.geometry.base import BaseGeometry

from src.data.gfm_client import GFMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplePoint:
    """One case-control sample point with full provenance."""

    lon: float
    lat: float
    flooded: bool
    event_date: date
    scene_date: date
    gfm_item_id: str


def _pixel_centers_to_lonlat(mask, aoi_crs: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lons, lats) for every True pixel in `mask`, in EPSG:4326.

    The source CRS is the RASTER's own (mask.rio.crs), never the AOI's. GFM
    rasters sit on Equi7 projected grids (metres), so their x/y are not
    lon/lat even when the AOI is. `aoi_crs` is kept only so the call signature
    stays unchanged. If the mask carries no CRS we raise rather than guess.
    """
    import pyproj

    ys, xs = np.where(mask.values)
    x_coords = mask.x.values[xs]
    y_coords = mask.y.values[ys]

    raster_crs = mask.rio.crs
    if raster_crs is None:
        raise ValueError("Flood mask carries no CRS; refusing to guess coordinates.")

    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_wkt(raster_crs.to_wkt()), "EPSG:4326", always_xy=True
    )
    lons, lats = transformer.transform(x_coords, y_coords)
    return np.asarray(lons), np.asarray(lats)


def sample_case_control_points(
    aoi: BaseGeometry,
    aoi_crs: str,
    event_date: date,
    n_per_class: int,
    gfm_client: GFMClient | None = None,
    random_seed: int = 42,
) -> list[SamplePoint]:
    """
    Draw up to n_per_class flooded points and n_per_class non-flooded
    points for one event, clipped to `aoi`.

    Returns fewer than 2 * n_per_class if the flood extent doesn't have
    enough distinct pixels of one class — never fabricates points to hit
    the target count. Returns an empty list if GFM has no coverage for
    this event (see GFMClient.get_peak_flood_extent).
    """
    client = gfm_client or GFMClient()
    result = client.get_peak_flood_extent(aoi=aoi, aoi_crs=aoi_crs, target_date=event_date)

    if result is None:
        logger.warning("No GFM coverage for event %s — cannot sample", event_date)
        return []

    rng = np.random.default_rng(random_seed)

    flood_lons, flood_lats = _pixel_centers_to_lonlat(result.flood_mask, aoi_crs)
    non_flood_mask = result.valid_mask & (~result.flood_mask)
    non_flood_lons, non_flood_lats = _pixel_centers_to_lonlat(non_flood_mask, aoi_crs)

    points: list[SamplePoint] = []

    for lons, lats, label in [
        (flood_lons, flood_lats, True),
        (non_flood_lons, non_flood_lats, False),
    ]:
        n_available = len(lons)
        if n_available == 0:
            logger.warning(
                "Zero %s pixels available for event %s",
                "flood" if label else "non-flood", event_date,
            )
            continue
        n_draw = min(n_per_class, n_available)
        idx = rng.choice(n_available, size=n_draw, replace=False)
        for i in idx:
            points.append(SamplePoint(
                lon=float(lons[i]),
                lat=float(lats[i]),
                flooded=label,
                event_date=event_date,
                scene_date=result.scene_date,
                gfm_item_id=result.contributing_item_id,
            ))

    return points
