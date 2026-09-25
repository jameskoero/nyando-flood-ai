"""
terrain_features.py — Elevation, slope, and HAND extraction via Google Earth Engine.

HAND (Height Above Nearest Drainage) is pulled from MERIT/Hydro/v1_0_1
(Yamazaki et al. 2019, Water Resources Research, doi:10.1029/2019WR024873),
band 'hnd'. It is a static terrain layer — valid for any flood event
regardless of the event's own date, since drainage topology doesn't change
on human timescales (barring rare channel avulsions).

Elevation and slope are derived from the same MERIT/Hydro product for
internal consistency with the HAND layer, rather than mixing DEM sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import ee

logger = logging.getLogger(__name__)

MERIT_HYDRO_ASSET = "MERIT/Hydro/v1_0_1"


@dataclass(frozen=True)
class TerrainSample:
    """Terrain feature values at one point."""

    elevation_m: float
    slope_deg: float
    hand_m: float


def _build_terrain_image() -> ee.Image:
    merit = ee.Image(MERIT_HYDRO_ASSET)
    elevation = merit.select("elv").rename("elevation")
    hand = merit.select("hnd").rename("hand")
    slope = ee.Terrain.slope(elevation).rename("slope")
    return elevation.addBands(slope).addBands(hand)


def extract_terrain_features(
    points: list[tuple[float, float]],  # (lon, lat) pairs
    scale_m: int = 90,
) -> list[TerrainSample | None]:
    """
    Extract elevation, slope, and HAND for each (lon, lat) point.

    Returns None for a point where GEE has no data (e.g. outside the
    MERIT/Hydro coverage extent) rather than fabricating a value.
    """
    image = _build_terrain_image()
    fc = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(lon, lat), {"idx": i}) for i, (lon, lat) in enumerate(points)]
    )

    sampled = image.sampleRegions(collection=fc, scale=scale_m, geometries=False)
    results: list[TerrainSample | None] = [None] * len(points)

    for feature in sampled.getInfo()["features"]:
        props = feature["properties"]
        idx = props["idx"]
        if "elevation" not in props or "hand" not in props or "slope" not in props:
            logger.warning("No terrain data at point index %d — outside MERIT/Hydro coverage?", idx)
            continue
        results[idx] = TerrainSample(
            elevation_m=props["elevation"],
            slope_deg=props["slope"],
            hand_m=props["hand"],
        )

    return results
