"""
raw_features.py — distance_river, rainfall_3day, clay_percent, land_cover.

All four sourced from verified real Earth Engine assets:
  distance_river : JRC/GSW1_4/GlobalSurfaceWater, band 'occurrence'
                    (Pekel et al. 2016, Nature) -- distance to a real
                    observed-water mask via fastDistanceTransform(), not
                    Image.distance()+Kernel, which is capped at a 512-
                    pixel kernel radius and cannot represent a real
                    tens-of-km search distance at 30m native resolution.
  rainfall_3day   : UCSB-CHG/CHIRPS/DAILY, band 'precipitation' (mm/day),
                    summed over the 3 days ending at the event date.
                    Batched per event date -- all points sharing a date
                    go in ONE sampleRegions() call, not one per point.
  clay_percent    : projects/soilgrids-isric/clay_mean, band
                    'clay_0-5cm_mean'. Raw units are g/kg; divided by 10
                    to get real percent.
  land_cover      : ESA/WorldCover/v100. Published as an ImageCollection
                    of global tiles, not a single Image -- mosaicked here
                    before use. Static 2020 snapshot -- not event-matched
                    per year, same caveat as HAND.

distance_river == 0 rows get an automatic, reproducible verification:
because the water mask IS the distance source, a zero-distance point by
construction sits on a pixel already classified as water by JRC's own
occurrence data. river_adjacent_verified is therefore computed, not
manually spot-checked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import ee

logger = logging.getLogger(__name__)

JRC_GSW_ASSET = "JRC/GSW1_4/GlobalSurfaceWater"
CHIRPS_ASSET = "UCSB-CHG/CHIRPS/DAILY"
SOILGRIDS_CLAY_ASSET = "projects/soilgrids-isric/clay_mean"
WORLDCOVER_ASSET = "ESA/WorldCover/v100"

WATER_OCCURRENCE_THRESHOLD = 50  # percent; >50% of observed time = water
JRC_GSW_NATIVE_SCALE_M = 30      # documented native resolution

# fastDistanceTransform's neighborhood is in PIXELS, not meters. 1024 px
# at 30m/px = ~30.7km search radius -- generous for a basin whose AOI
# bbox is roughly 70km wide, since real sample points are drawn from
# inside the basin's own flood/non-flood extent, not its far corners.
# Unverified upper limit: the API docs don't state a max neighborhood
# size explicitly; if this value itself throws a size-limit error,
# that will be the next real thing to investigate, not assumed away.
DISTANCE_NEIGHBORHOOD_PX = 1024


@dataclass(frozen=True)
class RawFeatureSample:
    distance_river_m: float
    river_adjacent_verified: bool
    rainfall_3day_mm: float
    clay_percent: float
    land_cover_class: int


def _build_static_image() -> ee.Image:
    """distance_river + clay_percent + land_cover -- none depend on event date."""
    occurrence = ee.Image(JRC_GSW_ASSET).select("occurrence")
    water_mask = occurrence.gt(WATER_OCCURRENCE_THRESHOLD).selfMask()

    # Squared pixel-distance to nearest water pixel, then convert to
    # real meters: sqrt(squared px distance) * native pixel size.
    squared_px_distance = water_mask.fastDistanceTransform(
        neighborhood=DISTANCE_NEIGHBORHOOD_PX,
        units="pixels",
        metric="squared_euclidean",
    )
    distance = (
        squared_px_distance.sqrt()
        .multiply(JRC_GSW_NATIVE_SCALE_M)
        .rename("distance_river")
    )

    clay_raw = ee.Image(SOILGRIDS_CLAY_ASSET).select("clay_0-5cm_mean")
    clay_percent = clay_raw.divide(10).rename("clay_percent")

    land_cover = (
        ee.ImageCollection(WORLDCOVER_ASSET)
        .mosaic()
        .select("Map")
        .rename("land_cover")
    )

    return distance.addBands(clay_percent).addBands(land_cover)


def _build_rainfall_image(event_date: date) -> ee.Image:
    """3-day precipitation sum ending at event_date (inclusive)."""
    start = ee.Date(str(event_date - timedelta(days=2)))
    end = ee.Date(str(event_date + timedelta(days=1)))  # exclusive end
    return (
        ee.ImageCollection(CHIRPS_ASSET)
        .filterDate(start, end)
        .select("precipitation")
        .sum()
        .rename("rainfall_3day")
    )


def extract_raw_features(
    points: list[tuple[float, float]],  # (lon, lat)
    event_date: date,
    scale_m: int = 90,
) -> list[RawFeatureSample | None]:
    """
    Extract all four raw features for every point sharing ONE event_date.
    Call once per event (not once per point) when assembling a full
    dataset across multiple events.

    Returns None for a point missing any required band, rather than
    fabricating a fill value.
    """
    static_image = _build_static_image()
    rainfall_image = _build_rainfall_image(event_date)
    combined = static_image.addBands(rainfall_image)

    fc = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(lon, lat), {"idx": i})
         for i, (lon, lat) in enumerate(points)]
    )
    sampled = combined.sampleRegions(collection=fc, scale=scale_m, geometries=False)

    required = {"distance_river", "rainfall_3day", "clay_percent", "land_cover"}
    results: list[RawFeatureSample | None] = [None] * len(points)

    for feature in sampled.getInfo()["features"]:
        props = feature["properties"]
        idx = props["idx"]
        if not required.issubset(props):
            logger.warning("Missing band(s) at point index %d for event %s", idx, event_date)
            continue
        dist = props["distance_river"]
        results[idx] = RawFeatureSample(
            distance_river_m=dist,
            river_adjacent_verified=(dist == 0),
            rainfall_3day_mm=props["rainfall_3day"],
            clay_percent=props["clay_percent"],
            land_cover_class=int(props["land_cover"]),
        )

    return results
