"""scripts/build_initial_dataset.py -- Stage 4b dataset assembly (build 3: many dates).

Changes vs build 2, each answering a finding from the real data:
  1. 30 extra scene dates spread from dry to wet (chosen from the GFM date scan by basin
     14-day rainfall, at least 14 days apart), so rainfall varies across dates.
  2. Each date is sampled from ALL its same-day GFM slices (date_mosaic). The 5 events keep
     the build-1/2 peak-scene logic so their 2,264 points stay identical.
  3. Flood pixels on any date are labelled flooded=1 (the old dry_control/discard rule is gone).
  4. Candidate rainfall columns: ward-area and upstream-basin means over 3, 7 and 14 days.
  5. One row per sample set in a sets table (flood area, coverage, rainfall, flood patches).
  6. Every drop is logged with the missing band; blanks are labelled 'unattributed'.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

REPO = Path("/content/nyando-flood-ai")
sys.path.insert(0, str(REPO))

from src.data.gfm_client import GFMClient, FloodExtentResult, FLOOD_VALUE, NODATA_VALUE
from src.data.case_control_sampler import sample_case_control_points
from src.data.terrain_features import extract_terrain_features, _build_terrain_image
from src.data.raw_features import extract_raw_features, _build_static_image, _build_rainfall_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_initial_dataset")

# GDAL probes for .aux/.xml sidecars next to every remote GFM raster and gets harmless 403s.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

EE_PROJECT = "nyando-flood-ai"
WARD_GEOJSON_PATH = REPO / "data/external/nyando_wards.geojson"
OUTPUT_CSV_PATH = REPO / "data/training/nyando_training_v2_multidate.csv"
DROPPED_CSV_PATH = REPO / "data/training/nyando_training_v2_multidate_dropped.csv"
SETS_CSV_PATH = REPO / "data/training/nyando_training_v2_multidate_sets.csv"
PRIOR_CSV_PATH = REPO / "data/training/nyando_training_v2.csv"            # build 2
STALE_MANIFEST_PATHS = [REPO / "data/training/nyando_training_v2_raw.csv", PRIOR_CSV_PATH]
MANIFEST_PATH = REPO / "data/MANIFEST.json"

N_PER_CLASS = 250     # per class, per flood event (peak scene)
DATE_N = 40           # per class, per extra scene date
TERRAIN_SCALE_M = 90
RAW_SCALE_M = 90
BASIN_OUTLET_HYBAS_ID = 1081149980   # HydroBASINS level 8; upstream basin = 9 sub-basins, 4,006 km2
PATCH_LINK_M = 60.0
CLAY_POLICY = "keep_blank"   # keep_blank: keep points where SoilGrids has no clay (left blank, never filled); drop: drop them
FULL_VALID_PX = 1050459       # valid pixels when a scene covers the whole ward area
MIN_VALID_PX = int(0.95 * FULL_VALID_PX)
BANDS = ["elevation", "slope", "hand", "distance_river", "clay_percent", "land_cover", "rainfall_3day"]
RAIN_COLS = ["rain_aoi_3d", "rain_aoi_7d", "rain_aoi_14d", "rain_basin_3d", "rain_basin_7d", "rain_basin_14d"]

EVENTS = [
    ("2020-04", date(2020, 4, 15)),
    ("2021-04", date(2021, 4, 15)),
    ("2022-05", date(2022, 5, 15)),
    ("2024-04_05", date(2024, 4, 25)),
    ("2026-03", date(2026, 3, 15)),
]

# Chosen from the real GFM date scan: basin 14-day rainfall in six equal-count bins, five dates
# per bin, seed 42, at least 14 days from each other and from the events. (date, bin 0=driest..5=wettest)
DATE_SETS = [
    ("2020-01-24", 0), ("2020-05-11", 5), ("2020-06-28", 2), ("2020-07-14", 3), ("2020-08-03", 4),
    ("2020-10-06", 4), ("2020-12-17", 1), ("2021-01-18", 1), ("2021-02-27", 2), ("2021-03-23", 2),
    ("2021-08-26", 3), ("2022-01-29", 3), ("2022-03-18", 0), ("2022-07-16", 1), ("2022-09-26", 3),
    ("2022-12-19", 3), ("2023-11-08", 5), ("2024-03-07", 4), ("2024-03-31", 4), ("2024-06-11", 0),
    ("2024-12-20", 1), ("2025-02-18", 0), ("2025-03-14", 2), ("2025-04-19", 5), ("2025-05-13", 5),
    ("2025-08-05", 1), ("2025-12-23", 4), ("2026-01-28", 0), ("2026-03-05", 5), ("2026-07-22", 2),
]

WARD_FIELD_CANDIDATES = ["ward", "WARD", "Ward", "ward_name", "WARD_NAME", "name", "Name", "NAME"]
GEOMS = {}


class CoordinateMismatchError(Exception):
    """Systemic: sampled points fall outside the AOI. Aborts the whole run."""


# ----------------------------------------------------------------------
# GFM clients
# ----------------------------------------------------------------------

class RecordingClient(GFMClient):
    """Peak-scene behaviour exactly as before, but remembers the last extent for statistics."""
    last = None

    def get_peak_flood_extent(self, aoi, aoi_crs, target_date, window_days=15):
        res = super().get_peak_flood_extent(aoi, aoi_crs, target_date, window_days)
        self.last = res
        return res


def same_grid(a, b):
    return (a.shape == b.shape and np.allclose(a.x.values, b.x.values)
            and np.allclose(a.y.values, b.y.values))


class DateMosaicClient(RecordingClient):
    """All GFM slices acquired on the target date, merged: flooded if any slice says flooded."""

    def get_peak_flood_extent(self, aoi, aoi_crs, target_date, window_days=15):
        window = f"{target_date - timedelta(days=1)}/{target_date + timedelta(days=1)}"
        items = [it for it in self.search_items(list(aoi.bounds), window)
                 if it.datetime and it.datetime.date() == target_date]
        if not items:
            self.last = None
            return None
        ref = flood = valid = None
        ids = []
        for it in items:
            da = rioxarray.open_rasterio(it.assets["ensemble_flood_extent"].href, masked=False)
            clipped = da.rio.clip([aoi], aoi_crs, drop=True, from_disk=True).squeeze()
            v = clipped != NODATA_VALUE
            f = (clipped == FLOOD_VALUE) & v
            if ref is None:
                ref, flood, valid = clipped, f, v
            elif not same_grid(ref, clipped):
                logger.warning("%s: slice %s is on a different grid; skipped", target_date, it.id)
                continue
            else:
                flood, valid = flood | f, valid | v
            ids.append(it.id)
        if ref.rio.crs is not None:
            flood = flood.rio.write_crs(ref.rio.crs)
            valid = valid.rio.write_crs(ref.rio.crs)
        res = FloodExtentResult(flood_mask=flood, valid_mask=valid, contributing_item_id=";".join(ids),
                                scene_date=target_date, requested_date=target_date, is_fallback=False)
        self.last = res
        return res


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

def init_earth_engine():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:
        raise SystemExit("Earth Engine is not authenticated in this runtime. Run ee.Authenticate() "
                         "in a notebook cell first.") from exc
    logger.info("Earth Engine initialized (project=%s)", EE_PROJECT)


def load_ward_geometry():
    path = WARD_GEOJSON_PATH
    if not path.exists():
        matches = sorted((REPO / "data").rglob("*ward*.geojson"))
        if not matches:
            raise FileNotFoundError("No ward GeoJSON found under data/")
        path = matches[0]
    wards = gpd.read_file(path)
    if wards.crs is not None and str(wards.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        wards = wards.to_crs("EPSG:4326")
    name_field = next((f for f in WARD_FIELD_CANDIDATES if f in wards.columns), None)
    if name_field is None:
        raise SystemExit(f"No ward-name column among {list(wards.columns)}; aborting before writing rows.")
    print(f"Ward file: {path} | name field '{name_field}' | wards: {sorted(wards[name_field].astype(str).unique())}")
    return wards, name_field, unary_union(list(wards.geometry.values))
# ---- part 2 of 3 ----
def build_rain_geometries(aoi_shp):
    """Ward-area geometry and the HydroBASINS upstream basin used for date-level rainfall."""
    hb = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_8")
    ids = {BASIN_OUTLET_HYBAS_ID}
    frontier = set(ids)
    for _ in range(40):
        up = hb.filter(ee.Filter.inList("NEXT_DOWN", sorted(frontier))).aggregate_array("HYBAS_ID").getInfo()
        new = {int(i) for i in up} - ids
        if not new:
            break
        ids |= new
        frontier = new
    basin_fc = hb.filter(ee.Filter.inList("HYBAS_ID", sorted(ids)))
    area = basin_fc.aggregate_sum("SUB_AREA").getInfo()
    assert len(ids) == 9 and abs(area - 4006) < 5, f"basin changed: {len(ids)} sub-basins, {area:.0f} km2"
    GEOMS["aoi"] = ee.Geometry(mapping(aoi_shp))
    GEOMS["basin"] = basin_fc.geometry()
    GEOMS["chirps"] = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    GEOMS["basin_info"] = {"hydrobasins_level": 8, "upstream_of_hybas_id": BASIN_OUTLET_HYBAS_ID,
                           "sub_basins": len(ids), "area_km2": round(float(area))}
    print(f"basin: {len(ids)} sub-basins, {area:.0f} km2")


def date_rainfall(d):
    """Mean CHIRPS rainfall (mm) over the ward area and the upstream basin, 3/7/14 days ending d."""
    chirps = GEOMS["chirps"]
    end = ee.Date(d.isoformat()).advance(1, "day")

    def win(n):
        return chirps.filterDate(end.advance(-n, "day"), end)

    img = win(3).sum().rename("r3").addBands(win(7).sum().rename("r7")).addBands(win(14).sum().rename("r14"))
    last_err = None
    for attempt in range(3):
        try:
            a = img.reduceRegion(ee.Reducer.mean(), GEOMS["aoi"], 5566, bestEffort=True, maxPixels=1000000000).getInfo()
            b = img.reduceRegion(ee.Reducer.mean(), GEOMS["basin"], 5566, bestEffort=True, maxPixels=1000000000).getInfo()
            n = ee.Dictionary({"n3": win(3).size(), "n7": win(7).size(), "n14": win(14).size()}).getInfo()
            break
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** (attempt + 1))
    else:
        raise RuntimeError(f"rainfall lookup failed for {d}: {last_err}")
    if (n["n3"], n["n7"], n["n14"]) != (3, 7, 14):
        raise ValueError(f"incomplete CHIRPS window for {d}: {n}")
    return {"rain_aoi_3d": a["r3"], "rain_aoi_7d": a["r7"], "rain_aoi_14d": a["r14"],
            "rain_basin_3d": b["r3"], "rain_basin_7d": b["r7"], "rain_basin_14d": b["r14"]}


def assign_ward(lon, lat, wards, name_field):
    hits = wards[wards.geometry.contains(Point(lon, lat))]
    return None if hits.empty else hits.iloc[0][name_field]


def already_done_ids():
    if not OUTPUT_CSV_PATH.exists():
        return set()
    return set(pd.read_csv(OUTPUT_CSV_PATH, usecols=["sample_set"])["sample_set"].unique())


def check_points_in_aoi(points, aoi, tol=0.05):
    minx, miny, maxx, maxy = aoi.bounds
    inside = sum(1 for p in points if minx - tol <= p.lon <= maxx + tol and miny - tol <= p.lat <= maxy + tol)
    print(f"  coordinate check: {inside}/{len(points)} points inside AOI bounds", flush=True)
    if inside / len(points) < 0.95:
        raise CoordinateMismatchError(f"Only {inside / len(points):.0%} of sampled points fall inside "
                                      f"the AOI bounds {aoi.bounds}. Stop and investigate the sampler.")


def missing_bands(lonlat, idxs, scene_date):
    """Audit only: which feature bands have no value at the dropped points."""
    out = {i: [] for i in idxs}
    try:
        combined = (_build_terrain_image().addBands(_build_static_image())
                    .addBands(_build_rainfall_image(scene_date)))
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point(lonlat[i][0], lonlat[i][1]), {"i": i}) for i in idxs])
        for b in BANDS:
            got = combined.select(b).sampleRegions(collection=fc, scale=RAW_SCALE_M).getInfo()["features"]
            present = {f["properties"]["i"] for f in got if b in f["properties"]}
            for i in idxs:
                if i not in present:
                    out[i].append(b)
    except Exception as exc:
        logger.warning("Could not attribute missing bands (%s: %s)", type(exc).__name__, str(exc)[:200])
        out = {i: ["unknown"] for i in idxs}
    return out


# ----------------------------------------------------------------------
# One sample set
# ----------------------------------------------------------------------

def extract_raw_tolerant(lonlat, event_date, scale_m):
    """Like extract_raw_features, but a missing SoilGrids clay value no longer removes the point.

    sampleRegions drops a point when any band is masked, so clay is unmasked to a sentinel inside
    Earth Engine only; the sentinel is turned back into NaN here and is never stored or used."""
    from types import SimpleNamespace
    sentinel = -9999.0
    static = _build_static_image()
    static = static.addBands(static.select("clay_percent").unmask(sentinel), overwrite=True)
    combined = static.addBands(_build_rainfall_image(event_date))
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(lon, lat), {"idx": i})
                               for i, (lon, lat) in enumerate(lonlat)])
    got = combined.sampleRegions(collection=fc, scale=scale_m, geometries=False).getInfo()["features"]
    out = [None] * len(lonlat)
    need = {"distance_river", "rainfall_3day", "land_cover"}
    for f in got:
        q = f["properties"]
        if not need.issubset(q):
            continue
        clay = q.get("clay_percent")
        clay = float("nan") if clay is None or clay == sentinel else clay
        out[q["idx"]] = SimpleNamespace(distance_river_m=q["distance_river"],
                                        river_adjacent_verified=(q["distance_river"] == 0),
                                        rainfall_3day_mm=q["rainfall_3day"], clay_percent=clay,
                                        land_cover_class=int(q["land_cover"]))
    return out


def process_set(job, clients, wards, ward_field, aoi):
    """Returns (rows, dropped_rows, set_stats); rows is None when GFM has no coverage."""
    set_id, anchor, group, mode = job["set_id"], job["anchor"], job["group"], job["mode"]
    print(f"\n--- {set_id} ({mode}, anchor {anchor}) ---", flush=True)
    client = clients[mode]
    n = N_PER_CLASS if mode == "event_peak" else DATE_N
    points = sample_case_control_points(aoi=aoi, aoi_crs="EPSG:4326", event_date=anchor,
                                        n_per_class=n, gfm_client=client)
    if not points:
        return None, [], None

    check_points_in_aoi(points, aoi)
    scene_date = points[0].scene_date
    res = client.last
    valid_px = int(res.valid_mask.sum()) if res is not None else 0
    flood_px = int(res.flood_mask.sum()) if res is not None else 0
    if valid_px < MIN_VALID_PX:
        print(f"  scene covers only {valid_px} of {FULL_VALID_PX} pixels; set skipped (low coverage)")
        return None, [], None
    n_case = sum(1 for p in points if p.flooded)
    print(f"  scene {scene_date}: {flood_px} flood px of {valid_px} valid | sampled {len(points)} "
          f"({n_case} flood / {len(points) - n_case} control)", flush=True)

    rain = date_rainfall(scene_date)
    lonlat = [(p.lon, p.lat) for p in points]
    terrain = extract_terrain_features(lonlat, scale_m=TERRAIN_SCALE_M)
    if CLAY_POLICY == "keep_blank":
        raw = extract_raw_tolerant(lonlat, scene_date, RAW_SCALE_M)
    else:
        raw = extract_raw_features(lonlat, event_date=scene_date, scale_m=RAW_SCALE_M)

    rows, drop_idx = [], []
    for i, (p, t, r) in enumerate(zip(points, terrain, raw)):
        if t is None or r is None:
            drop_idx.append(i)
            continue
        row = {
            "event_id": group, "sample_set": set_id, "sample_mode": mode,
            "sample_type": "flood_case" if p.flooded else "flood_control",
            "event_date": scene_date.isoformat(), "gfm_item_id": p.gfm_item_id,
            "lon": p.lon, "lat": p.lat, "ward": assign_ward(p.lon, p.lat, wards, ward_field),
            "flooded": int(p.flooded), "elevation": t.elevation_m, "slope": t.slope_deg, "hand": t.hand_m,
            "distance_river": r.distance_river_m, "river_adjacent_verified": r.river_adjacent_verified,
            "rainfall_3day": r.rainfall_3day_mm, "clay_percent": r.clay_percent, "land_cover": r.land_cover_class,
        }
        row.update(rain)
        rows.append(row)

    dropped = []
    if drop_idx:
        reasons = missing_bands(lonlat, drop_idx, scene_date)
        for i in drop_idx:
            p = points[i]
            dropped.append({"event_id": group, "sample_set": set_id, "event_date": scene_date.isoformat(),
                            "lon": p.lon, "lat": p.lat, "flooded": int(p.flooded),
                            "ward": assign_ward(p.lon, p.lat, wards, ward_field),
                            "missing": ";".join(reasons[i]) or "unattributed"})
        logger.warning("%s: dropped %d/%d points (%d flood-labelled)", set_id, len(dropped), len(points),
                       sum(d["flooded"] for d in dropped))

    stats = {"sample_set": set_id, "sample_mode": mode, "event_id": group, "scene_date": scene_date.isoformat(),
             "n_slices": res.contributing_item_id.count(";") + 1 if res is not None else 0,
             "valid_px": valid_px, "flood_px": flood_px,
             "flood_frac": flood_px / valid_px if valid_px else float("nan"),
             "n_kept": len(rows), "n_kept_flood": sum(r["flooded"] for r in rows),
             "n_dropped": len(dropped), "n_dropped_flood": sum(d["flooded"] for d in dropped),
             "n_clay_blank": sum(1 for r in rows if r["clay_percent"] != r["clay_percent"])}
    stats.update(rain)
    print(f"  -> {len(rows)} usable rows", flush=True)
    return rows, dropped, stats
# ---- part 3 of 3 ----
# ----------------------------------------------------------------------
# After the run: patches, reproducibility, manifest, summary
# ----------------------------------------------------------------------

def patch_counts(df):
    """Flood-case patches per sample set: points within PATCH_LINK_M are one patch."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    out = {}
    for sid, g in df[df.sample_type == "flood_case"].groupby("sample_set"):
        xy = np.column_stack([g.lon.values * 111320.0, g.lat.values * 110570.0])
        pairs = cKDTree(xy).query_pairs(PATCH_LINK_M, output_type="ndarray")
        if len(pairs):
            m = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(g), len(g)))
            out[sid] = int(connected_components(m, directed=False)[0])
        else:
            out[sid] = len(g)
    return out


def reproducibility_check(df):
    if not PRIOR_CSV_PATH.exists():
        print("\nreproducibility: no build-2 file on disk; skipped")
        return
    old = pd.read_csv(PRIOR_CSV_PATH)
    old = old[old.sample_type != "dry_control"]
    key = lambda d: set(zip(d.event_id, d.lon.round(6), d.lat.round(6)))
    new_keys = key(df[df.sample_mode == "event_peak"])
    old_keys = key(old)
    print(f"\nreproducibility vs build 2: {len(new_keys & old_keys)}/{len(old_keys)} event points re-derived "
          f"identically; {len(new_keys - old_keys)} extra (kept with clay blank under CLAY_POLICY={CLAY_POLICY})")


def write_manifest_entry(csv_path, df, dropped_df, sets_df):
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    for p in STALE_MANIFEST_PATHS:
        if manifest.pop(str(p.relative_to(REPO)), None) is not None:
            print(f"manifest: removed superseded entry {p.relative_to(REPO)}")
    fl = df[df.flooded == 1]
    manifest[str(csv_path.relative_to(REPO))] = {
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "date_added": datetime.now(timezone.utc).date().isoformat(),
        "row_count": int(len(df)),
        "flood_rate": round(float(df["flooded"].mean()), 4),
        "sample_sets": int(df["sample_set"].nunique()),
        "sample_modes": {k: int(v) for k, v in df["sample_mode"].value_counts().items()},
        "sample_types": {k: int(v) for k, v in df["sample_type"].value_counts().items()},
        "flood_patches_total": int(sets_df["flood_patches"].sum()) if "flood_patches" in sets_df else None,
        "label_source": "Copernicus GFM ensemble_flood_extent (Sentinel-1); inundation on the scene date, outside permanent water",
        "date_set_selection": "GFM scene-date scan; basin 14-day rainfall in six equal-count bins, 5 dates per bin, "
                              "seed 42, at least 14 days apart and from the events",
        "rainfall_basin": GEOMS.get("basin_info"),
        "wards_without_flood_labels": sorted(set(df["ward"]) - set(fl["ward"])),
        "dropped_points": int(len(dropped_df)),
        "dropped_flood_labelled": int(dropped_df["flooded"].sum()) if len(dropped_df) else 0,
        "dropped_log": str(DROPPED_CSV_PATH.relative_to(REPO)),
        "sets_table": str(SETS_CSV_PATH.relative_to(REPO)),
        "features": ["elevation", "slope", "hand", "distance_river", "rainfall_3day", "clay_percent", "land_cover"],
        "clay_policy": CLAY_POLICY,
        "min_valid_pixels": MIN_VALID_PX,
        "candidate_rainfall_features": RAIN_COLS,
        "generated_by": "scripts/build_initial_dataset.py (Stage 4b, build 4)",
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote manifest entry -> {MANIFEST_PATH}")


def print_summary(df, dropped_df, sets_df, failed, no_coverage):
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print(f"Rows: {len(df)} | sets: {df.sample_set.nunique()} | flood rate {df['flooded'].mean():.1%}")
    print("\nrows by mode and type:")
    print(pd.crosstab(df.sample_mode, df.sample_type).to_string())
    print("\nflooded by ward:")
    print(pd.crosstab(df.ward, df.flooded).to_string())
    no_fl = sorted(set(df.ward) - set(df[df.flooded == 1].ward))
    if no_fl:
        print(f"\nNO FLOOD LABELS in: {no_fl} -- must be stated in the model card")
    d = sets_df.copy()
    d["valid_vs_max"] = d.valid_px / d.valid_px.max()
    print("\nper set (flood area, coverage vs best scene, flood patches, basin 14-day rain):")
    cols = ["sample_set", "flood_px", "flood_frac", "valid_vs_max", "n_kept_flood", "flood_patches", "rain_basin_14d"]
    print(d[cols].round(4).to_string(index=False))
    print("\nSpearman of set-level flood fraction with rainfall (n = %d sets):" % len(sets_df))
    rho = sets_df[["flood_frac"] + RAIN_COLS].corr(method="spearman")["flood_frac"].drop("flood_frac")
    print(rho.round(2).to_string())
    ind = sets_df[sets_df.sample_mode == "date_mosaic"]
    wet, dry = ind[ind.rain_basin_14d >= 90], ind[ind.rain_basin_14d < 90]
    print("\nindependent date sets: basin 14-day rain >= 90 mm: %d of %d had flood pixels | below 90 mm: %d of %d" %
          ((wet.flood_px > 0).sum(), len(wet), (dry.flood_px > 0).sum(), len(dry)))
    print("points with blank clay kept:", int(sets_df.n_clay_blank.sum()))
    if len(dropped_df):
        print("\ndropped points by set and label:")
        print(dropped_df.groupby(["sample_set", "flooded"]).size().to_string())
        print("missing bands:", dropped_df["missing"].value_counts().to_dict())
    if failed:
        print(f"\nFAILED sets (re-run to retry): {failed}")
    if no_coverage:
        print(f"\nNo GFM coverage / nothing sampleable: {no_coverage}")
    print("\nThis is a readout, not the gate. tests/test_data_gate.py decides mergeability.")


def main():
    init_earth_engine()
    wards, ward_field, aoi = load_ward_geometry()
    build_rain_geometries(aoi)
    clients = {"event_peak": RecordingClient(), "date_mosaic": DateMosaicClient()}

    jobs = [{"set_id": e, "anchor": a, "group": e, "mode": "event_peak"} for e, a in EVENTS]
    jobs += [{"set_id": f"date-{d}", "anchor": date.fromisoformat(d), "group": f"date-{d}", "mode": "date_mosaic"}
             for d, _ in DATE_SETS]
    done = already_done_ids()
    if done:
        print(f"Resuming: already built: {len(done)} sets")

    failed, no_coverage = [], []
    for job in jobs:
        if job["set_id"] in done:
            continue
        t0 = time.time()
        try:
            rows, dropped, stats = process_set(job, clients, wards, ward_field, aoi)
        except CoordinateMismatchError:
            raise
        except Exception as exc:
            logger.error("%s failed (%s: %s); skipped, re-run to retry", job["set_id"],
                         type(exc).__name__, str(exc)[:300])
            failed.append(job["set_id"])
            continue
        if rows is None:
            no_coverage.append(job["set_id"])
            continue
        if not rows:
            failed.append(job["set_id"])
            continue

        OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV_PATH, mode="a", header=not OUTPUT_CSV_PATH.exists(), index=False)
        if dropped:
            pd.DataFrame(dropped).to_csv(DROPPED_CSV_PATH, mode="a", header=not DROPPED_CSV_PATH.exists(), index=False)
        pd.DataFrame([stats]).to_csv(SETS_CSV_PATH, mode="a", header=not SETS_CSV_PATH.exists(), index=False)
        print(f"  wrote {len(rows)} rows, logged {len(dropped)} drops ({time.time() - t0:.0f}s)", flush=True)

    if not OUTPUT_CSV_PATH.exists():
        logger.error("No rows were ever written. failed=%s no_coverage=%s", failed, no_coverage)
        return

    df = pd.read_csv(OUTPUT_CSV_PATH)
    sets_df = pd.read_csv(SETS_CSV_PATH)
    dropped_df = (pd.read_csv(DROPPED_CSV_PATH) if DROPPED_CSV_PATH.exists()
                  else pd.DataFrame(columns=["sample_set", "flooded", "missing"]))
    patches = patch_counts(df)
    sets_df["flood_patches"] = sets_df["sample_set"].map(patches).fillna(0).astype(int)
    sets_df.to_csv(SETS_CSV_PATH, index=False)
    print_summary(df, dropped_df, sets_df, failed, no_coverage)
    reproducibility_check(df)
    write_manifest_entry(OUTPUT_CSV_PATH, df, dropped_df, sets_df)


if __name__ == "__main__":
    main()
