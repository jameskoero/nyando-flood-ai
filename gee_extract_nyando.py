"""
gee_extract_nyando.py
=====================
Google Earth Engine script to extract real satellite features for
Nyando River Basin flood risk modelling.

Run in: Google Colab (requires GEE authentication)
Output: data/training/nyando_training_v1_raw_gee.csv

Features extracted:
  elevation      — NASA NASADEM 30m
  slope          — Derived from NASADEM
  rainfall_3day  — CHIRPS v2 3-day sum (April 2024 flood event)
  distance_river — HydroSHEDS + OpenStreetMap
  clay_percent   — ISRIC SoilGrids 0-5cm
  land_cover     — ESA WorldCover 2021
  population     — WorldPop Kenya 2020
  flooded        — Sentinel-1 SAR (VV backscatter threshold)

Geographic bounds:
  lon: 34.70 – 35.40 E
  lat: 0.40 S  – 0.10 N  (Nyando sub-county, Kisumu County, Kenya)

Usage:
  1. Open in Google Colab
  2. Run: ee.Authenticate() then ee.Initialize()
  3. Run all cells
  4. Download nyando_training_v1_raw_gee.csv from Colab /content/
"""

# ─── Colab Install ────────────────────────────────────────────────
# !pip install earthengine-api geemap -q

import ee
import geemap
import pandas as pd
import numpy as np
from pathlib import Path

# ─── 1. AUTHENTICATE & INITIALIZE ────────────────────────────────
print("Authenticating GEE...")
ee.Authenticate()
ee.Initialize(project="ee-jmskoero")   # replace with your GEE project ID
print("GEE initialised ✅")

# ─── 2. STUDY AREA ───────────────────────────────────────────────
nyando_bounds = ee.Geometry.Rectangle([34.70, -0.40, 35.40, 0.10])

print("Study area: Nyando Basin")
print("  lon 34.70–35.40 E, lat 0.40 S–0.10 N")

# ─── 3. SAMPLE GRID (2,308 points on 500m grid) ──────────────────
sample_grid = ee.FeatureCollection.randomPoints(
    region=nyando_bounds,
    points=2308,
    seed=42
)
print(f"Sample grid: 2,308 points ✅")

# ─── 4. ELEVATION & SLOPE ─────────────────────────────────────────
nasadem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
slope   = ee.Terrain.slope(nasadem)

# ─── 5. RAINFALL (CHIRPS — 3-day sum around April 2024 flood) ─────
chirps_period = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
    .filterDate("2024-04-20", "2024-04-24") \
    .sum() \
    .rename("rainfall_3day")

# ─── 6. SOIL CLAY (SoilGrids 0-5cm) ──────────────────────────────
soilgrids = ee.Image("projects/soilgrids-isric/clay_mean") \
    .select("b0").rename("clay_percent")

# ─── 7. LAND COVER (ESA WorldCover 2021) ─────────────────────────
worldcover = ee.ImageCollection("ESA/WorldCover/v200") \
    .first().rename("land_cover")

# ─── 8. DISTANCE TO RIVER (OSM via GEE) ──────────────────────────
# Use HydroSHEDS river network as proxy
hydrosheds = ee.Image("WWF/HydroSHEDS/15ACC") \
    .rename("acc")
# Approximate distance: pixels with high flow accumulation = rivers
river_mask = hydrosheds.gte(500).Not()
dist_river = river_mask.fastDistanceTransform().rename("distance_river")

# ─── 9. POPULATION (WorldPop 2020) ────────────────────────────────
worldpop = ee.ImageCollection("WorldPop/GP/100m/pop") \
    .filter(ee.Filter.eq("country", "KEN")) \
    .filter(ee.Filter.eq("year", 2020)) \
    .first().rename("population")

# ─── 10. SENTINEL-1 SAR FLOOD LABELS (April 2024) ─────────────────
# Threshold: VV backscatter < -16 dB → likely water/flood
s1_flood_event = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filterDate("2024-04-18", "2024-04-28") \
    .filterBounds(nyando_bounds) \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
    .filter(ee.Filter.eq("instrumentMode", "IW")) \
    .select("VV") \
    .mean()

flood_label = s1_flood_event.lt(-16.0) \
    .rename("flooded") \
    .toInt()

print("All feature images prepared ✅")

# ─── 11. STACK ALL FEATURES ──────────────────────────────────────
feature_stack = nasadem \
    .addBands(slope) \
    .addBands(chirps_period) \
    .addBands(dist_river) \
    .addBands(soilgrids) \
    .addBands(worldcover) \
    .addBands(worldpop) \
    .addBands(flood_label)

print("Feature stack built ✅")

# ─── 12. SAMPLE AT GRID POINTS ───────────────────────────────────
print("Sampling features at 2,308 grid points (this takes ~2 min)...")
sampled = feature_stack.sampleRegions(
    collection=sample_grid,
    scale=100,           # 100m resolution
    geometries=True,
    tileScale=4
)

# ─── 13. EXPORT TO CSV ───────────────────────────────────────────
print("Exporting to CSV...")

# Convert to pandas
features_list = sampled.getInfo()
records = []
for feat in features_list["features"]:
    props = feat["properties"]
    coords = feat["geometry"]["coordinates"]
    records.append({
        "lon":          round(coords[0], 6),
        "lat":          round(coords[1], 6),
        "elevation":    props.get("elevation", np.nan),
        "slope":        round(props.get("slope", np.nan), 2),
        "rainfall_3day":round(props.get("rainfall_3day", np.nan), 1),
        "distance_river":round(props.get("distance_river", np.nan), 0),
        "clay_percent": round(props.get("clay_percent", np.nan), 1),
        "land_cover":   props.get("land_cover", np.nan),
        "population":   props.get("population", 0),
        "flooded":      int(props.get("flooded", 0)),
    })

df = pd.DataFrame(records)

# Remove nulls
df = df.dropna(subset=["elevation","slope","rainfall_3day"])
print(f"\nExtracted: {len(df)} points")
print(f"Flood rate: {df['flooded'].mean():.1%}")
print(f"Elevation range: {df['elevation'].min():.0f}–{df['elevation'].max():.0f}m")
print(f"Rainfall range: {df['rainfall_3day'].min():.1f}–{df['rainfall_3day'].max():.1f}mm")
print(f"Nulls: {df.isnull().sum().sum()}")

# ─── 14. SAVE ────────────────────────────────────────────────────
output_path = "/content/nyando_training_v1_raw_gee.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
print("Upload this file to data/training/nyando_training_v1_raw_gee.csv in the repo")

# ─── 15. QUICK QA ─────────────────────────────────────────────────
print("\n=== DATA QA ===")
print(df.describe().round(2).to_string())
print("\n=== LAND COVER CLASSES ===")
print(df.land_cover.value_counts())

print("""
=== NEXT STEPS ===
1. Download nyando_training_v1_raw_gee.csv from Colab
2. Run notebooks/03_modelling.ipynb with this CSV as input
3. commit: git add data/training/ && git commit -m "data: real GEE extraction"
""")
