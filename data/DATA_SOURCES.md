# Data sources and datasheet: Nyando Flood AI V2

Structured after Gebru et al. (2021), "Datasheets for Datasets". The asset IDs and the GFM endpoint below are compared with the code by `tests/test_data_docs.py`, so this page cannot drift from the pipeline unnoticed.

No personal data is used. No data-protection compliance determination is documented in this repository.

## 1. Motivation
Training data for a research model that ranks locations, inside the areas the Global Flood Monitoring (GFM) product can map, by flood susceptibility in the Nyando ward area (`data/external/nyando_wards.geojson`). It is not an operational warning dataset.

## 2. Composition
- 4,420 rows: 1,970 flood points and 2,450 controls (`flooded` is 1 for a point inside a GFM flood extent, 0 otherwise).
- 35 sample sets: 5 event scenes (April 2020, April 2021, May 2022, April/May 2024, March 2026; 250 floods and 250 controls each) and 30 other Sentinel-1 scene dates (up to 40 of each class). 12 of those 30 dates have no flood pixels, so they contribute controls only.
- Every row has `event_date`, the Sentinel-1 scene date that GFM returned. Rainfall aggregates per sample set (ward area and upstream basin, 3, 7 and 14 days) are in `data/training/nyando_training_v2_multidate_sets.csv`.
- A point with no clay value keeps `clay_percent` blank: blank on 28% of flood points and 1.8% of controls, so a blank is informative. Every other missing value drops the point, and drops are logged. Nothing is imputed.
- 945 of the 2,450 controls (38.6%) lie inside GFM's exclusion mask, where GFM cannot map floods. "Control" therefore means "not detected as flooded", not "known dry".

### 2.1 Confirmatory data: D20 Block A
- `data/confirmatory/nyando_block_a.csv`: 3,780 rows (1,890 floods, 1,890 controls) from 27 of the 27 committed dates (0 failed and were not replaced), 70 points per class per date, built by `scripts/build_block_a.py` through the same pipeline as the training file. Controls outside the GFM exclusion mask: 1,172 of 1,890.
- Its manifest entries carry no `label_source`, so no test treats it as a training file. It was scored once and is not used for training (`docs/D20_PROTOCOL.md`, Sections 12 and 13). A second block of 26 dates (Block B) is selected and sealed for Phase D: it is not built.

## 3. Collection process
- Labels: GFM flood extent from the EODC STAC API (`https://stac.eodc.eu/api/v1`, collection `GFM`).
- Sampling: case-control per scene, because flood pixels are rare (0.07% to 1.3% of valid pixels per scene).
- Dates: the five events are the named floods above. The 30 other dates were chosen by rule: full ward coverage, spread across six equal-count bins of upstream-basin 14-day rainfall, at least 14 days apart, fixed seed, never by flood outcome.
- Features: Earth Engine, through `src/data/terrain_features.py` and `src/data/raw_features.py`; assembly in `scripts/build_initial_dataset.py`.
- Training file: `data/training/nyando_training_v2_multidate.csv`, SHA-256 `0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a`.

## 4. Sources
| Quantity | Source | Earth Engine asset or endpoint | Terms |
|---|---|---|---|
| flooded (label) | Copernicus Global Flood Monitoring flood extent, via EODC | `https://stac.eodc.eu/api/v1`, collection `GFM` | see the provider's terms |
| elevation, slope, HAND | MERIT Hydro | `MERIT/Hydro/v1_0_1` | CC-BY-NC 4.0 or ODbL 1.0 (dual licence, applied to derived data) |
| distance_river | JRC Global Surface Water, band `occurrence` | `JRC/GSW1_4/GlobalSurfaceWater` | see the catalogue entry |
| clay_percent | SoilGrids clay, band `clay_0-5cm_mean` | `projects/soilgrids-isric/clay_mean` | see the catalogue entry |
| land_cover | ESA WorldCover 10 m 2020 (v100) | `ESA/WorldCover/v100` | CC-BY-4.0 |
| rainfall_3day and the rainfall aggregates | CHIRPS daily precipitation | `UCSB-CHG/CHIRPS/DAILY` | see the catalogue entry |
| upstream basin for the rainfall aggregates | HydroSHEDS basins, level 8 | `WWF/HydroSHEDS/v1/Basins/hybas_8` | see the catalogue entry |

The land cover map is the 2020 product, so the April 2020 event shares a year with it. Whether that leaks label information is not excluded (register row D14).

## 5. Licence of the data files in this repository (open decision)
The training files contain values derived from MERIT Hydro (elevation, slope, HAND). Its terms apply to derived data: CC-BY-NC 4.0, or ODbL 1.0, under which derived data must be publicly available under ODbL. Which licence this repository's data files are distributed under is an owner decision, recorded as register row D37 (open). This page is not legal advice.

## 6. Known limitations
- All labels come from one product (GFM). No independent flood-extent check exists yet.
- 23 events can be scored, because the 12 flood-free dates cannot give a per-event AUC.
- Confirmatory replication on new dates is planned (register row D20).

## 7. Maintenance
Rebuild with `scripts/build_initial_dataset.py`. Hashes live in `data/MANIFEST.json` and are recomputed by CI (`data-gate`, `manifest-check`). Changes go through pull requests and `docs/ROADMAP_DEVIATIONS.md`.

## river_adjacent_verified (register row D11)

The column is derived from distance_river == 0 and is not independent evidence of a river. OSM comparison, snapshot 2026-10-07T10:50:03Z (docs/D11_OSM_CHECK.json): OSM answered 15 of 15 rows; of those, 0 have a waterway line within 50 m, 1 have another OSM feature within 50 m that is not a waterway (tags in the file), and 14 lie inside an OSM water or wetland area. The column keeps its name because renaming it would change the training file hash.
