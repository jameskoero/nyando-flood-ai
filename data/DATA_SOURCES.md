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
| flooded (label) | Copernicus Global Flood Monitoring flood extent, via EODC | `https://stac.eodc.eu/api/v1`, collection `GFM` | Copernicus Emergency Management Service terms: free reproduction, distribution, adaptation and combination with other data, with the notice in Section 4.1; not an early warning |
| elevation, slope, HAND | MERIT Hydro | `MERIT/Hydro/v1_0_1` | ODbL 1.0 chosen from the dual licence CC-BY-NC 4.0 or ODbL 1.0 (applied to derived data); citation in Section 4.1 |
| distance_river | JRC Global Surface Water, band `occurrence` | `JRC/GSW1_4/GlobalSurfaceWater` | Copernicus Programme: free of charge, without restriction of use; acknowledge and cite (Section 4.1) |
| clay_percent | SoilGrids clay, band `clay_0-5cm_mean` | `projects/soilgrids-isric/clay_mean` | CC BY 4.0 (ISRIC); citation in Section 4.1 |
| land_cover | ESA WorldCover 10 m 2020 (v100) | `ESA/WorldCover/v100` | CC-BY-4.0 |
| rainfall_3day and the rainfall aggregates | CHIRPS daily precipitation | `UCSB-CHG/CHIRPS/DAILY` | public domain, copyright waived (Earth Engine catalogue entry); citation in Section 4.1 |
| upstream basin for the rainfall aggregates | HydroSHEDS basins, level 8 | `WWF/HydroSHEDS/v1/Basins/hybas_8` | HydroSHEDS version 1 licence agreement: free for commercial use, notice in Section 4.1; the layers themselves are not distributed here |

The land cover map is the 2020 product, so the April 2020 event shares a year with it. Whether that leaks label information is not excluded (register row D14).

### 4.1 Required credits and notices

Anyone who shares these data, or works made from them, should keep these credits.

- Copernicus Global Flood Monitoring (flood labels): Contains modified Copernicus Emergency Management Service information 2020-2026. The years are those of the scene dates in the training and Block A files. The service terms allow free reproduction, distribution, adaptation and combination with other data, and say the data are not an early warning.
- MERIT Hydro (elevation, slope, HAND): Yamazaki, D., et al. (2019), MERIT Hydro: a high-resolution global hydrography map based on latest topography datasets, Water Resources Research 55, 5053-5073, doi:10.1029/2019WR024873.
- JRC Global Surface Water (distance_river): Pekel, J.-F., Cottam, A., Gorelick, N., Belward, A. S. (2016), High-resolution mapping of global surface water and its long-term changes, Nature 540, 418-422, doi:10.1038/nature20584. Source: EC JRC/Google.
- SoilGrids (clay_percent): Poggio, L., et al. (2021), SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty, SOIL 7, 217-240, doi:10.5194/soil-7-217-2021. CC BY 4.0, ISRIC.
- ESA WorldCover 10 m 2020 v100 (land_cover): CC BY 4.0, credit the ESA WorldCover project.
- CHIRPS (rainfall): Funk, C., et al. (2015), The climate hazards infrared precipitation with stations, a new environmental record for monitoring extremes, Scientific Data 2, 150066, doi:10.1038/sdata.2015.66. In the public domain.
- HydroSHEDS (basins for the rainfall aggregates): the HydroSHEDS version 1 licence agreement requires this notice with any product that incorporates its data: "This product (the Nyando Flood AI data files) incorporates data from the HydroSHEDS version 1 database which is (c) World Wildlife Fund, Inc. (2006-2022) and has been used herein under license. WWF has not evaluated the data as altered and incorporated within the Nyando Flood AI data files, and therefore gives no warranty regarding its accuracy, completeness, currency or suitability for any particular purpose. Portions of the HydroSHEDS v1 database incorporate data which are the intellectual property rights of (c) USGS (2006-2008), NASA (2000-2005), ESRI (1992-1998), CIAT (2004-2006), UNEP-WCMC (1993), WWF (2004), Commonwealth of Australia (2007), and Her Royal Majesty and the British Crown and are used under license. The HydroSHEDS v1 database and more information are available at https://www.hydrosheds.org." The citation is Lehner, B., Verdin, K., Jarvis, A. (2008), New global hydrography derived from spaceborne elevation data, Eos, Transactions, AGU 89(10), 93-94. The licence does not allow its layers to be distributed as a stand-alone product, so the files here hold only rainfall aggregates computed over the basins: no HydroSHEDS layer, geometry or basin identifier.

### 4.2 Ward polygons

data/external/nyando_wards.geojson holds five ward polygons (Ahero, Awasi/Onjiko, East Kano/Wawidhi, Kabonyo/Kanyagwal, Kobura; Nyando constituency, Kisumu county), committed on 2026-09-25 (commit 5765ef5). They are a modified subset, filtered to Kisumu county and Nyando constituency, of the Kenya County Assembly Boundaries dataset by Benedict Aboki Omare and George Omare (2017), https://github.com/benaboki/Kenya-County-Assembly-Boundaries, licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The dataset's own README lists the IEBC 2012 delimitation review, KNBS and SID 2013 and OpenStreetMap among its references. The commit message records no source; the source is recorded from the owner's working session of 25 and 26 September 2026 and agrees with the file's attribute names, which are those of that dataset.

## 5. Licence of the data files in this repository

Decision of the repository owner on 2026-10-31, recorded as register row D37 (closed). This page is not legal advice.

- ODbL 1.0 (https://opendatacommons.org/licenses/odbl/1.0/): the V2 data files, that is data/training/nyando_training_v2_multidate.csv and its sets file, the four data/confirmatory/nyando_block_a files and the files in data/derived/. They contain values derived from MERIT Hydro (elevation, slope, HAND), whose terms offer CC-BY-NC 4.0 or ODbL 1.0 and ask that derived data made public go under ODbL. ODbL was chosen: anyone may use these files, commercially too, if they credit the sources in Section 4.1, share adapted versions under ODbL and keep them open.
- CC BY 4.0: data/external/nyando_wards.geojson, five ward polygons taken from another creator's dataset (Section 4.2). They keep that licence and are not relicensed here.
- Not covered by this statement: the legacy v1 files data/training/nyando_training_v1.csv and data/training/nyando_training_v1_raw_gee.csv (data of the retracted v1 model). Their sources are in data/external/sources.md, an older list that this datasheet supersedes for V2 data (it lists CHIRPS as CC-BY, while the Earth Engine catalogue entry gives public domain).
- Code: MIT (LICENSE).
- The ODbL files hold derived values and ward names only: no MERIT Hydro raster and no HydroSHEDS layer, geometry or basin identifier.

## 6. Known limitations
- All labels come from one product (GFM). No independent flood-extent check exists yet.
- 23 events can be scored, because the 12 flood-free dates cannot give a per-event AUC.
- Confirmatory replication on new dates is planned (register row D20).

<!-- phase-k:begin -->
### 6.1 Ward-level coverage (Phase K)

Counts by ward in the two frames of `docs/PHASE_K_RESULTS.json`. They are counts only: the samples are case-control, so they are not flood rates.

| Ward | Primary: flood / control rows | Primary: events with both classes | Secondary: flood / control rows | Secondary: events with both classes | Estimable (primary / secondary) |
|---|---|---|---|---|---|
| Ahero | 196 / 181 | 14 | 334 / 225 | 13 | yes / yes |
| Awasi/Onjiko | 4 / 247 | 1 | 2 / 207 | 1 | no / no |
| East Kano/Wawidhi | 83 / 266 | 6 | 502 / 389 | 9 | yes / yes |
| Kabonyo/Kanyagwal | 1218 / 260 | 25 | 1008 / 412 | 20 | yes / yes |
| Kobura | 92 / 215 | 8 | 124 / 272 | 8 | yes / yes |

Awasi/Onjiko: not estimable in either frame, so no per-ward statement is made for it. Metrics and flags are in MODEL_CARD.md, section "Ward-level slices (Phase K, register rows D52 and D53)".
<!-- phase-k:end -->

<!-- phase-l:begin -->
### 6.2 Land-cover class and clay coverage (Phase L)

Counts of the two frames of `docs/PHASE_L_RESULTS.json`. They are counts only: the samples are case-control, so they are not flood rates.

| land_cover class | Primary: flood / control rows | Primary: events with both classes | Secondary: flood / control rows | Secondary: events with both classes | Enters W (primary / secondary) |
|---|---|---|---|---|---|
| 10 | 313 / 15 | 10 | 269 / 27 | 11 | no / no |
| 20 | 18 / 67 | 7 | 4 / 80 | 3 | no / no |
| 30 | 139 / 856 | 20 | 327 / 1083 | 16 | yes / yes |
| 40 | 139 / 155 | 6 | 678 / 235 | 11 | yes / yes |
| 50 | 0 / 4 | 0 | 0 / 2 | 0 | no / no |
| 60 | 0 / 3 | 0 | 1 / 1 | 1 | no / no |
| 80 | 572 / 16 | 13 | 327 / 11 | 6 | no / no |
| 90 | 412 / 53 | 21 | 364 / 66 | 14 | yes / yes |

| clay_percent | Primary rows | Secondary rows |
|---|---|---|
| blank | 933 | 576 |
| observed | 1829 | 2899 |

Metrics and outcomes are in MODEL_CARD.md, section "Leakage audit of land_cover and clay_percent (Phase L, register rows D14 and D55)".
<!-- phase-l:end -->

## 7. Maintenance
Rebuild with `scripts/build_initial_dataset.py`. Hashes live in `data/MANIFEST.json` and are recomputed by CI (`data-gate`, `manifest-check`). Changes go through pull requests and `docs/ROADMAP_DEVIATIONS.md`.

## river_adjacent_verified (register row D11)

The column is derived from distance_river == 0 and is not independent evidence of a river. OSM comparison, snapshot 2026-10-07T10:50:03Z (docs/D11_OSM_CHECK.json): OSM answered 15 of 15 rows; of those, 0 have a waterway line within 50 m, 1 have another OSM feature within 50 m that is not a waterway (tags in the file), and 14 lie inside an OSM water or wetland area. The column keeps its name because renaming it would change the training file hash.
