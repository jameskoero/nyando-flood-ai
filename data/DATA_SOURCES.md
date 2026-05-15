# Data Sources — Nyando Flood AI (Real GEE Data)

**All sources: open, non-personal, CC-BY-4.0 / Public Domain. Kenya DPA 2019 + GDPR compliant.**

| Feature | Source | GEE ID | License |
|---|---|---|---|
| elevation, slope | NASA NASADEM | NASA/NASADEM_HGT/001 | Public Domain |
| rainfall_3day | CHIRPS v2 (UCSB) | UCSB-CHG/CHIRPS/DAILY | CC-BY-4.0 |
| flooded (labels) | Sentinel-1 SAR | COPERNICUS/S1_GRD | Free Copernicus |
| clay_percent | ISRIC/OpenLandMap | OpenLandMap/SOL/... | CC-BY-4.0 |
| distance_river | HydroSHEDS/OSM | WWF/HydroSHEDS/v1/... | ODbL |
| land_cover | ESA WorldCover 2021 | ESA/WorldCover/v200 | CC-BY-4.0 |
| population | WorldPop Kenya 2020 | WorldPop/GP/100m/pop | CC-BY-4.0 |

## Real Data Statistics (from GEE extraction)
- Observations: 2,308 real satellite points
- Bounds: lon 34.70°E–35.40°E, lat 0.40°S–0.10°N
- Elevation: 1,131m – 2,588m (real NASADEM values)
- 3-Day Rainfall: 81.8mm – 162.3mm (real CHIRPS April 2024)
- SAR flood confirmed: 2 pixels (Sentinel-1 April 2024)

## Independent Validation
Awino & Machanda (2024) arXiv:2512.13710 — used same basin, same features, confirms methodology.
