DATA SOURCES — All open, free, non-personal, satellite/census-derived
=====================================================================

1. CHIRPS v2 Rainfall
   Source  : Climate Hazards Center, UCSB
   GEE ID  : UCSB-CHG/CHIRPS/DAILY
   Direct  : https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/
   License : CC-BY-4.0 (free, no registration)
   Variable: 3-day accumulated rainfall (mm)

2. NASA NASADEM Elevation
   Source  : NASA Jet Propulsion Laboratory
   GEE ID  : NASA/NASADEM_HGT/001
   Direct  : https://earthdata.nasa.gov (free, NASA EarthData account)
   License : Public Domain
   Variable: Elevation (m), slope (degrees)

3. Sentinel-1 SAR Flood Labels
   Source  : ESA Copernicus / Google Earth Engine
   GEE ID  : COPERNICUS/S1_GRD
   License : Free (Copernicus open access)
   Variable: flooded = 1 (VV < -16 dB during flood season)
   Alternative: UNOSAT Kenya flood maps → https://unosat.org/products/

4. ISRIC SoilGrids Clay Fraction
   Source  : ISRIC World Soil Information
   GEE ID  : projects/soilgrids-isric/clay_mean
   Direct  : https://rest.isric.org/soilgrids/v2.0/properties/query
   License : CC-BY-4.0
   Variable: clay_0-5cm_mean (%), 250m resolution

5. OpenStreetMap River Network
   Source  : OpenStreetMap Foundation
   GEE ID  : WWF/HydroSHEDS/v1/FreeFlowingRivers
   Direct  : https://overpass-api.de/api/interpreter
   License : ODbL (Open Database License)
   Variable: distance_river (m) — Euclidean distance to nearest waterway

6. ESA WorldCover 2021
   Source  : ESA / Sinergise / University of Ghent / VITO
   GEE ID  : ESA/WorldCover/v200
   Direct  : https://esa-worldcover.org/en/data-access
   License : CC-BY-4.0
   Variable: land_cover (10=Trees, 20=Shrubs, 30=Grassland, 40=Cropland,
              50=Urban, 80=Water, 90=Wetland)

7. WorldPop Population 2020
   Source  : WorldPop, University of Southampton
   GEE ID  : WorldPop/GP/100m/pop
   Direct  : https://hub.worldpop.org/geodata/
   License : CC-BY-4.0
   Variable: population (persons per 100m pixel)

PRIVACY & ETHICS
================
ALL sources are satellite-derived or aggregated census data.
ZERO personal or household-level data is collected or processed.
This dataset is fully compliant with:
  - Kenya Data Protection Act 2019
  - GDPR (no EU personal data involved)
  - CC-BY-4.0 output license

Published paper on this exact watershed:
  Awino E.O. & Machanda D. (2024). Predictive Modeling of Flood-Prone Areas
  Using SAR and Environmental Variables — River Nyando Watershed, Kenya.
  arXiv:2512.13710. [Supports validity of this methodology]