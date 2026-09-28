# Nyando Flood AI

Ward-level flood susceptibility for five wards of the Nyando River basin, Kisumu County, Kenya, built from open satellite data. **The model is being rebuilt (V2).**

[![CI](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml)
[![data-gate](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/data-gate.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/data-gate.yml)
[![manifest-check](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/manifest-check.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/manifest-check.yml)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)

> **Correction notice (Sept 2026).** The performance figures previously shown in this README (AUC-ROC 0.9717, F1 0.9022, CV AUC 0.9727) are **retracted**. They came from a training file with only 2 flood-labelled rows out of 2,308 (see [CHANGES.md](CHANGES.md)), after SMOTE was applied before the train/test split. A V2 rebuild with independent Copernicus GFM flood labels is in progress. **Do not cite the retracted figures.**

## Contents

[Status](#status) · [What this is](#what-this-is-and-is-not) · [Data](#data-v2) · [Method](#how-the-dataset-was-built) · [Findings](#findings-so-far) · [Limitations](#known-limitations) · [Integrity controls](#integrity-controls) · [Quick start](#quick-start) · [API](#api-legacy-model) · [Layout](#repository-layout) · [Roadmap](#roadmap) · [License](#license-and-data-terms)

## Status

| Area | State |
|---|---|
| Training data (V2) | Built and gated: 4,420 rows from 35 Sentinel-1 scene dates. Passes the A-Gate in CI. |
| Integrity controls | Live: data gate, manifest check, protected `main`. |
| Model | **Not retrained yet.** The deployed API still serves a pre-V2 model whose metrics are unverified (Phase C). |
| Experiment tracking | Not started (Phase B). |
| Dashboard | Live, but it still displays a retracted metric that is hard-coded in the front end (fix planned in Phase F). |
| Early warning (SMS, forecasts) | Not built. |

## What this is and is not

**Is:** a reproducible, tested pipeline that turns open satellite products into a labelled flood dataset for the Nyando wards, plus the tooling that keeps that dataset honest.

**Is not (yet):** a forecast or an operational warning system. No lead time is claimed. The trained model, its metrics and its calibration come after this data work.

## Data (V2)

Labels come from the **Copernicus Global Flood Monitoring (GFM)** ensemble flood extent (Sentinel-1): inundation on the scene date, outside permanent water. Labels are never derived from the features.

| Column | Source | Notes |
|---|---|---|
| `flooded` | Copernicus GFM `ensemble_flood_extent` | 1 = flooded pixel on that scene date |
| `elevation`, `slope`, `hand` | MERIT/Hydro v1.0.1 (90 m) | HAND = height above nearest drainage |
| `distance_river` | JRC Global Surface Water occurrence | distance to a water-mask pixel, in metres |
| `rainfall_3day` | CHIRPS daily | 3-day sum at the point (about 5.5 km pixels) |
| `rain_aoi_*`, `rain_basin_*` | CHIRPS daily | mean over the wards and over the upstream basin (HydroBASINS level 8, 9 sub-basins, 4,006 km2), 3, 7 and 14 days |
| `clay_percent` | ISRIC SoilGrids, 0-5 cm | left **blank** where SoilGrids has no value; never filled |
| `land_cover` | ESA WorldCover v100 (2020) | categorical |

Every row also carries its GFM scene id, the real Sentinel-1 acquisition date, the ward and a `sample_type`.

| Dataset at a glance | |
|---|---|
| Rows | 4,420 |
| Scene dates (sample sets) | 35 (5 flood events + 30 screened dates) |
| Flood cases / controls | 1,970 / 2,450 |
| Rows with blank clay | 599 |

The flood share of the file (44.6%) is set by the sampling design and is not the real flood rate.

## How the dataset was built

1. **Case-control sampling per scene.** Flood pixels are rare (0.07% to 1.3% of valid pixels per scene), so each scene contributes flood and non-flood points: 250 of each for the five event scenes, up to 40 of each for the other dates.
2. **Events:** April 2020, April 2021, May 2022, April/May 2024 and March 2026.
3. **30 extra dates chosen by rule, not by hand.** Sentinel-1 scene dates from 2020 to 2026 were screened for full ward coverage (23 of 42 candidates in the last screen had no usable data over the wards) and spread across six equal-count bins of upstream-basin 14-day rainfall, at least 14 days apart, with a fixed seed.
4. **Real dates.** `event_date` is the Sentinel-1 scene date that GFM returned, and CI checks it against the scene id on every row.
5. **No imputation.** Points missing a clay value are kept with clay blank. Every other missing value drops the point, and drops are logged.
6. **Reproducible.** `scripts/build_initial_dataset.py` re-derives the earlier 2,264 event points identically.

## Findings so far

- **Rainfall separates dates.** Across the 30 non-event dates (chosen by coverage and rainfall, never by flood outcome), 8 of 8 dates with upstream-basin 14-day rainfall of 90 mm or more had flood pixels, against 10 of 22 below 90 mm (Fisher exact one-sided p = 0.0075).
- **Point rainfall is weak on its own.** `rainfall_3day` alone separates the classes at AUC 0.55, while `elevation` alone reaches 0.82.
- **Not circular.** A bare 4-feature logistic regression (elevation, slope, rainfall, distance) scores 0.841 leave-one-event-out, under the 0.90 gate.
- **Flood points cluster.** The 1,970 flood cases form about 1,087 separate patches (points within 60 m linked), so evaluation has to be leave-one-event-out and patch-aware.

## Known limitations

- 549 rows (12.4%) sit at the DEM floor (1130.5 m) in Kabonyo/Kanyagwal, all with HAND 0, and 96% of them are flooded.
- Blank clay is informative: it is blank on 28% of flood cases but only 1.8% of controls. Models must be tested with and without it.
- Awasi/Onjiko has only 2 flood cases. Ward-level fairness checks cannot cover it.
- A label means inundation on the scene date, not newly arrived flooding.
- Rainfall windows are date-level features, so the 35 dates give only 35 distinct values.
- Some dates contribute a single flood patch; weight by patch when evaluating.

## Integrity controls

- **A-Gate** (`tests/test_data_gate.py`): row count and flood rate, real dates matching scene ids, GFM provenance, points inside the wards, no unexpected blanks, the 4-feature logistic-regression check, no single feature separating the classes (AUC below 0.90), documented reasons for repeated values, and a consistent river flag. The original "clay flat between classes" rule is kept as an expected-failure report: labels never use clay, so a class difference reflects floodplain soils.
- **Manifests** (`data/MANIFEST.json`, `models/MANIFEST.json`, checked by `scripts/check_manifests.py`): SHA-256 recomputed independently, models named by algorithm and hash prefix, no two binaries sharing a file name. Pre-V2 files are recorded as `legacy`.
- **Protected `main`:** pull request required, and `test`, `data-gate` and `manifest-check` must pass. No bypass, no force-push.
- **CI hygiene:** the gate workflows are read-only, use no secrets and run on every pull request.

## Quick start

Verify the data (no Earth Engine account needed):

```bash
git clone https://github.com/jameskoero/nyando-flood-ai.git
cd nyando-flood-ai
python -m pip install pandas numpy scipy scikit-learn pytest
python -m pytest tests/test_data_gate.py -q --noconftest -rx
python scripts/check_manifests.py
```

Expected: `13 passed, 1 xfailed`, then `manifest-check OK`.

Rebuild the dataset (needs an Earth Engine account and a Cloud project). The script assumes Google Colab paths (`/content/nyando-flood-ai`) and the project name `nyando-flood-ai`; both are constants at the top of `scripts/build_initial_dataset.py`. In a Colab notebook, authenticate in a cell with `ee.Authenticate()`, then run:

```python
%run scripts/build_initial_dataset.py
```

It resumes after an interruption and takes about 10 minutes. The full test suite (`pytest tests/`) also needs Earth Engine credentials.

## API (legacy model)

The deployed service predates V2 and serves a legacy model. Interactive docs are disabled in production.

- `GET /health` reports the loaded model's file name, SHA-256 and provenance.
- `GET /metrics` returns the metrics endpoint output.
- `POST /predict` is rate-limited. The example below shows the shape only; the values are illustrative.

```json
{"elevation": 1142.5, "slope": 2.3, "rainfall_3day": 87.4, "distance_river": 320.0,
 "clay_percent": 42.1, "land_cover": 1, "ward": "Ahero"}
```

```json
{"risk_score": 0.87, "risk_class": "HIGH", "risk_label": "Prepare evacuation routes",
 "ward": "Ahero", "model_version": "1.0.0"}
```

Legacy risk classes: LOW below 0.35, MEDIUM 0.35 to 0.65, HIGH above 0.65.

The API runs on Render's free tier, so the first request after idle can take about a minute. Live services: [dashboard](https://nyando-flood-ai.vercel.app) and [API health](https://nyando-flood-ai.onrender.com/health).

## Repository layout

```text
data/
  MANIFEST.json                            SHA-256 and provenance for every data file
  external/nyando_wards.geojson            ward boundaries
  training/
    nyando_training_v2_multidate.csv       V2 training data
    nyando_training_v2_multidate_sets.csv  one row per scene date
    nyando_training_v1*.csv                legacy, discredited
models/
  MANIFEST.json                            pre-V2 model files, recorded as legacy
scripts/
  build_initial_dataset.py                 rebuilds the V2 dataset (needs Earth Engine)
  check_manifests.py                       independent manifest verification
src/data/                                  GFM client, terrain, sampler, raw features
tests/                                     A-Gate, sampler regression, README checks, legacy tests
backend/                                   FastAPI service (legacy model)
frontend/                                  React dashboard
notebooks/                                 v1 notebooks (legacy)
.github/workflows/                         ci.yml, data-gate.yml, manifest-check.yml
```

## Roadmap

- [x] **A. Data integrity:** GFM-labelled dataset, A-Gate in CI, manifests, protected `main`.
- [ ] A (remainder). Monotonicity test and leakage audit; both need a trained model, so they land with Phase C.
- [ ] **B. Experiment tracking:** MLflow on DagsHub, with the data hash logged on every run.
- [ ] **C. Model suite:** logistic regression, gradient boosting and XGBoost with monotonic constraints, evaluated leave-one-event-out.
- [ ] **D. Physics-constrained MLP**, exported to ONNX.
- [ ] **E. LLM advisory layer**, cached and rate-limited, never on the `/predict` path.
- [ ] **F. Dashboard:** no hard-coded metrics, loading and error states, accessibility.
- [ ] **G-H. SMS alerts** (only after a public WARMA feed is confirmed) and an MCP/LangGraph showcase.
- [ ] **J-K. Model card, datasheet, paper**, and a Zenodo erratum before any new numbers are cited.

## License and data terms

Code: MIT ([LICENSE](LICENSE)). The training data is derived from third-party products that keep their own terms. `elevation`, `slope` and `hand` come from MERIT/Hydro, which is dual-licensed CC BY-NC 4.0 or ODbL 1.0, so the dataset **cannot be published under CC BY 4.0**. A dataset license is still to be chosen (ODbL 1.0 is the natural fit for an open project); until then, treat the data as non-commercial and check each source's terms.

## Citation, security and contributing

The Zenodo record [10.5281/zenodo.20088663](https://doi.org/10.5281/zenodo.20088663) predates the correction and contains the retracted figures; an erratum is planned before any new numbers are cited. Report vulnerabilities as described in [SECURITY.md](SECURITY.md); see [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Author

**James Koero**, ML engineer, Kisumu, Kenya. [GitHub](https://github.com/jameskoero) · [LinkedIn](https://linkedin.com/in/jameskoero)

Academic advisors: Prof. Samuel Liyala (JOOUST, Kenya) and Prof. Johan Loeckx (Vrije Universiteit Brussel, VUB AI Lab, Belgium).
