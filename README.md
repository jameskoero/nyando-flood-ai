# Nyando Flood AI

Ward-level flood susceptibility for five wards of the Nyando River basin, Kisumu County, Kenya, built from open satellite data. **The model is being rebuilt (V2).**

[![CI](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml)
[![data-gate](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/data-gate.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/data-gate.yml)
[![manifest-check](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/manifest-check.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/manifest-check.yml)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)

> **Correction notice (Sept 2026).** The performance figures previously shown in this README (AUC-ROC 0.9717, F1 0.9022, CV AUC 0.9727) are **retracted**. They came from a training file with only 2 flood-labelled rows out of 2,308 (see [CHANGES.md](CHANGES.md)), after SMOTE was applied before the train/test split. The V2 rebuild with independent Copernicus GFM flood labels is live (see Status below). **Do not cite the retracted figures.**

## Contents

[Status](#status) · [What this is](#what-this-is-and-is-not) · [Data](#data-v2) · [Method](#how-the-dataset-was-built) · [Findings](#findings-so-far) · [Limitations](#known-limitations) · [Integrity controls](#integrity-controls) · [Quick start](#quick-start) · [API](#api-legacy-model) · [Layout](#repository-layout) · [Roadmap](#roadmap) · [License](#license-and-data-terms)

## Status

| Area | State |
|---|---|
| Training data (V2) | Built and gated: 4,420 rows from 35 Sentinel-1 scene dates. Passes the A-Gate in CI. |
| Integrity controls | Live: data gate, manifest check, protected `main`, and a separate `live-data` job for the tests that call external services (register row D44). |
| Model | **Registered and served live at `POST /v2/score` (D20 promotion, 2026-10-07).** `hgb:con`, a histogram gradient-boosting model with monotonic constraints trained on the mappable frame, is registered as `models/nyando_hgbcon_5ae81ad8b030.onnx` (ONNX, SHA-256 prefix `5ae81ad8b030`), decided by the gates of the [promotion protocol](docs/PROMOTION_PROTOCOL.md); record: [docs/REGISTRATION.json](docs/REGISTRATION.json). The Phase C model `logistic:con` (`models/nyando_logcon_7a909898d4f6.onnx`) is retired; see the [Phase C protocol](docs/PHASE_C_PROTOCOL.md) and the [closure record](docs/PHASE_C_CLOSURE.md). Scores rank locations inside the areas GFM can map (outside its exclusion mask) and are not flood probabilities. |
| Experiment tracking | Live (Phase B): MLflow runs on DagsHub log the training-data SHA-256, the git commit and the origin (Colab or Termux). The D20 scoring, promotion-gate and registration runs are recorded in `docs/D20_RESULTS.json`, `docs/PROMOTION_RESULTS.json` and `docs/REGISTRATION.json`. |
| Phase D (physics-constrained MLP) | Stage 0 only: the protocol and its rules as code are committed (`docs/PHASE_D_PROTOCOL.md`, `src/models/phase_d.py`, register row D46). No MLP is trained, and Block B stays unbuilt and unscored until the Phase D freeze file is merged. |
| Dashboard | Live and labelled demonstration only: the retracted metrics were removed (`tests/test_frontend_claims.py` guards this) and every score carries a not-validated notice. It still shows legacy-model output; the live metrics panel, loading and error states are Phase F. |
| Early warning (SMS, forecasts) | Not built. Phase G stays in sandbox and shadow mode while the service stays on Render's free tier (register row D45); no public alerts. |

## What this is and is not

**Is:** a reproducible, tested pipeline that turns open satellite products into a labelled flood dataset for the Nyando wards, plus the tooling that keeps that dataset honest.

**Is not (yet):** a forecast or an operational warning system. No lead time is claimed. The registered model ranks locations and is not calibrated to probabilities; calibration comes with the risk-bucket work in Phase E.

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
- **Registered model.** `hgb:con`, mean per-event AUC on the mappable frame: 0.943 over the 23 existing events and 0.971 over the 27 new D20 events (full frame: 0.955 and 0.965). Against `logistic:con`, pooled over the 50 events, the paired difference is +0.0138 [0.0036, 0.0230] (mappable) and +0.0164 [0.0081, 0.0245] (full). No advantage is shown under the 1 km buffered split (-0.0088 [-0.0296, 0.0091]) or out of time (+0.0039 [-0.0130, 0.0195]). Scores rank locations and are not flood probabilities (`docs/REGISTRATION.json`).

## Known limitations

- 549 rows (12.4%) sit at the DEM floor (1130.5 m) in Kabonyo/Kanyagwal, all with HAND 0, and 96% of them are flooded.
- Blank clay is informative: it is blank on 28% of flood cases but only 1.8% of controls. Models must be tested with and without it.
- Awasi/Onjiko has only 2 flood cases. Ward-level fairness checks cannot cover it.
- A label means inundation on the scene date, not newly arrived flooding.
- Rainfall windows are date-level features, so the 35 dates give only 35 distinct values.
- Some dates contribute a single flood patch; weight by patch when evaluating.
- GFM cannot map everywhere: 945 of the 2,450 controls (38.6%) lie inside its exclusion mask, where flood pixels are reset to no flood (`data/derived/gfm_layer_flags.csv`). Scores say nothing about locations inside that mask or about floods GFM cannot detect.
- The registered model `hgb:con` beats `logistic:con` on the new D20 events and pooled, but no advantage is shown under the 1 km buffered split or out of time (`MODEL_CARD.md`); the evidence is in `docs/PROMOTION_RESULTS.json`.
- `river_adjacent_verified` is derived from `distance_river == 0` and is not evidence of a river. The OSM check (`docs/D11_OSM_CHECK.json`) found 14 of the 15 zero-distance rows inside OSM water or wetland areas and 0 with an OSM waterway line within 50 m (register row D11).
- Cross-validation of the bare 4-feature logistic baseline, leave-one-event-out (recomputed by `tests/test_v2_audit.py`): pooled AUC 0.841 (all held-out predictions together; the A-Gate limit of 0.90 applies to this figure), mean per-event AUC 0.910 (mean over held-out events that contain both classes), and pooled AUC 0.826 when training also excludes every location that appears in the held-out event. The location-excluded figure is the more conservative estimate of generalisation.

## Integrity controls

- **A-Gate** (`tests/test_data_gate.py`): row count and flood rate, real dates matching scene ids, GFM provenance, points inside the wards, no unexpected blanks, the 4-feature logistic-regression check, no single feature separating the classes (AUC below 0.90), documented reasons for repeated values, and a consistent river flag. The original "clay flat between classes" rule is kept as an expected-failure report: labels never use clay, so a class difference reflects floodplain soils.
- **Manifests** (`data/MANIFEST.json`, `models/MANIFEST.json`, checked by `scripts/check_manifests.py`): SHA-256 recomputed independently, models named by algorithm and hash prefix, no two binaries sharing a file name. Pre-V2 files are recorded as `legacy`.
- **Protected `main`:** pull request required, and `test`, `data-gate` and `manifest-check` must pass. No bypass, no force-push.
- **CI hygiene:** every workflow declares read-only token permissions and pins each action to a commit SHA, and Dependabot keeps those pins current. The gate workflows use no secrets and run on every pull request; the gate installs exact versions (`requirements-gate.txt`) and the test job installs under exact version constraints (`constraints-ci.txt`), both with a retried install. Fork and Dependabot pull requests have no Actions secrets, so the tests that need an Earth Engine session are skipped there with a visible notice; a trusted run without the key fails with a clear message, and runs with the key execute everything.
- **Live-data tests:** tests that call the live EODC GFM catalogue run in a separate `live-data` job that is not a required check, so a slow third-party service cannot block merges; the required `test` job runs the rest (register row D44).
- **Sealed test block:** Block B of the D20 selection stays unbuilt and unscored; `tests/test_phase_d.py` fails if a Block B file appears before the Phase D freeze file is merged.
- **Not built yet:** the scheduled `train-check` and `drift-monitor` workflows and a gated `deploy` workflow (register row D13).

## Quick start

Verify the data (no Earth Engine account needed):

```bash
git clone https://github.com/jameskoero/nyando-flood-ai.git
cd nyando-flood-ai
python -m pip install -r requirements-gate.txt
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

The deployed service serves the registered V2 model at `POST /v2/score` and still serves the legacy model at `POST /predict`. Interactive docs (`/docs`, `/redoc`) are disabled in production (checked live: both return 404). The registered V2 model is served at `POST /v2/score` by this code; it loads only after its SHA-256 matches `models/MANIFEST.json`, and the live service reports whether it is loaded in `GET /health`.

- `POST /v2/score` scores one location with the registered model and returns `score`, a ranking score that is not a flood probability, with the model's hash and claim limit. It has no risk classes, because none were validated. `clay_percent` may be omitted (the response then carries a warning).
- `GET /v2/metrics` returns the stored evaluation of the registered model: `docs/REGISTRATION.json` when it names the loaded model, otherwise `docs/PHASE_C_RESULTS.json`; 503 when neither is available.
- `GET /health` reports both models: the legacy model's file name, SHA-256 and provenance, and `registered_model` (loaded or not, file, SHA-256, training-data hash).
- `GET /metrics` returns validated metrics, or HTTP 503 when none are published for the served model.
- `POST /predict` is rate-limited. The example below shows the shape only; the values are illustrative.

```json
{"elevation": 1142.5, "slope": 2.3, "rainfall_3day": 87.4, "distance_river": 320.0,
 "clay_percent": 42.1, "land_cover": 1, "ward": "Ahero"}
```

```json
{"flood_probability": 0.87, "risk_score": 0.87, "risk_class": "CRITICAL", "prediction": 1,
 "ward": "Ahero", "model_version": "1.0.0",
 "notice": "Legacy model, not validated. Do not use for safety decisions."}
```

Legacy risk classes (from the code): LOW below 0.35, MEDIUM 0.35 to 0.60, HIGH 0.60 to 0.80, CRITICAL 0.80 and above. The score is a legacy-model output, not a validated flood probability.

The API runs on Render's free tier, so the first request after idle can take about a minute. It stays on the free tier by the owner's decision (register row D45). Live services: [dashboard](https://nyando-flood-ai.vercel.app) and [API health](https://nyando-flood-api.onrender.com/health).

## Repository layout

```text
data/MANIFEST.json                       SHA-256 and provenance for every data file
data/DATA_SOURCES.md                     where each data source comes from
data/external/                           ward boundaries (nyando_wards.geojson) and source notes
data/confirmatory/                       D20 Block A: the built confirmatory dates (no label_source; see docs/D20_PROTOCOL.md)
data/training/                           V2 training data and its per-date sets table; the v1 files are legacy and discredited
data/derived/                            gfm_layer_flags.csv: GFM exclusion-mask and reference-water flags for each training point
models/MANIFEST.json                     SHA-256, status and training-data hash for every model file
models/PROVENANCE.json                   provenance notes for the model files
models/nyando_hgbcon_5ae81ad8b030.onnx   the registered model (hgb:con, ONNX TreeEnsemble), D20 promotion
models/nyando_logcon_7a909898d4f6.onnx   the Phase C model (logistic:con, ONNX), retired
models/nyando_xgb_v1.pkl                 pre-V2 model file, recorded as legacy
src/data/                                GFM client, terrain, case-control sampler, raw features, audit and validation helpers
src/features/                            build_features.py
src/models/                              Phase C modules: cv (evaluation harness), baseline, boosters, constrained, decision, registration, export_onnx, robustness (battery), closure (record), promotion (gates), export_hgb (tree exporter), phase_d (Phase D rules); also train_model.py and evaluate_model.py
src/utils/                               geo_utils.py
src/visualization/                       shap_plots.py
src/tracking.py                          MLflow tracking wrapper
scripts/                                 dataset build, manifest check, GFM layer audit, Phase C, D20, promotion and registration runs, export and results scripts, D11 OSM check
tests/                                   A-Gate, README checks, evaluation harness, boosters, constrained logistic, registration, export, robustness and legacy tests
docs/                                    protocols, results and records: Phase C, D20, promotion and Phase D protocols, registration record, D11 OSM check, deviation register, hardening audit
docs/funding/                            superseded concept note
backend/                                 FastAPI service: legacy model on /predict, registered ONNX model on /v2/score (backend/models/ holds the legacy model files)
frontend/                                React dashboard
notebooks/                               v1 notebooks (legacy)
.github/workflows/                       ci.yml, data-gate.yml, manifest-check.yml

README.md  CHANGES.md                    this file; dated change log
CONTRIBUTING.md  LICENSE  SECURITY.md    contribution rules, license, security policy
MODEL_CARD.md                            model card
Dockerfile  docker-compose.yml           API image (python:3.11-slim, pinned versions) and a compose file
vercel.json                              Vercel settings for the dashboard
requirements.txt                         project and CI dependencies
requirements-gate.txt                    exact pins for the data-gate job
requirements-onnx.txt                    ONNX export and scoring dependencies, installed by CI
constraints-ci.txt                       exact versions of every package the CI test job can install (pip constraints)
pytest.ini  conftest.py                  pytest configuration
gee_extract_nyando.py                    v1 Earth Engine extraction script (legacy; used by tests/test_gfm_client.py)
```

## Roadmap

- [x] **0. Hotfix (part):** `GET /health` reports the loaded models' SHA-256, `/docs` and `/redoc` return 404 on the live service, and `GET /metrics` answers 503 when no validated metrics are published.
- [ ] **0. Hotfix (remainder):** the per-client rate limit on `/predict` does not hold; only the shared cap does (`docs/PHASE_CLOSURE.md`; cause not established).
- [x] **A. Data integrity:** GFM-labelled dataset, A-Gate in CI, manifests, protected `main`.
- [x] A (remainder). The OSM check of the 15 zero-distance rows is recorded (`docs/D11_OSM_CHECK.json`, register row D11). The monotonicity test is in CI; the permutation audit and the temporal holdout ran in Phase C (see the closure record), and leakage in `land_cover` is not excluded.
- [x] **B. Experiment tracking:** MLflow on DagsHub, with the data hash logged on every run.
- [x] **C. Model suite:** logistic regression, gradient boosting and XGBoost with monotonic constraints, evaluated leave-one-event-out. Closed with the registered model `logistic:con` (replaced below); see the [closure record](docs/PHASE_C_CLOSURE.md) for what the boosters did and did not show. D20 then tested these results on new events ([D20 protocol](docs/D20_PROTOCOL.md), Section 13): the advantage over elevation only did not replicate and both boosters beat `logistic:con`; the [promotion protocol](docs/PROMOTION_PROTOCOL.md) then registered `hgb:con` ("D20 PROMOTED", record: [docs/REGISTRATION.json](docs/REGISTRATION.json)).
- [x] **Gates:** `data-gate`, `manifest-check` and `test` are required checks on protected `main`.
- [ ] **Gates (remainder):** scheduled `train-check`, `drift-monitor` and a gated `deploy` workflow (register row D13).
- [x] **D. Stage 0:** the physics-constrained MLP protocol and its rules as code are committed ([protocol](docs/PHASE_D_PROTOCOL.md), register row D46); nothing is trained.
- [ ] **D. (remainder):** the MLP itself: code, Colab training, ONNX export and one decision on Block B. Serving through ONNX Runtime is already done (register row D33).
- [ ] **E. LLM advisory layer**, cached and rate-limited, never on the `/predict` path.
- [ ] **F. Dashboard:** no hard-coded metrics, loading and error states, accessibility.
- [ ] **G-H. SMS alerts** (only after a public WARMA feed is confirmed; sandbox and shadow mode only while the service stays on Render's free tier, register row D45) and an MCP/LangGraph showcase.
- [x] **K (part):** the model card and the datasheet (`data/DATA_SOURCES.md`) are rewritten to V2 (register row D38).
- [ ] **J-K (remainder):** ward-level fairness slices, the paper, and a Zenodo erratum before any new numbers are cited.

## License and data terms

Code: MIT ([LICENSE](LICENSE)). The training data is derived from third-party products that keep their own terms. `elevation`, `slope` and `hand` come from MERIT/Hydro, which is dual-licensed CC BY-NC 4.0 or ODbL 1.0, so the dataset **cannot be published under CC BY 4.0**. A dataset license is still to be chosen (ODbL 1.0 is the natural fit for an open project); until then, treat the data as non-commercial and check each source's terms.

## Citation, security and contributing

The Zenodo record [10.5281/zenodo.20088663](https://doi.org/10.5281/zenodo.20088663) predates the correction and contains the retracted figures; an erratum is planned before any new numbers are cited. Report vulnerabilities as described in [SECURITY.md](SECURITY.md); see [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Author

**James Koero**, ML engineer, Kisumu, Kenya. [GitHub](https://github.com/jameskoero) · [LinkedIn](https://linkedin.com/in/jameskoero)

Academic advisors: Prof. Samuel Liyala (JOOUST, Kenya) and Prof. Johan Loeckx (Vrije Universiteit Brussel, VUB AI Lab, Belgium).
