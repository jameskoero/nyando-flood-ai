# Phase K protocol: ward-level fairness slicing, Model Card and Datasheet

This protocol is fixed before any ward-level metric exists. Its rules are code in `src/models/phase_k.py` and are checked by `tests/test_phase_k.py`.

## 1. Scope and claim limit

Phase K reports how the registered model (hgb:con, `models/nyando_hgbcon_5ae81ad8b030.onnx`) ranks locations inside each of the five Nyando wards. It tests nothing about causes, takes no promotion decision and adds no claim beyond the E1 claim: a ranking score, not a flood probability. Per-ward numbers are descriptive. The training and Block A samples are case-control, so a ward's share of flood rows is a property of the sampling and not a flood rate.

## 2. Frames

Both frames are frozen from E1 (`docs/PHASE_E_PROTOCOL.md`, `docs/PHASE_E_RESULTS.json`).

- **Primary:** the Block A rows kept after the exclusion of rows at training locations (E1 amendment r1), scored by the registered artifact, with the bands of the E1 cutpoints. The primary frame decides what the Model Card may say.
- **Secondary:** the out-of-fold scores of hgb:con at the registered parameters on the mappable training rows, leave-one-event-out with shared locations removed, with the cutpoints recomputed as E1 does. It is reported for context. The run recomputes the E1 reference and checks that it reproduces E1 before any slicing, as E1 does.

## 3. Slices and estimability

The slices are the five wards of `data/external/nyando_wards.geojson`. A ward is estimable in a frame only if it has at least MIN_ROWS_PER_CLASS = 30 flood rows, at least 30 control rows and at least MIN_EVENTS_BOTH = 5 events that contain both classes in that frame. Otherwise it is not estimable: no metric, no interval, no claim, and it is never imputed, pooled or merged with another ward.

Event-clustered intervals need several events that contain both classes, and 30 rows per class is a conventional floor for a rate. The thresholds are conventions. They were chosen after the per-ward row counts of the training file below were seen and before any ward-level metric was computed, and they are not tuned.

All rows of the V2 training file (descriptive counts, checked by `tests/test_phase_k.py`):

| Ward | Control rows | Flood rows | Events with controls | Events with floods |
|---|---|---|---|---|
| Ahero | 302 | 334 | 34 | 14 |
| Awasi/Onjiko | 519 | 2 | 35 | 1 |
| East Kano/Wawidhi | 675 | 502 | 35 | 9 |
| Kabonyo/Kanyagwal | 568 | 1008 | 35 | 20 |
| Kobura | 386 | 124 | 35 | 8 |

Awasi/Onjiko has 2 flood rows in 1 event in the whole file, so it cannot be estimable in any subset of it.

## 4. Metrics per estimable ward and frame

- **M1, level rates:** rows, flood rows and flood share in each band (LOW, MEDIUM, HIGH), with `rules.level_rates`.
- **M2, band separation:** HIGH minus LOW flood share, with the event-clustered bootstrap `rules.cluster_bootstrap_diff`.
- **M3, ranking inside the ward:** the mean over events that contain both classes of the per-event AUC, with an event-clustered bootstrap interval.
- **M4, gap to the pooled value:** the ward's M3 minus the M3 of all wards of the same frame, with its interval.

## 5. Intervals

Every interval is an event-clustered percentile bootstrap that uses N_BOOT, SEED and ALPHA from `src/advisory/rules.py`, the E1 convention. They are imported, not re-implemented.

## 6. Flags and statements

The flags are descriptive. There is no gate and no promotion.

- `not_estimable`: the rule of Section 3 fails.
- `bands_not_separated`: the lower end of the M2 interval is at or below 0.
- `auc_not_above_chance`: the lower end of the M3 interval is at or below 0.5.
- `differs_from_pooled`: the M4 interval excludes 0.

Every flag raised is reported in the Model Card and the Datasheet. A ward that is not estimable is stated as "no claim possible". The outcome string is PHASE K REPORTED.

## 7. What this does not show

No causal reading. No real flood prevalence. Results hold inside the GFM-mappable areas only. The DEM-floor rows of Kabonyo/Kanyagwal and the land-cover question (register row D14) are not separated here. Lead time and early-warning skill are not tested, so no early-warning claim follows.

## 8. Outputs

`docs/PHASE_K_RESULTS.json` (schema 1: protocol SHA-256, commit, model SHA-256, per-frame and per-ward counts, estimability, metrics and flags), one MLflow run read back, then the Model Card (`docs/MODEL_CARD.md`) and the Datasheet (`data/DATA_SOURCES.md`) sections that cite the results by name.

## 9. Deviations and disclosure

Register row D52. The thresholds were chosen after the per-ward counts were seen and before any ward metric. The frames follow E1.
