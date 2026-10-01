# Phase C protocol: classical model suite

Status: fixed before any Phase C model result exists. Roadmap reference: Section 3 (Phase C) and the Phase A items deferred to it. Any change made after the first result is listed in the deviations log at the end, with its reason.

## 1. What this dataset can support

The training file is data/training/nyando_training_v2_multidate.csv. The numbers in this section are recomputed from it by tests/test_cv_harness.py, so the text cannot drift from the data.

- The file holds 35 events: 23 events with both classes (the only ones with a per-event AUC), 12 events with no flood rows (controls only), and none with floods only.
- Five event-peak sets have exactly 250 flood and 250 control rows each, by design. They hold 2,500 of 4,420 rows (56.6%), so pooled metrics lean on five events, and inside those events the flood share carries no information. The file as a whole is 44.6% flood by design.
- 837 rows (18.9%) sit at a location that appears in more than one event.
- The between-event variance share of `rainfall_3day` is 0.844: rainfall is mostly a property of the event, not of the point.
- The six candidate rainfall columns (rain_aoi_3d, rain_aoi_7d, rain_aoi_14d, rain_basin_3d, rain_basin_7d, rain_basin_14d) are constant within an event (between-event share 1.000 each). With 35 events they cannot be validated, so no Phase C model uses them.
- `clay_percent` is blank on 599 rows, 587 of them in Kabonyo/Kanyagwal. 549 rows (12.4%) sit at the DEM elevation floor, all in the same ward. Both are concentrated in one ward.
- Awasi/Onjiko has 2 flood rows in 521, so it cannot be tested by ward.
- land_cover is nominal. Its classes differ sharply: class 20 (403 rows, flood share 0.010) and class 80 (359 rows, flood share 0.911).

## 2. What Phase C may claim

Inside an event, the terrain features separate flood from control and `rainfall_3day` does not:

| Feature | Declared | Per-event mean AUC | Events agreeing |
|---|---|---|---|
| rainfall_3day | + | 0.523 | 11 of 22 |
| elevation | - | 0.092 | 23 of 23 |
| distance_river | - | 0.123 | 23 of 23 |
| slope | - | 0.276 | 22 of 23 |
| hand | none | 0.194 | 23 of 23 |

A per-event AUC below 0.5 for a feature declared "-" means lower values rank flood above control, as declared. "Events agreeing" counts events where the feature ranks in the declared direction. For hand, which is not constrained, the count uses "-" (lower is wetter) for information.

Consequences:

- The models trained here are terrain susceptibility rankings over a case-control sample.
- The rainfall constraint is a physical prior that this file cannot confirm. The monotonicity test checks that the trained model respects it, not that the data support it.
- Scores are not probabilities, because the sample is balanced by design. No calibration, risk threshold, lead time or operational-alert claim is made.
- Expectations to test, not assume: (H1) removing `rainfall_3day` from the six-feature set changes the selection metric by less than the bootstrap can resolve (the A-Gate's bare 4-feature set contains `rainfall_3day`; the arm without it is the five-feature ablation in Section 4); (H2) the paired per-event differences from adding `rainfall_3day` have an interval that includes 0.

## 3. Models

| Arm | Estimator | Monotone constraints | Note |
|---|---|---|---|
| Baseline | Logistic regression | Not available; coefficient signs are checked and reported | scikit-learn 1.6.1 LogisticRegression has no monotonic_cst parameter |
| Boosting A | HistGradientBoostingClassifier | rainfall_3day +, elevation -, distance_river -, slope - | Replaces the roadmap's GradientBoosting: only this scikit-learn boosting class accepts constraints (checked in Colab on 1.6.1). Early stopping stays off, because an internal validation split would ignore events. |
| Boosting B | XGBoost | the same four | xgboost 3.4.1 in Colab. The production Dockerfile installs no xgboost, so an XGBoost winner cannot load in production until it does. |

`hand` is not a roadmap constraint and stays unconstrained.

## 4. Features and preprocessing

- Primary set, six features: elevation, slope, rainfall_3day, distance_river, clay_percent, land_cover. This is the roadmap's full set and the six inputs the current API accepts.
- Ablations, declared now: the bare 4-feature set (the A-Gate set); the six without `clay_percent`, without `land_cover`, and without `rainfall_3day`; and the six plus `hand`. If the `hand` arm wins under the rules below, adopting it is a separate decision, because it changes the API's input contract.
- `land_cover` is treated as a category (one-hot for logistic, native categorical for the boosters), never as an ordered number.
- `clay_percent` stays blank in the data. The boosters read blanks natively. The logistic baseline uses a median and a missing-value indicator, both fitted on training folds only.
- No SMOTE. The sample is balanced by design, so there is no imbalance to correct, and SMOTE would add synthetic rows.

## 5. Evaluation

Splits, implemented in src/models/cv.py:

1. Leave-one-event-out (LOEO): each of the 35 events is held out in turn.
2. LOEO with shared locations removed (the selection split): training rows at any location present in the held-out event are dropped, so a model cannot score a held-out point by recognising its place.
3. Leave-one-ward-out: the four wards with at least 30 flood and 30 control rows (Ahero, East Kano/Wawidhi, Kabonyo/Kanyagwal, Kobura). Holding out Kabonyo/Kanyagwal is an extrapolation test, because it holds almost all blank `clay_percent` values and every elevation-floor row.

Reported for every arm: pooled AUC, per-event mean AUC over the 23 scorable events, per-ward AUC for the four wards, and the same under the label-shuffle null (labels permuted within each event; expected AUC near 0.5).

Selection rule, fixed now:

- The selection metric is pooled AUC under split 2.
- Primary comparisons, two only: Boosting A and Boosting B, each on the six-feature set, against the baseline on the six-feature set. A challenger beats the baseline only if its paired per-event AUC differences have a 97.5% bootstrap interval entirely above 0 (alpha 0.025 per comparison, two comparisons, 10,000 resamples of events, seed 42) and its selection metric is higher. Wins and losses are reported next to the interval, because 23 events is a coarse sample.
- The winner must also pass the monotonicity test, a model-load check under the production Python and library versions, and the A-Gate on the exact training file.
- Ablation arms (Section 4) are descriptive. An ablation that beats the winner on the selection metric is reported as a finding and triggers a dated protocol revision before anything is adopted; it cannot replace the winner silently.
- If the verdict (the winner, or whether the baseline is beaten) reverses under any sensitivity run below, the result is reported as unresolved, not as a win.
- Pooled AUC can reward a model for predicting how flooded an event is overall, not where. That is why the paired per-event interval is part of the rule above, and why per-event mean AUC is reported beside the pooled value.
- Tuning grids are written into the pull request that adds each model, before any score exists. Tuning uses inner event-grouped cross-validation only.

Sensitivity runs for the winner and the baseline: without the 549 floor rows, without the 15 zero-distance rows, and on the rows where `clay_percent` is present.

## 6. Exit criteria and what stays outside Phase C

Phase C closes when:

- the winner beats the baseline under the rule above;
- the A-Gate and the monotonicity test pass in CI;
- the model-load check passes under Python 3.11, numpy 1.26.4 and scikit-learn 1.6.1 (the production pins);
- every reported number is traceable to an MLflow run with a training-data hash (the retraction guard already requires both for any published score);
- the winner is entered in models/MANIFEST.json under the name rule nyando_<algorithm>_<hash prefix>.

The permutation-importance audit on the six-feature set (a Phase A item) runs on the winner. The live API keeps serving the labelled legacy model until a separate deploy pull request. That pull request must also make the loader compare the model file's SHA-256 with models/MANIFEST.json before unpickling, because pickle loading executes code; in backend/main.py when this protocol was written the file is loaded before its hash is read. The physics-constrained network, advisory, dashboard and alerting belong to later phases.

## 7. Reproducibility

Seed 42 throughout. The training file is read through the manifest, and the harness refuses to run if its SHA-256 or row count differs from data/MANIFEST.json.

## 8. Deviations from the roadmap, decided before any result

1. HistGradientBoostingClassifier replaces scikit-learn's GradientBoostingClassifier (constraints).
2. No SMOTE (balanced by design, synthetic rows).
3. The logistic baseline is not constrained (unsupported); its signs are reported.
4. Primary feature set is six features; `hand` is an ablation (API contract).

Pre-result amendments (r2, after review, while no model result existed): H1 wording corrected; primary comparisons fixed at two with alpha 0.025 each; ablations made descriptive; reversal-under-sensitivity rule added; hash-before-unpickle requirement added for the deploy pull request.

Deviations log after the first result: none yet.
