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

## 9. Findings from the baseline run (2026-10-02)

Source: MLflow run 88e8f6a33fd540b89f4801a8f7fa50b7, logged from commit 95251ef0e897e03c2726636fe4ef937e6b42fdb3 on Colab, training file hash 0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a. Values in the first table are read back from that run. The reference arms, the land-cover lookup, the elevation AUC per land-cover class, the row counts in this section, and the six-feature selection-split and null values are recomputed from the data by tests/test_phase_c_amendment.py. The remaining values come from the read-only audit of 2026-10-02 and are not re-tested.

Logistic arms, selection split (shared locations removed) and its single-seed shuffle null:

| Arm | Pooled AUC | Per-event mean | Null pooled | Null per-event mean |
|---|---|---|---|---|
| bare4 | 0.8264 | 0.9104 | 0.4607 | 0.5010 |
| six | 0.8988 | 0.9391 | 0.4573 | 0.5053 |
| seven | 0.8989 | 0.9399 | 0.4555 | 0.5086 |
| six_no_clay | 0.8861 | 0.9380 | 0.4565 | 0.5121 |
| six_no_land_cover | 0.8361 | 0.9108 | 0.4613 | 0.4995 |
| six_no_rainfall | 0.8907 | 0.9353 | 0.4288 | 0.5052 |

Reference arms (no fitting; a lower raw value ranks more flood), pooled AUC and per-event mean over the 23 scorable events:

| Reference | Pooled AUC | Per-event mean |
|---|---|---|
| elevation | 0.8239 | 0.9077 |
| distance_river | 0.7988 | 0.8771 |
| hand | 0.7871 | 0.8061 |
| slope | 0.7276 | 0.7243 |
| land-cover class lookup (leave-one-event-out, shared locations removed) | 0.7844 | 0.8687 |

Observations:

- The six-feature logistic exceeds elevation alone by 0.0314 in per-event mean (0.9391 against 0.9077). Without land_cover the arm scores 0.9108, so land_cover accounts for 0.0283 of that margin; clay_percent accounts for 0.0011 and rainfall_3day for 0.0038. The bare4 arm is 0.0027 above elevation alone.
- Inside land-cover classes, pooled across events, the AUC of raw elevation is 0.930 (class 10), 0.814 (class 30), 0.453 (class 40), 0.594 (class 80) and 0.776 (class 90). For distance_river it is 0.934, 0.757, 0.539, 0.380 and 0.780. Classes 20 (4 floods), 50 (0 floods) and 60 (1 flood) cannot be scored.
- Outside classes 10, 80 and 90 (3,195 rows, 1,010 floods, 17 events with both classes), adding land_cover raises per-event mean AUC from 0.8632 to 0.8893 when the model is trained on all rows and scored outside, and from 0.8310 to 0.8808 when it is trained and scored outside. Pooled AUC rises from 0.7897 to 0.8882 and from 0.7936 to 0.8841.
- 549 rows sit at the minimum elevation (1130.5), all in Kabonyo/Kanyagwal. HAND is 0 on all of them and their median slope is 0. 529 are floods (26.9% of the 1,970 flood rows) and 20 are controls.
- The single-seed null gives a pooled AUC of 0.4288 to 0.4613 in all six arms and a per-event mean of 0.4995 to 0.5121.
- src/data/case_control_sampler.py draws cases inside the GFM flood extent and controls from the valid area outside it, and contains no reference to land cover. src/data/gfm_client.py (lines 92 to 140) reads only the ensemble flood extent asset, defines valid as not NODATA_VALUE and flooded as FLOOD_VALUE inside valid, and for each event keeps the scene in the search window (default plus or minus 15 days) with the most flood pixels inside the AOI. A search of that file finds no use of GFM's exclusion mask or reference water mask; the GFM documentation lists both as separate output layers, and the exclusion mask marks where Sentinel-1 flood delineation is hampered. In that file NODATA_VALUE is 255, FLOOD_VALUE is 1 and NO_FLOOD_VALUE is 0. The sampler draws both classes uniformly at random without replacement (seed 42) from the valid pixels of the chosen scene, with no distance buffer and no filter by land cover, water or exclusion. Not established: how the flood-extent layer encodes exclusion-mask and reference-water areas (as 255, 0 or 1).
- Not established: the cause of the land_cover contribution, the cause of the below-chance pooled null, and the nature of the elevation-floor cluster.

## 10. Amendments r3 (2026-10-02, after the baseline run)

Each change has its basis in Section 9.

1. Reference arms. Every comparison reports the five reference arms of Section 9, and a headline claim states its margin over elevation alone in per-event mean AUC. Basis: the six-feature logistic is 0.0314 above elevation alone.
2. Stratified reporting. Every reported arm also gets the pooled AUC inside the non-floor rows and inside each land-cover class that has at least 30 rows of each label. The 549 floor rows cannot be scored on their own (20 controls). Basis: elevation separates classes very unevenly, and 26.9% of flood rows sit in the floor cluster.
3. Claim limits. No statement that terrain explains a model's skill is made unless the six_no_land_cover arm and the non-floor stratum support it. Basis: land_cover accounts for 0.0283 of the 0.0314 margin over elevation alone.
4. Null distribution and selection rule, fixed before the distribution is computed. The headline arm is shuffled within events under 100 seeds (seeds 42 to 141). The rule is implemented as selection_metric_rule in src/models/cv.py: pooled AUC stays the selection metric unless its 95% null interval excludes 0.5; then per-event mean is used if its own null interval does not; if both exclude 0.5 the metric is reported as unresolved. Until that run is logged, Section 5 stands. Basis: the single-seed pooled null is below 0.5 in all six arms.
5. Exit criterion added. The winner is refit and re-scored under the production pins (Python 3.11, numpy 1.26.4, scikit-learn 1.6.1), and the logged run records that environment. Basis: the baseline run came from a Colab environment that differs from those pins.
6. Unchanged. The primary feature set stays the six features of Section 4. No observation so far justifies removing one.
7. Tracked, not blocking: whether controls include areas where GFM flood delineation is hampered (the sampler reads only the flood-extent layer, not the exclusion mask or the reference water mask); the cause of the land_cover contribution; the cause of the below-chance pooled null.

## 11. Findings from the reference-and-null run and the GFM layer audit (2026-10-02)

Sources. Reference-and-null run: MLflow run 3d3778ee609c42049fde93f166583833, commit c3da67794260799e35f6772a547f1a0e7d408bd1, training file hash 0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a. Layer audit: data/derived/gfm_layer_flags.csv, produced by scripts/audit_gfm_layers.py from the 35 id groups in gfm_item_id (67 component scenes, no failed lookups); scenes of one group are combined by the maximum value per point. The counts and arm values in this section are recomputed from that file and the training file by tests/test_mappable_frame.py. Values marked logged are read from the MLflow run and are not re-tested.

Null distribution (logged; six-feature logistic, selection split, 100 label shuffles within events): pooled AUC mean 0.4601, 95% interval 0.4349 to 0.4830; per-event mean AUC 0.5012, interval 0.4721 to 0.5256. The pooled interval excludes 0.5 and the per-event one does not, so the rule of Section 10.4 selects per-event mean AUC.

Stratified pooled AUC (logged; six against six_no_land_cover): class 10 0.8587 and 0.8767, class 30 0.8270 and 0.8317, class 40 0.6019 and 0.6060, class 80 0.3555 and 0.4140, class 90 0.7906 and 0.7935, non-floor rows 0.8755 and 0.7876. Inside every scored land-cover class the model with land_cover ranks no better than the one without it; across the non-floor rows it adds 0.0879.

GFM layers at the sample points. The flood-extent value agrees with the label on all 4,420 rows (1 on the 1,970 flood rows, 0 on the 2,450 controls). No flood row lies inside the exclusion mask or the reference water mask. 945 of the 2,450 controls (38.6%) lie inside the exclusion mask and 8 (0.3%) inside the reference water mask. Controls inside the exclusion mask, by land-cover class:

| Class | Controls | Inside the exclusion mask |
|---|---|---|
| 10 | 123 | 96 |
| 20 | 399 | 319 |
| 30 | 1,523 | 440 |
| 40 | 252 | 17 |
| 50 | 9 | 7 |
| 60 | 2 | 1 |
| 80 | 32 | 21 |
| 90 | 110 | 44 |

What the GFM Product User Manual says (fetched 2026-10-02): the observed flood extent leaves out pixels that are normally under water; a flood pixel inside the exclusion mask or the reference water mask is reset to no flood; the exclusion mask marks where SAR water mapping is not technically feasible (no sensitivity, as in urban areas and dense vegetation; water look-alikes such as flat impervious or sandy surfaces; strong topography; radar shadow; low Sentinel-1 coverage); the no-sensitivity part is applied only to pixels classified as non-flooded; the exclusion mask and topography (DEM, HAND index) are inputs to the flood algorithm.

Two frames. The mappable frame holds every flood row and the controls outside the exclusion mask: 3,475 rows (1,970 floods, 1,505 controls), 23 events with both classes. Pooled AUC and per-event mean AUC on both frames (the arms use the selection split):

| Arm | Full frame | Mappable frame |
|---|---|---|
| elevation only | 0.8239, 0.9077 | 0.7945, 0.8975 |
| distance_river only | 0.7988, 0.8771 | 0.7827, 0.8749 |
| hand only | 0.7871, 0.8061 | 0.7301, 0.7456 |
| slope only | 0.7276, 0.7243 | 0.6654, 0.6726 |
| bare4 logistic | 0.8264, 0.9104 | 0.8053, 0.9030 |
| six logistic | 0.8988, 0.9391 | 0.8841, 0.9383 |
| six_no_land_cover logistic | 0.8361, 0.9108 | 0.8105, 0.9067 |

Observations: the per-event mean of the six-feature arm changes by -0.0008 between the frames, against -0.0605 for hand only, -0.0517 for slope only, -0.0102 for elevation only and -0.0022 for distance_river only. The per-event contribution of land_cover (six minus six_no_land_cover) is 0.0283 on the full frame and 0.0316 on the mappable frame. The margin over elevation only in per-event mean is 0.0314 (six), 0.0027 (bare4) and 0.0031 (six_no_land_cover) on the full frame, and 0.0408, 0.0055 and 0.0092 on the mappable frame.

Not established: the cause of the land_cover contribution, of the class 80 result and of the below-chance pooled null.

## 12. Amendments r4 (2026-10-02)

Each change has its basis in Section 11.

1. Two frames. Every reported arm is scored on the full frame and on the mappable frame. Basis: no flood row lies inside the exclusion mask, 38.6% of controls do, and the GFM flood layer resets flood pixels inside it to no flood, so on those controls the label records that GFM cannot map the pixel, not that it stayed dry.
2. Headline frame. The headline is the selection split on the mappable frame. The shuffle null is recomputed on that frame (100 seeds, 42 to 141) and the rule of Section 10.4 decides the selection metric there. Until that run is logged, per-event mean AUC (the outcome of Section 10.4 on the full frame) stands. Basis: item 1, and the six-feature per-event mean differs by 0.0008 between the frames, so no earlier statement changes.
3. Claim limit. Scores say nothing about locations inside the exclusion mask, or about flooding that GFM cannot detect. The README and the model card must say so before any model is deployed.
4. Margins over elevation only are reported on both frames (Section 11 gives the baseline values).
5. land_cover stays in the primary set. Its contribution does not come from the excluded controls (0.0283 on the full frame, 0.0316 on the mappable frame); its cause is not established and Section 10.3 stays in force.
6. No data rebuild now. The mappable frame measures the effect without changing the training file whose hash every logged run cites. Resampling controls outside the exclusion mask changes that file and is decided after the Phase C results, starting from data/derived/gfm_layer_flags.csv.
7. Resolved: the open question of Section 10 item 7 about the GFM valid mask. Controls do include pixels where GFM flood delineation is hampered (945 of 2,450).
8. Tracked, not blocking: correct the docstring of src/data/case_control_sampler.py (controls are every valid pixel with value 0, which includes the exclusion mask) and add the limitation of item 3 to the README with its test; the cause of the land_cover contribution; the class 80 result; the below-chance pooled null.

Deviations log after the first result:

- r3 (2026-10-02): Sections 9 and 10 added after the baseline run, with the amendments above. No earlier section was edited.
- r4 (2026-10-02): Sections 11 and 12 added after the reference-and-null run and the GFM layer audit. No earlier section was edited.
