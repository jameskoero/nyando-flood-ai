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

## 13. Booster protocol (2026-10-02, committed before any booster score exists)

13.0 Result that closes Section 12 item 2. MLflow run 611bea17379241adb0576279bc7cebc3 (commit 2572fcedb7776c8f8aedd0cdadc182a473f0dedb, training file hash 0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a, layer flags hash 01723237a4cc953d5f5dc6131911c7474d9a86d3b0cfc3adcb1639c79195b4b2; 100 shuffles within events on the mappable frame): pooled AUC null mean 0.4537, 95% interval 0.4203 to 0.4874 (excludes 0.5); per-event mean AUC null mean 0.4998, interval 0.4730 to 0.5333 (includes 0.5). The rule of Section 10.4 selects per-event mean AUC on the mappable frame as it did on the full frame. The same run logged the mappable-frame land-cover lookup (pooled 0.7890, per-event mean 0.8843, recomputed by tests/test_boosters.py), and the other frame values of Section 11 were read back from it with no mismatch above 5e-4.

13.1 Arms. Six features. hgb:con and xgb:con use the monotone constraints rainfall_3day +, elevation -, distance_river -, slope -; land_cover and clay_percent are unconstrained. hgb:free and xgb:free have no constraints and are diagnostic only. land_cover is a categorical column in both libraries (XGBoost with the fixed categories 10, 20, 30, 40, 50, 60, 80, 90); clay_percent stays blank in both.

13.2 Grids and fixed settings, fixed before any score exists.

- HistGradientBoosting: max_depth in {2, 4}; max_iter in {100, 300}; min_samples_leaf in {20, 100}
- XGBoost: max_depth in {2, 4}; n_estimators in {100, 300}; min_child_weight in {1, 10}
- Fixed, HistGradientBoosting: learning_rate = 0.05; l2_regularization = 1.0; early_stopping = False
- Fixed, XGBoost: learning_rate = 0.05; reg_lambda = 1.0; subsample = 1.0; tree_method = hist

Each grid has 8 points, ordered simplest first (the first parameter varies slowest). Seed 42 throughout; XGBoost runs with n_jobs = 1.

13.3 Tuning. Nested: leave-one-event-out outside with shared locations removed (Section 5); inside each outer fold, 4 event-grouped folds over the outer training rows (events dealt round-robin in sorted order; training rows at any location present in the inner test fold dropped). The grid point with the highest inner per-event mean AUC (rounded to 4 decimals) is refit on all outer training rows; ties keep the earlier grid point.

13.4 Decision rules.

1. Primary comparisons, mappable frame, selection split: hgb:con against the six-feature logistic, and xgb:con against it (paired per-event AUC differences, 10,000 bootstrap resamples of events, seed 42, 97.5% interval). An arm beats the baseline only if the lower end of its interval is above 0 and its per-event mean AUC is higher.
2. If both beat the baseline, the arm with the higher per-event mean wins, except that xgb:con also needs a paired interval against hgb:con whose lower end is above 0; otherwise hgb:con wins, because the production image installs scikit-learn and not xgboost. This tie-break is descriptive and is not a third primary comparison.
3. If exactly one beats the baseline, it wins. If neither does, the logistic baseline is the Phase C winner, the exit criterion that the winner beats the baseline is recorded as not met, and no booster is described as better.
4. A winner is described as adding value beyond elevation only if its paired interval against elevation only (same settings) has a lower end above 0 on the mappable frame; otherwise the claim limit of Section 10.3 stands.
5. If a constrained arm beats the baseline, the registered model is a constrained model. The free arms measure the cost of the constraints (per-event mean of free minus constrained, with a paired interval) and never select.
6. The full frame is run for the same arms and reported. If the verdict of item 1 differs between the frames, the result is reported as unresolved.
7. Still required before Phase C closes: the sensitivity runs of Section 5 (without the 549 elevation-floor rows, without the 15 zero-distance rows, rows with clay_percent present), the buffered-neighbour split, the permutation audit, and the refit and load check under the production pins.
8. Rules 1 to 6 are implemented as code in src/models/decision.py and tested with arithmetic vectors in tests/test_roadmap_conformance.py, so the verdict cannot depend on the scores seen. Every departure from the roadmap is listed, with its basis and its test, in docs/ROADMAP_DEVIATIONS.md.

13.5 Monotonicity. tests/test_monotonicity.py trains each constrained arm (heaviest grid point) on the mappable frame and sweeps each constrained feature over its observed 1st to 99th percentile on 200 real rows with the other features fixed; the number of rows moving against the declared direction must be zero for every constrained feature. Each booster run also logs these counts for a model refit with the most frequently chosen grid point, for the constrained and the free arms.

13.6 Facts behind these settings. On randomly generated test inputs (not data), both libraries accepted blank clay_percent, a categorical land_cover and the four constraints, and returned finite predictions for a land-cover class absent from the training rows. One fit on the 3,475-row mappable frame took: HistGradientBoosting 0.13 s (depth 2, 100 iterations) and 1.03 s (depth 4, 400 iterations); XGBoost 0.12 s (depth 2, 100 trees) and 0.62 s (depth 4, 400 trees), on 2 CPUs. Session versions: Python 3.13.15, scikit-learn 1.6.1 (the production pin), xgboost 3.4.1 (requirements.txt says only xgboost>=1.7.0).

## 14. Registration protocol (2026-10-02, committed before any logistic:con score exists)

14.0 Record. Booster runs: 620e9d6219b64bc5818ba0fba82d92d7 (mappable frame, hgb:con and xgb:con), b74d0c183b914b298a6b404e3d49718a (full frame, four arms) and ea49a2b989ed4f6e866be834c3048e2b (mappable frame, free arms; d1ed5bb1b77f43debb85d0d987551718 is an earlier run of the same arms, and the read-back used the later one), all at commit 3557899. Per-event mean AUC, selection split. Mappable: logistic 0.9383, hgb:con 0.9431, xgb:con 0.9436, hgb:free 0.9543, xgb:free 0.9527. Full: logistic 0.9391, hgb:con 0.9554, xgb:con 0.9505, hgb:free 0.9591, xgb:free 0.9586. Paired against the unconstrained logistic (97.5%): mappable hgb:con +0.0048 [-0.0117, 0.0192], xgb:con +0.0053 [-0.0098, 0.0181]; full hgb:con +0.0163 [0.0026, 0.0294], xgb:con +0.0114 [-0.0009, 0.0232]. xgb:con minus hgb:con: mappable +0.0005 [-0.0046, 0.0040], full -0.0049 [-0.0107, 0.0003]. The rules of src/models/decision.py give logistic on the mappable frame, hgb:con on the full frame, and rule 6 reports unresolved.

Cost of the constraints, free minus constrained, paired per-event 97.5%: mappable hgb +0.0112 [-0.0009, 0.0261], xgb +0.0091 [-0.0042, 0.0249]; full hgb +0.0037 [-0.0083, 0.0153], xgb +0.0081 [-0.0024, 0.0180]. All four intervals include 0. Monotonicity violations (rows of 200; rainfall_3day, elevation, distance_river, slope): every constrained arm 0, 0, 0, 0; full hgb:free 200, 21, 200, 200; full xgb:free 200, 47, 200, 200; mappable hgb:free 200, 200, 200, 200; mappable xgb:free 200, 185, 200, 200. Rule 4 for the unconstrained logistic against elevation only: full +0.0314 [0.0013, 0.0658] (wins 12, losses 10); mappable +0.0408 [0.0093, 0.0778] (wins 15, losses 7). The logged unconstrained logistic (run 88e8f6a33fd540b89f4801a8f7fa50b7) has coefficient signs rainfall_3day +, elevation -, distance_river -, slope +: slope is against its declared direction.

Exploratory, post hoc, not a rule: logistic and hgb:con trained on the full frame and scored on three row sets (pooled AUC, per-event mean). All rows: logistic 0.8988, 0.9391; hgb:con 0.9072, 0.9554 (+0.0163). Mappable rows: logistic 0.8813, 0.9304; hgb:con 0.8861, 0.9455 (+0.0151, paired 97.5% [-0.0031, 0.0327], wins 15, losses 6). Floods plus excluded controls: logistic 0.9267, 0.9511; hgb:con 0.9409, 0.9704 (+0.0192). By the reading fixed beforehand (interval includes 0 with a gain on the third set) this is consistent with the boosters learning to separate excluded-zone controls, but the mappable-rows gain is 93% of the all-rows gain, so the exclusion mask explains at most about 0.001 of the point difference under this scoring; the result is inconclusive. Trained on the mappable frame, the logistic scores 0.9383 on the mappable frame and hgb:con 0.9431; trained on the full frame and scored on the mappable rows they score 0.9304 and 0.9455.

14.1 Candidate. logistic:con is the six-feature logistic with the same preprocessing as the baseline and the same objective as scikit-learn's L2 logistic regression (C = 1, intercept unpenalized), solved by bounded maximum likelihood (scipy L-BFGS-B) with the standardized coefficients of rainfall_3day >= 0 and elevation, distance_river, slope <= 0 (src/models/constrained.py). Tests: with no bounds the fit equals scikit-learn's; with the bounds the KKT conditions hold; zero monotonicity violations on real rows.

14.2 Training frame. The registered model is trained on the mappable frame (every flood row plus the controls outside the GFM exclusion mask). Basis: the logistic scores 0.9383 trained on that frame and 0.9304 trained on the full frame, both scored on the mappable rows (14.0), and it is the headline frame of Section 12.

14.3 Rules, fixed before any logistic:con score exists.

R1. A model is eligible only with zero monotonicity violations (the probe of Section 13.5) for rainfall_3day, elevation, distance_river and slope, when fitted on the mappable frame.
R2. Candidates: logistic:con (the champion), hgb:con and xgb:con (the challengers). The unconstrained logistic and the free boosters are reference arms and never register.
R3. A booster replaces the champion only if, on both frames, its paired per-event AUC difference against logistic:con (10,000 bootstrap resamples of events, seed 42, 97.5% interval) has a lower end above 0 and its per-event mean is higher. Otherwise logistic:con is registered.
R4. If both boosters qualify, xgb:con is registered only if its per-event mean on the mappable frame is higher and its paired interval against hgb:con on that frame has a lower end above 0; otherwise hgb:con (the production image installs scikit-learn and not xgboost).
R5. The cost of the constraints is reported as logistic:con minus the unconstrained logistic (paired, 97.5%). It does not change the registration: R1 decides compliance.
R6. The registered model gets the rule-4 test of Section 13.4 against elevation only on the mappable frame; if the lower end of its interval is not above 0, the claim limit of Section 10.3 stands.
R7. Every number is logged to MLflow whatever the outcome, and the exit criterion that the winner beats the baseline is recorded as met only if a booster is registered by R3.

14.4 Evaluation. scripts/run_constrained.py scores logistic:con on both frames (selection split), compares it with the unconstrained logistic and with elevation only, reads the per-event AUCs of the logged booster runs (620e9d6219b64bc5818ba0fba82d92d7 for the mappable frame, b74d0c183b914b298a6b404e3d49718a for the full frame), applies R1 to R4 through src/models/registration.py, and logs the result.

14.5 After Phase C. New GFM scene dates are added from catalogue metadata (flood pixels inside the AOI) by a rule fixed beforehand, never from model scores. The champion and the two challengers are scored on the new events only, with training on the existing dates and shared locations dropped; a challenger replaces the champion only if it passes R3 on the new events and on the pooled set.

14.6 Not decided here: ONNX export with a parity test, logging the model artifact with its run, and the load check under Python 3.11 and numpy 1.26.4 follow the registration.

## 15. ONNX export and the registered artifact (2026-10-02)

15.0 Registration result. MLflow run 54afe7e812fd4c4985280283d52f3793, commit 7a25c17 (pull request 23 merged). Per-event mean AUC, selection split. Full frame: logistic:con 0.9390, against the unconstrained logistic -0.0002 [-0.0022, 0.0022], against elevation only +0.0313 [0.0013, 0.0655]; hgb:con minus logistic:con +0.0165 [0.0029, 0.0291] (wins 16, losses 5, beats), xgb:con minus logistic:con +0.0116 [-0.0007, 0.0233] (wins 15, losses 6, does not beat). Mappable frame: logistic:con 0.9372, against the unconstrained logistic -0.0011 [-0.0039, 0.0014], against elevation only +0.0397 [0.0076, 0.0774]; hgb:con minus logistic:con +0.0058 [-0.0117, 0.0213] (wins 12, losses 7), xgb:con minus logistic:con +0.0064 [-0.0096, 0.0202] (wins 12, losses 6); neither beats. Violations of logistic:con fitted on the mappable frame: 0, 0, 0, 0 (rainfall_3day, elevation, distance_river, slope). By rules R1 to R4 the registered arm is logistic:con; by R7 the exit criterion that the winner beats the baseline is recorded as not met, because no booster registered.

15.1 The registered model. logistic:con fitted on the mappable frame (3,475 rows; converged in 100 iterations). Standardized coefficients in design order: elevation -1.8624, slope +0.0000 (its bound is active, so the model ignores slope), rainfall_3day +0.6537, distance_river -1.0963, clay_percent +0.7656, missing-clay indicator +0.0296, land_cover 10 +1.1798, 20 -2.5650, 30 -1.2384, 40 +0.9053, 50 -0.6118, 60 -0.1874, 80 +2.0217, 90 +0.4958; intercept -0.2047. tests/test_registered_model.py recomputes these from the data.

15.2 Why an exporter of our own. On 2026-10-02 skl2onnx 1.20.0 failed to convert the pipeline: it has no shape calculator for sklearn.impute.MissingIndicator. src/models/export_onnx.py writes the fitted pipeline as an explicit graph (median imputation, standardization, the clay_percent indicator, a one-hot land_cover, a linear score, a sigmoid), checks the structure it understands and refuses anything else, uses double precision, opset 13, IR version 8 and deterministic serialization. Scoring needs only onnxruntime. Parity on all 3,475 mappable rows: maximum absolute difference 2.22e-16 from scikit-learn for logistic:con and for the unconstrained logistic, predicted labels agree on every row, 43 graph nodes (tests/test_onnx_export.py, tolerance 1e-9). Environment record: in the Colab notebook process protobuf 5.29.6 was loaded while a fresh interpreter loaded 7.36.2, and onnx 1.23.1 refused to import there (its generated code needs 6.31.1 or newer although its package metadata does not say so); a clean virtual environment with numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.6.1, onnx 1.23.1, onnxruntime 1.30.0 and protobuf 7.36.2 ran the export and the tests. requirements-onnx.txt pins onnx, onnxruntime and protobuf >= 6.31.1 for CI.

15.3 The artifact. models/nyando_logcon_7a909898d4f6.onnx, SHA-256 7a909898d4f67efc3291f656d2aa9e7c2559418f5e421eaaa68a158838f1cf8d, 5082 bytes, written by scripts/export_registered.py and entered in models/MANIFEST.json as active (algorithm logistic_constrained, trained_on the training-file hash). Its metadata names the model, the frame, the features, the training-data and layer-flag hashes, this section, the registration run, the input contract and the claim limit: scores rank locations inside the areas GFM can map, they are not flood probabilities, and they say nothing about locations inside the GFM exclusion mask. Two exports in the same environment gave identical bytes. tests/test_registered_model.py checks the manifest entry, the metadata, that refitting from the committed data reproduces the artifact's scores within 1e-5 (a tolerance chosen with margin, not measured), zero monotonicity violations, that the artifact ignores slope, and that missing clay and an unseen land-cover class are scored.

15.4 CI. The test job installs requirements-onnx.txt next to requirements.txt, and the ONNX tests fail instead of skipping there when onnx or onnxruntime is unusable. That run is the load check of the artifact under Python 3.11 and numpy 1.26.4.

15.5 Not decided here. Serving: the backend would load the file with onnxruntime after checking its SHA-256 against models/MANIFEST.json, the Dockerfile would add onnxruntime and the API's input contract would change; the legacy model stays served until that pull request. Logging the artifact to run 54afe7e812fd4c4985280283d52f3793 (register row D12 stays open). The robustness battery, the sensitivity runs and the README claim-limit line follow in separate pull requests.

Deviations log after the first result:

- r3 (2026-10-02): Sections 9 and 10 added after the baseline run, with the amendments above. No earlier section was edited.
- r4 (2026-10-02): Sections 11 and 12 added after the reference-and-null run and the GFM layer audit. No earlier section was edited.
- r5 (2026-10-02): Section 13 added after the mappable-frame run: the booster arms, grids, tuning protocol and decision rules, committed before any booster score exists, with every departure from the roadmap listed in docs/ROADMAP_DEVIATIONS.md. No earlier section was edited.
- r6 (2026-10-02): Section 14 added after the booster runs and their read-back: the sign-constrained logistic candidate, the training frame and the registration rules R1 to R7, committed before any logistic:con score exists. No earlier section was edited.
- r7 (2026-10-02): Section 15 added after the registration run and the ONNX probe: the registration result, the registered model, the exporter, the artifact and the CI load check. No earlier section was edited.
