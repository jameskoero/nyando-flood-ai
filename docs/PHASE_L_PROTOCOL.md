# Phase L protocol: leakage audit of land_cover and clay_percent (register rows D14 and D55)

This protocol and the rules in `src/models/phase_l.py` are fixed before any Phase L metric exists. The run (L1), its results file (L2) and the Model Card text that cites them follow. Nothing here changes the registered model, a score or a band.

## 1. Scope

Two questions about the registered model hgb:con (SHA-256 `5ae81ad8b0303c8a8bef79f177199e34f1d3cad87cb609afc6825a4b9980b723`):

- D14: does `land_cover` add discrimination between rows of the same land-cover class, or does its contribution come only from how land-cover classes differ in the sample?
- D55: does the model's use of `clay_percent` come from the observed value, or from whether the value is missing?

The training and Block A samples are case-control. The outcome changes only what the Model Card may say. There is no causal reading, no real flood prevalence, and no lead-time or early-warning claim.

## 2. Why these two questions (facts on record)

- MODEL_CARD.md: `land_cover` has the largest permutation drop within events, +0.1045 [0.0853, 0.1239], and its contribution survives out of time, +0.0398 [0.0144, 0.0706].
- docs/PHASE_C_CLOSURE.md: on the 27 new D20 events the `land_cover` contribution is +0.0157 [-0.0028, 0.0378] (wins 13, losses 13), it is not shown on the new events, and leakage is not excluded either way. Register row D14 stays open.
- docs/PHASE_C_PROTOCOL.md Section 11 reports the stratified pooled AUC inside land-cover classes and the controls inside the GFM exclusion mask by class. These rules were written after those results were known and do not use them.
- MODEL_CARD.md: A blank clay value is informative in the training data, so scores without clay are not comparable with scores that include it.

## 3. Frames and arms

- Primary frame (decides what the Model Card may say): the Block A rows kept after the exclusion of rows at training locations (E1 amendment r1), scored as in E1.
- Secondary frame (context only): the out-of-fold reference scores on the mappable training rows, built as in E1.
- Arms: `six` is the registered model (primary frame) or its out-of-fold reference (secondary frame). `no_land_cover` and `no_clay` are audit models trained with the registered fixed parameters, the same monotonic constraints and the same rows and procedure as the registered artifact, with the one feature removed. Audit models are not registered, promoted or served.

## 4. Estimands

Within-class concordance W. A pair is one flood row and one control row of the same land-cover class. W is the share of such pairs, pooled over the estimable classes, in which the flood row has the higher score; a tie counts one half. Worked example (arithmetic, not data): class A has flood scores 0.9 and 0.6 and control scores 0.4 and 0.5, so 4 of 4 pairs agree; class B has flood score 0.3 and control scores 0.7 and 0.2, so 1 of 2 pairs agrees; W = (4 + 1) / (4 + 2) = 5/6.

Total T. The per-event mean AUC over the frame, as in E1 and Phase K, the quantity earlier contribution figures use.

Contrasts, always the registered arm minus the audit arm:

- D14: delta W for land_cover, W(six) - W(no_land_cover), and for context delta T for land_cover.
- D55: delta T_all for clay, T(six) - T(no_clay) on all rows of the frame, and delta T_present for clay, the same difference on the rows where clay is observed. No value is imputed or filled in.

Intervals: event-clustered percentile bootstrap, the 2.5th to 97.5th percentile of N_BOOT = 10000 resamples of events with SEED = 42 (a two-sided 95% interval), the constants of `src/advisory/rules.py`. Both arms are resampled with the same events.

## 5. Estimability

A land-cover class enters W in a frame only if it has at least MIN_ROWS_PER_CLASS = 30 flood rows, 30 control rows and MIN_EVENTS_BOTH = 5 events that contain both classes (the Phase K thresholds, reused unchanged). Classes that do not meet the rule are listed with their counts and left out; they are never merged or imputed. For D55 the clay-observed rows must meet the same rule as a whole. If too little remains, the outcome is `not_estimable` and no claim is possible.

## 6. Decision rules and the claims they allow

D14, on the primary frame, from the interval of delta W:

- `within_class_gain`: the lower end is above 0. In Block A the model discriminates better with land_cover than without it between rows of the same land-cover class; this is an association in a case-control sample, not a cause.
- `within_class_loss`: the upper end is below 0. In Block A land_cover lowers the ranking between rows of the same land-cover class; this audit leaves the registered model unchanged and the finding goes to the owner as a decision.
- `between_class_only`: the interval includes 0. In Block A land_cover does not add discrimination between rows of the same land-cover class; its contribution comes from how land-cover classes differ in the sample, a limit the Model Card states.
- `not_estimable`. No claim is possible: too little of Block A meets the estimability rule.

D55, on the primary frame, checked in this order:

- `value_gain`: the lower end of delta T_present is above 0. Where clay is observed, its value adds discrimination in Block A.
- `missingness_only`: the lower end of delta T_all is above 0 and the interval of delta T_present includes 0. The contribution of clay in Block A comes through whether it is missing, not through its observed value; the Model Card states that the model reads a missing clay value as a signal.
- `no_contribution`: both intervals include 0. No contribution of clay is shown in Block A, with or without its missing values.
- `other`: anything else. The two clay contrasts do not fit one summary; both are reported as measured and no summary claim is made.
- `not_estimable`. No claim is possible: too little of Block A meets the estimability rule.

A result in the secondary frame that differs from the primary one is stated, not averaged. An interval that includes 0 does not show that a contribution is absent: Block A has 27 events and the intervals can be wide. The audit models are not registered, promoted or served; no causal reading, no real flood prevalence and no lead-time or early-warning claim follows.

## 7. Procedure and records

- L1 is a manual, read-only workflow that trains the audit models, computes the contrasts on both frames, prints one RESULT_JSON line and logs one MLflow run. It stops if the registered artifact, the training file or the E1 cutpoints differ from the records.
- L2 stores `docs/PHASE_L_RESULTS.json` (pinning the SHA-256 of this file), generates the Model Card and Datasheet text from it, and closes D14 and D55 on a results test.
- A change after the run is recorded as a register row; this file is not edited.

## 8. Counts of the training file (generated)

<!-- phase-l:counts:begin -->
Training file `data/training/nyando_training_v2_multidate.csv`: 4,420 rows (1,970 flood rows, 2,450 control rows).

| land_cover class | Control rows | Flood rows |
|---|---|---|
| 10 | 123 | 269 |
| 20 | 399 | 4 |
| 30 | 1,523 | 327 |
| 40 | 252 | 678 |
| 50 | 9 | 0 |
| 60 | 2 | 1 |
| 80 | 32 | 327 |
| 90 | 110 | 364 |

| clay_percent | Control rows | Flood rows |
|---|---|---|
| blank | 45 | 554 |
| observed | 2,405 | 1,416 |
<!-- phase-l:counts:end -->

## 9. What this does not do

It does not change the registered model, test the sampling design itself, explain why land-cover classes differ between flood and control rows, test lead time, or cover other features.
