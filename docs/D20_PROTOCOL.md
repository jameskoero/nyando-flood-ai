# D20 protocol: confirmatory replication of the Phase C models on new GFM scene dates

Register row D20 (open); this protocol is row D39. It was written before any new date was built and before any model scored a new event. The selection and sample-size rules are code in `src/models/d20.py`, and `tests/test_d20.py` recomputes the committed selection from the committed scan, so the lists cannot be edited by hand.

## 1. Why
Phase C closed with the exit criterion "the winner beats the baseline" not met. It left 23 scorable events, an advantage over elevation alone that held only on the selection split, and an open question about land_cover (see `docs/PHASE_C_CLOSURE.md`). D20 tests the frozen Phase C models on events none of them has seen. A second block of dates is sealed for the Phase D candidate.

## 2. Selection rule
Written down before the scan was run. The scan reads label metadata only (valid pixels and flood pixels per GFM scene date over the ward area), never model output. The scan file and this rule are committed together, so the order cannot be proven from the repository alone.
- Window: GFM scene dates from 2020-01-01 to 2026-08-31.
- Excluded: any date within 14 days of one of the 35 existing scene dates.
- Eligible: at least 997,936 valid pixels (95% of the 1,050,459 pixels of the ward area) and at least 100 flood pixels, counted on the merged slices of the date.
- Selection: the eligible dates in ascending order, shuffled with `random.Random(42)`; a date is accepted if it is at least 14 days from every existing and every accepted date; at most 60 dates.
- Blocks: Block A takes the even acceptance positions and Block B the odd ones.

## 3. Scan result
- The GFM catalogue held 756 scene dates in the window over the ward bounding box (scan log). 462 of them are at least 14 days from every existing date and were scanned (`data/derived/d20_scan.csv`, SHA-256 `effe991a0d4906c11424253a46bff624acb1dac393fe3faab7ea9d71d2ea3dc7`, 0 errors).
- Valid pixels over the ward area: none on 264 dates, partial on 43, full on 155.
- Eligible under the rule: 112 dates. Selected: 53 dates (by year: 2020: 8, 2021: 6, 2022: 5, 2023: 11, 2024: 8, 2025: 7, 2026: 8; 47 calendar months), 27 in Block A and 26 in Block B. The lists are in `data/derived/d20_selection.json`.
- Every selected date has flood pixels, so every new date can give a per-event AUC. The 12 flood-free dates of the training file cannot.

## 4. Sample size for the new dates
70 points per class per date. The rule: the smallest multiple of 10, at least 40, for which the share of controls kept outside the GFM exclusion mask, at the 10th percentile (nearest rank) of the 30 existing date sets (0.500; range 0.400 to 0.675), leaves at least 35 controls, the 30 that `MIN_CLASS_N` requires plus a margin of 5. Flood points are never removed by the mappable restriction.

## 5. Frozen models
- Champion: the registered `logistic:con` artifact (`models/nyando_logcon_7a909898d4f6.onnx`), trained on the mappable frame of the 35-date file.
- Challengers: `hgb:con` and `xgb:con`, refit on the same frame at the modal grid points of the logged booster runs (MLflow runs `620e9d6219b64bc5818ba0fba82d92d7` and `b74d0c183b914b298a6b404e3d49718a`).
- References: elevation only; the six-feature logistic without `land_cover`.
- Training rows at any location that appears in the scored block are dropped, as in the selection split.

## 6. Scoring
Each model is scored once on Block A. The new points are built into a separate file with its own manifest entry; the 35-date file is not changed. Per-event AUC on the mappable frame is primary and the full frame secondary. An event is scorable on a frame with at least 30 rows of each class. Intervals: paired per-event bootstrap, 97.5%, 10,000 resamples, seed 42.

## 7. Decision rules (`src/models/d20.py`)
- R-A, value beyond elevation: it replicates if the lower end of the interval for the champion minus elevation only on Block A is above 0. Otherwise the README, the model card and the closure record say it did not replicate.
- R-B, challengers: a challenger replaces the champion only if its interval against the champion has a lower end above 0 and a higher mean both on Block A and on all events pooled (the 23 existing events plus Block A), and its monotonicity checks pass. Otherwise the champion stays.
- R-C, land_cover: the contribution (six features minus six without land_cover) on Block A is reported; it survives if the lower end is above 0. Leakage is never declared excluded.
- S-1, sensitivity: the same differences bootstrapped over calendar-month clusters. If a verdict holds by event but the cluster interval's lower end is not above 0, the report says "not robust to clustering by calendar month". The verdict itself does not change.
No rule changes after the first score. A rule found wrong later is recorded as a deviation with its reason, and the original verdict stays in the record.

## 8. Sealed block for Phase D
Block B's dates are fixed now. Its features are not built, and no model is scored on it, until the Phase D candidate and its grid are frozen in a merged protocol. Each Phase D candidate and the champion are then scored once on Block B. Building or scoring Block B earlier is a protocol violation.

## 9. What this does not settle
The data licence (D37); the label source (all labels come from GFM, so D20 cannot detect a label-process artefact); and whether Block A joins a future training file.

## 10. What the new events can resolve
The booster-versus-champion half-width on the mappable frame is about 0.0165 on 23 events. If per-event spread holds, it scales with 1 / sqrt(events): about 0.0112 on 23 + 27 events. Differences near 0.01 stay borderline; the larger gap over elevation is the main test.

## 11. Dates that fail to build (added 2026-10-05, before any Block A date was built)
Section 6 did not say what happens when a committed Block A date cannot be built or scored. This rule was added before any Block A feature was built and before any model scored a new event.
- A Block A date that fails to build (an exception, an incomplete CHIRPS window, coverage under 997,936 valid pixels at build time, or no usable rows) is recorded with its reason in the Block A build report and is never replaced. It is not replaced from Block B: Block B stays sealed.
- A built date that has fewer than 30 rows of a class on a frame is reported as not scorable on that frame, and is not replaced either.
- Block A may therefore have fewer than 27 dates. The scored set is the built dates that are scorable on the frame in question.
- Floor: with fewer than 15 scorable Block A events on the mappable frame, R-A, R-B and R-C are reported as not estimable and no claim is made. At 15 events the booster-versus-champion half-width would be about 0.0204, wider than Phase C's 0.0165.
- In code: `built_block_a` and `block_a_estimable` in `src/models/d20.py`; `tests/test_d20.py` checks them and the text of this section.

## 12. Block A build record (2026-10-05, before any model scored Block A)
- Built 27 of 27 committed dates. Failed dates, listed and not replaced (Section 11): none.
- Rows: 3780 (1890 floods, 1890 controls) at 70 points per class per date.
- Controls outside the GFM exclusion mask: 1172 of 1890 (62.0%).
- Scorable events (at least 30 rows of each class): 27 on the mappable frame and 27 on the full frame. The floor of 15 on the mappable frame is met.
- Files: `data/confirmatory/`; their manifest entries carry no `label_source`. No model has scored these rows, and Block B is untouched.

## 12.1 Locations shared with Block A (added 2026-10-05, after the first scoring attempt stopped on them and before any Block A score existed)
Section 5 says training rows at any location that appears in the scored block are dropped, as in the selection split, but the registered artifact was fit on every mappable row. The first scoring attempt checked for shared locations and stopped before it scored anything: 236 locations of Block A also appear in the mappable training frame. Block A's points come from the same GFM pixel grid as the training file, so overlap is expected, as it is inside the training file itself (the selection split removes training rows at held-out locations for that reason).
- Primary analysis: every arm is fit on the mappable training frame with the rows at locations that appear in Block A removed (all Block A rows, so one training set serves both frames). The champion is refit by the same pipeline as the registered artifact (`logistic:con`, the same code and settings) on that reduced frame, and the challengers are refit on it at the same modal grid points.
- Sensitivity: the registered artifact itself, fit on every mappable row and therefore having seen those locations, is scored on Block A and reported next to the primary champion. The primary verdicts do not depend on it.
- In code: `shared_training_rows` in `src/models/d20.py`; the number of shared locations and of training rows dropped are stored in `docs/D20_RESULTS.json`.
- No other rule changes.
