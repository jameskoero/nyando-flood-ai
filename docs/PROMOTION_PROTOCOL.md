# Promotion protocol for hgb:con

Date: 2026-10-06. Committed before any promotion gate is computed. Rules as code: `src/models/promotion.py`. Tests: `tests/test_promotion.py`. Register row D41.

## 0. Basis

- D20 (`docs/D20_PROTOCOL.md` Section 13): both boosters passed R-B. Phase C rules R2 to R4 (`src/models/registration.py`), applied to the stored Block A per-event AUCs, select hgb:con: xgb:con minus hgb:con is -0.0002 [-0.0016, 0.0012] on the mappable frame.
- Phase C protocol Section 14.5 requires that a challenger pass R3 on the new events and on the pooled set, and R3 requires both frames. Phase C Section 5 reports a verdict that reverses under a sensitivity run as unresolved.
- Known before this protocol: the D20 results, including the Block A per-event means on the mappable frame (elevation only 0.939, champion 0.950, hgb:con 0.971), and the Phase C battery summaries. No gate below has been computed.
- Nothing here changes a Phase C or D20 result. Block B stays sealed.

## 1. Candidate and statistics

- The candidate is hgb:con as D20 tested it: `build_hgb` with the six features and the monotonic constraints, max_depth 4, max_iter 100, min_samples_leaf 20 (the modal grid point in `docs/D20_RESULTS.json`), with the fixed settings of `src/models/boosters.py`. Nothing is tuned under this protocol, except where a gate reproduces a Phase C evaluation exactly.
- "A beats B" means: the paired per-event bootstrap of A minus B (10,000 resamples of events, seed 42, 97.5% interval) has a lower end above 0, and A has the higher mean (`beats` in `src/models/robustness.py`).

## 2. Statistical gates (all must pass)

- S1. R3 on the new events: hgb:con beats the champion on Block A, on the mappable and the full frame, as stored in `docs/D20_RESULTS.json`.
- S2. R3 on the pooled set: hgb:con beats the champion on the existing events plus Block A, on the mappable frame and on the full frame. The existing events are evaluated as in Phase C (selection split: the champion leave one event out, hgb:con by `nested_scores`), on the full frame for the full-frame test, and each recomputed per-event mean must reproduce the stored Phase C value before it is pooled. The Block A per-event AUCs are those stored by D20. The pooled full-frame test therefore mixes frame definitions (Block A was scored by models trained on the mappable frame); this is recorded with the result.
- S3. Sensitivity subsets, read by reading A plus the floor check (the owner's decision, 2026-10-06). For each subset of `subset_masks` (without the elevation-floor rows, without the zero-distance rows, rows where clay_percent is present), on the new events and on the pooled set, mappable frame: the win is unresolved if the champion beats hgb:con. In the subset without the elevation-floor rows the win is also unresolved if hgb:con's mean advantage over the champion is not above 0. A subset removes the same rows from training and scoring. hgb:con uses the fixed point, the champion is refit on the subset, and Block A is scored by models trained on the existing mappable frame without the rows at locations shared with Block A (D20 Section 12.1) and without the subset's removed rows.
- S4. Buffered split (`buffered_splits`, 1,000 m) on the existing events, mappable frame, the champion and hgb:con at the fixed point: unresolved if the champion beats hgb:con.
- S5. Temporal holdout (`temporal_holdout`, cutoff year 2021), mappable frame, the same reading.
- S6. R6 against elevation only, as a gate (stricter than Phase C R6, which only limits claims; the owner's decision, 2026-10-06): hgb:con beats elevation only on the pooled set, mappable frame (the existing events from the D20 recomputation, plus Block A).
- Reported, not deciding: what reading B (hgb:con must beat the champion in every run) and reading C (its mean advantage must be above 0 in every run) would give. Every run in which hgb:con's mean advantage is not above 0 is listed as a limitation for the model card.

## 3. Export gates (computed only if S1 to S6 pass)

- E1. Through the production feed of `backend/registered.py` (six named [N, 1] inputs, double except land_cover as int64; a double [N, 2] output), the maximum absolute probability difference from scikit-learn is at most 1e-6 on the mappable training rows and on the Block A rows.
- E2. The per-event AUC differences are at most 1e-6 on both.
- E3. The artifact has zero monotonicity violations (`monotone_violations`, 200 real rows): rainfall_3day non-decreasing; elevation, distance_river and slope non-increasing.
- E4. The artifact's SHA-256 is recorded in `models/MANIFEST.json` and recomputed by a test, and the file is named `nyando_hgbcon_<first 12 hex characters of the SHA-256>.onnx`.

## 4. Decision

- Exactly one outcome is recorded: "D20 PROMOTED" when S1 to S6 and E1 to E4 pass, otherwise "D20 NOT PROMOTED — EXISTING CHAMPION RETAINED".
- If any statistical gate fails, no exporter is built, and a regression test records the failing gate.
- If all pass, registration follows in its own pull request, which changes `models/MANIFEST.json` and every file that assumes one active model together.
- Phase D's comparator is the model registered after this decision.
