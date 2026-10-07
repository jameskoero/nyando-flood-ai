# Phase D protocol: physics-constrained MLP

Date: 2026-10-07 (r1). Committed before any MLP is trained or scored. Rules as code: `src/models/phase_d.py`. Tests: `tests/test_phase_d.py`. Register row D46.

## 0. Basis

- Roadmap Section 3 (Phase D): a small MLP with a few thousand parameters and a soft-constraint loss that penalises violations of dP/d elevation at most 0 and dP/d rainfall at least 0; trained on CPU in Colab, exported to ONNX, served by ONNX Runtime in the same container, no PyTorch in production. It is named a physics-constrained MLP and nothing stronger: there is no PDE loss. It is also Model B of the robustness study.
- Roadmap exit criteria: the winner beats the logistic baseline on grouped-event cross-validation; the A-Gate still passes on the exact training file; the ONNX model loads and predicts correctly inside the Render container; every metric is re-derivable from its MLflow run.
- Serving through ONNX Runtime already exists (register row D33), so Phase D adds a model, not a serving path.
- Comparator: `models/nyando_hgbcon_5ae81ad8b030.onnx` (hgb:con, registered by D43). `logistic:con` is retired.
- `docs/D20_PROTOCOL.md` Section 8: Block B stays unbuilt and unscored until the Phase D candidate and its grid are frozen in a merged protocol; each candidate and the champion are then scored once on it.
- Known before this protocol: the D20 and promotion results, including hgb:con on Block A. Block A is therefore reported for the MLP but does not decide. Block B decides.

## 1. Candidate

- Inputs (14): elevation, slope, rainfall_3day, distance_river and clay_percent standardised with the training-fold mean and standard deviation; clay_percent blank is filled with the training-fold median and flagged by one blank indicator (the data are never imputed; only the model input is); land_cover one-hot over the 8 classes [10, 20, 30, 40, 50, 60, 80, 90], an unseen class mapping to zero.
- Architecture: fully connected, tanh activation, one sigmoid output. Hidden sizes (32, 16) or (64, 32).
- Loss: binary cross-entropy plus the penalty weight times the mean, over the real training rows of the batch, of the sum over constrained features of the positive part of (minus the required sign times dP/dx). The derivative is taken by autograd at the real rows, in standardised units, which keeps its sign. No synthetic rows are used anywhere.
- Constrained features, the same four as hgb:con (`CONSTRAINTS` in `src/models/cv.py`): rainfall_3day up, elevation down, distance_river down, slope down. The roadmap names two; the other two are added for a like-for-like comparison (register row D46).
- A soft penalty does not guarantee monotonicity. Export gate X3 measures it on the artifact.
- Fixed settings: activation tanh, optimizer adam, lr 0.001, weight_decay 0.0001, batch_size 256, epochs 100, seed 42, dtype float32, device cpu.
- Grid, simplest first (6 points): hidden (32, 16) with penalty weight 0 (1025 parameters), hidden (32, 16) with penalty weight 1 (1025 parameters), hidden (32, 16) with penalty weight 10 (1025 parameters), hidden (64, 32) with penalty weight 0 (3073 parameters), hidden (64, 32) with penalty weight 1 (3073 parameters), hidden (64, 32) with penalty weight 10 (3073 parameters). Nothing else is tuned and the grid is never extended.
- Selection: leave-one-event-out with shared locations removed, nested like `nested_scores` in `src/models/boosters.py`, 4 inner event-grouped folds, per-event mean AUC on the mappable frame; ties go to the simplest point.
- Control, reported and not deciding: the selected hidden size with penalty weight 0. No SMOTE. Seeds reported for sensitivity, not deciding: [42, 43, 44, 45, 46].

## 2. Evaluation on the existing events (Stage 2; informs selection, does not decide)

- Leave-one-event-out, shared locations removed, mappable frame as the headline and the full frame reported; paired per-event bootstrap, 10,000 resamples, seed 42, 97.5% interval.
- Reported against logistic:con (the roadmap baseline) and against hgb:con, with the violation rate of the MLP on real rows.

## 3. Freeze, then Block B

- After Stage 2, `docs/PHASE_D_FREEZE.json` records the selected grid point, the training-data SHA-256, the git commit, the torch version, the seeds and the MLflow run, and is merged in its own pull request. Only then is Block B built, by the Block A procedure, and scored once: the selected MLP, its control, hgb:con and elevation only.
- A failed Block B date is listed and never replaced. Under 15 scorable events the comparison is not estimable and the MLP is not promoted.

## 4. Statistical gates (Block B; all must pass)

- G1. The MLP beats hgb:con on the mappable frame and on the full frame.
- G2. The MLP beats elevation only on the mappable frame.
- G3. In the subset without the elevation-floor rows, the MLP's mean advantage over hgb:con is above 0.
- Reported, not deciding: Block A, the existing events, the buffered split and the temporal holdout. Every run in which the MLP's mean advantage is not above 0 is listed as a limitation in the model card.

## 5. Export gates (computed only if G1 to G3 pass)

- X1. Through the production feed, the maximum absolute probability difference between the ONNX artifact and PyTorch is at most 1e-05 on the mappable training rows and on the Block A rows.
- X2. The per-event AUC differences are at most 1e-05 on both.
- X3. The artifact has zero monotonicity violations (`monotone_violations`, 200 real rows) on the four constrained features.
- X4. The artifact's SHA-256 is recorded in `models/MANIFEST.json` and recomputed by a test; the file is named `nyando_mlp_<first 12 hex characters of the SHA-256>.onnx`.
- The hash-verified loader loads it and the container build check passes, as for D33.

## 6. Decision

- Exactly one outcome is recorded: "PHASE D PROMOTED" when G1 to G3 and X1 to X4 pass, otherwise "PHASE D NOT PROMOTED - hgb:con RETAINED". Not promoted is a valid result; the MLP is Model B of the robustness study either way, with its violation rate stated.
- If promoted, registration follows in its own pull request, which changes `models/MANIFEST.json` and every file that assumes one active model together.

## 7. What this does not settle

- Calibration, low/medium/high risk buckets, the alert rule and forecast rainfall belong to Phase E and G. Scores stay rankings; no probability claim is made here.
- The data licence (D37) and the label source (all labels come from GFM).

## 8. Stages

- Stage 1: model code and tests offline; `torch` only in `requirements-train.txt`, never in the production files. Stage 2: Colab CPU training with MLflow. Stage 3: ONNX export with the parity gates. Stage 4: the freeze, Block B, the decision applied once, then serving through the hash-verified loader. Stop after each stage.

## 9. Stage 2 procedure (r2, 2026-10-07; committed before any MLP score exists)

- Frames: the mappable frame selects and is the headline; the full frame is reported. Outer folds: one per event_id of the frame (leave-one-event-out, shared locations removed). An event is scorable with at least MIN_CLASS_N = 30 rows of each class; per-event AUCs are stored in millionths.
- Selection: `select_params` of `src/models/boosters.py` over the six grid points inside every outer fold, computed one fold at a time and cached by `src/models/phase_d_cv.py` (equal to `nested_scores`, tested).
- Selected point: the grid point chosen by most outer folds of the mappable run; ties go to the earlier (simpler) point (`modal_point`). The same rule gives hgb:con's point from its grid.
- Headline (mappable frame, existing events): the MLP's nested outer scores minus hgb:con's nested outer scores (same splits, same selection rule), minus logistic:con (fixed, `build_constrained_logistic`, no tuning) and minus elevation only (score = minus elevation).
- Full frame (reported): the selected points of the MLP and of hgb:con, and logistic:con, each fitted with fixed settings in leave-one-event-out on the full frame.
- Control: the selected hidden size with penalty weight 0, fixed, on both frames. Seeds 42 to 46: the selected point, fixed, mappable frame.
- Statistics: paired per-event bootstrap of the MLP minus each reference, 10000 resamples, seed 42, alpha 0.025, computed from the stored millionths.
- Monotonicity: `monotone_violations` (200 real rows) on whole-mappable fits of the selected MLP, its control and the selected hgb:con; reported, no gate at this stage.
- Reproduction checks, reported and never a gate: hgb:con's mappable mean per-event AUC against 0.943 (tolerance 0.001) and logistic:con's against 0.937239 (tolerance 0.0005).
- Record: `docs/PHASE_D_RESULTS.json` from the run's raw results, with one MLflow run read back. Nothing in this section decides promotion: Block B decides (Section 4). The frozen candidate in `docs/PHASE_D_FREEZE.json` is the selected point, in its own pull request.

## 10. Amendment r3 (2026-10-07, before the Stage 2 fixed-point evaluations are rerun)

- What happened: the first Stage 2 run used MIN_CLASS_N = 30, taken from `src/models/cv.py` where it governs ward-level evaluation, as the scorable-event rule of Section 9. On the mappable frame only 5 events qualified, not the 23 of Phase C, and both reproduction checks reported not reproduced (hgb:con 0.841302 against 0.943, logistic:con 0.840106 against 0.937239). The checks did their job: the rule was wrong, not the models.
- Amendment: an event is scorable when both classes are present, MIN_CLASS_N = 1, the rule of Phase C and of `select_params` (the 5 event scenes and the 18 dates with flood pixels give 23 events on the mappable frame). Section 9's rule of 30 is withdrawn.
- Effect: the selection is unaffected, because `select_params` already used 1; the selected points (MLP: hidden (64, 32), penalty weight 1; hgb:con: depth 4, 100 iterations, minimum leaf 20) come from the nested outer folds, whose probabilities are reused from the first run (the nested code is unchanged). Everything scored with the rule of 30, that is the fixed-point evaluations and every statistic, is recomputed.
- Disclosure: the first run's output under the rule of 30 was seen before this amendment, including its full-frame comparisons. The amendment follows from the event count and the failed reproduction checks, not from a comparison, and it applies to every arm alike.

## 11. Amendment r4 (2026-10-07, before any Block B date is built): early decision on X3

- Stage 2's violation check on the whole-mappable fits (the X3 measure of Section 5: `monotone_violations`, 200 real rows) found the selected MLP violating the constraints on many rows (rainfall_3day 79, elevation 148, distance_river 80, slope 198 of 200), the penalty-weight-0 control on a similar number (125, 136, 160, 197) and hgb:con on none. A soft penalty does not guarantee monotonicity (Section 1) and X3 requires zero violations, so the selected MLP cannot pass X3.
- Rule: if the selected MLP has any violation in that check, the outcome is "PHASE D NOT PROMOTED - hgb:con RETAINED" without building Block B (`early_decision` in `src/models/phase_d.py`). Block B stays unbuilt, unscored and sealed for a future candidate: spending the one unbiased test on a model that cannot be promoted would use it up for nothing.
- Basis and disclosure: the rule rests only on the violation counts, which do not depend on r3, and on no AUC comparison; the counts were seen before this amendment.
- The MLP remains Model B of the robustness study, with its violation rate stated. A future candidate (a stronger penalty, or a network monotone by construction) needs its own protocol and may use Block B.
