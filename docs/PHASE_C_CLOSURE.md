# Phase C closure record

Date: 2026-10-05. Rules: [PHASE_C_PROTOCOL.md](PHASE_C_PROTOCOL.md), Sections 1 to 16. Values: [PHASE_C_RESULTS.json](PHASE_C_RESULTS.json), read from MLflow by `scripts/build_phase_c_results.py`. This record is rendered from those values by `src/models/closure.py`, and `tests/test_phase_c_results.py` fails if the two differ.

## Outcome

- Registered model: `models/nyando_logcon_7a909898d4f6.onnx`, a sign-constrained logistic regression (`logistic:con`) trained on the mappable frame; SHA-256 `7a909898d4f67efc3291f656d2aa9e7c2559418f5e421eaaa68a158838f1cf8d`.
- Exit criterion "the winner beats the baseline": **not met**. No booster beat `logistic:con` on both frames (rules R1 to R4, protocol Section 14.3), so the registered model is the simplest candidate that satisfies the declared monotonic constraints. No booster is described as better.
- Robustness (protocol Section 16.4): the verdict **stands** in all three sensitivity subsets; no booster beats `logistic:con` in any of them.
- Claim limit: scores rank locations inside the areas GFM can map. They are not flood probabilities (the sample is case-control) and say nothing about locations inside the GFM exclusion mask or about floods GFM cannot detect.
- Value beyond elevation only (protocol Section 13.4 item 4, lower end of the paired interval above 0): selection split holds; 1,000 m buffered split not demonstrated; out of time (events after 2021) not demonstrated.

## Exit criteria (protocol Section 6)

| Criterion | State | Evidence |
|---|---|---|
| The winner beats the baseline | Not met | rules R1 to R4 applied to the logged runs: registered arm `logistic:con` |
| The A-Gate and the monotonicity test pass in CI | Met | required checks `data-gate` and `test` on main; `tests/test_monotonicity.py`, `tests/test_onnx_export.py`, `tests/test_registered_model.py`; violations of the registered model (rows of 200): rainfall_3day 0, elevation 0, distance_river 0, slope 0 |
| The model-load check passes under the production pins | Met | the CI job `test` installs `requirements.txt` (numpy 1.26.4, scikit-learn 1.6.1) and `requirements-onnx.txt` on Python 3.11 and loads the committed artifact with onnxruntime in `tests/test_registered_model.py` |
| Every reported number is traceable to an MLflow run with a training-data hash | Met | the runs below, all with training-data SHA-256 `0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a` |
| The winner is entered in `models/MANIFEST.json` | Met | `models/nyando_logcon_7a909898d4f6.onnx`, name rule `nyando_<algorithm>_<hash prefix>` |
| Permutation audit on the six features (a Phase A item) | Done | run `7582915498fb46b4be850fe0440ed149`; reliance is not legitimacy, see below |
| The registered artifact is logged to the registration run (register row D12) | Done | MLflow artifact listing of run `54afe7e812fd4c4985280283d52f3793` shows the file |

## Registration (MLflow run `54afe7e812fd4c4985280283d52f3793`, commit `7a25c17d128eccce8ab8b0a72fc28d9ba398e916`)

Per-event mean AUC under the selection split (leave one event out, shared locations removed). Intervals: paired per-event bootstrap, 97.5%, 10,000 resamples of events, seed 42.

| Frame | logistic:con | vs unconstrained logistic | vs elevation only | hgb:con minus logistic:con | xgb:con minus logistic:con |
|---|---|---|---|---|---|
| mappable | 0.9372 | -0.0011 [-0.0039, 0.0014] (wins 8, losses 10) | +0.0397 [0.0076, 0.0774] (wins 13, losses 9) | +0.0058 [-0.0117, 0.0213] (wins 12, losses 7) | +0.0064 [-0.0096, 0.0202] (wins 12, losses 6) |
| full | 0.9390 | -0.0002 [-0.0022, 0.0022] (wins 7, losses 14) | +0.0313 [0.0013, 0.0655] (wins 12, losses 10) | +0.0165 [0.0029, 0.0291] (wins 16, losses 5) | +0.0116 [-0.0007, 0.0233] (wins 15, losses 6) |

## Robustness battery (MLflow run `7582915498fb46b4be850fe0440ed149`, commit `ae2cbc8a2df172fc1aca516b9bc2e47cf8ff501d`)

### Prior-only null (16.1)

- mappable frame: pooled AUC 0.4104, per-event mean 0.5000. The pooled shuffle null interval is 0.4203 to 0.4874, so the mechanism (the training prevalence falls when a flood-heavy event is held out) is **not supported**.
- full frame: pooled AUC 0.4020, per-event mean 0.5000. The pooled shuffle null interval is 0.4349 to 0.4830, so the mechanism (the training prevalence falls when a flood-heavy event is held out) is **not supported**.

### Permutation audit of the registered model (16.2)

Unpermuted: pooled AUC 0.8830, per-event mean 0.9372. 20 repeats per feature.

| Feature | Within-event drop (97.5% interval) | Relied upon | Pooled drop (sd over repeats) |
|---|---|---|---|
| elevation | +0.0657 [0.0538, 0.0785] (wins 23, losses 0) | yes | +0.0693 (0.0051) |
| slope | -0.0000 [-0.0000, 0.0000] (wins 6, losses 7) | no | +0.0000 (0.0000) |
| rainfall_3day | +0.0053 [-0.0014, 0.0139] (wins 14, losses 7) | no | +0.0206 (0.0023) |
| distance_river | +0.0558 [0.0367, 0.0761] (wins 21, losses 2) | yes | +0.0517 (0.0037) |
| clay_percent | +0.0125 [0.0002, 0.0269] (wins 14, losses 9) | yes | +0.0254 (0.0033) |
| land_cover | +0.1045 [0.0853, 0.1239] (wins 23, losses 0) | yes | +0.1265 (0.0062) |

The slope drop is 0 because the registered model's slope coefficient is held at 0 by its sign bound. A feature that does not vary inside an event would also have a within-event drop of 0 by construction, so its pooled drop is the column to read. Reliance is not legitimacy: this audit does not show that any feature is free of leaked signal.

### Temporal holdout (16.3)

Trained on events dated 2021 or earlier (1106 rows), scored on later events (2186 rows, 16 scorable test events; 0 events straddle the cutoff).

- Per-event mean: logistic:con 0.9257; without land_cover 0.8858; elevation only 0.9011.
- Six features against elevation only: +0.0246 [-0.0150, 0.0758] (wins 6, losses 9). The value beyond elevation is not demonstrated out of time.
- Six features against the arm without land_cover: +0.0398 [0.0144, 0.0706] (wins 13, losses 2). The land_cover contribution survives out of time.
- Neither reading shows that land_cover is free of leaked signal; register row D14 stays open.

### Sensitivity subsets (16.4)

| Subset | Rows | Events | logistic:con | vs elevation only | hgb:con minus con | xgb:con minus con | Reversal |
|---|---|---|---|---|---|---|---|
| without the 549 elevation-floor rows | 2935 | 22 | 0.9123 | +0.0463 [0.0021, 0.0953] (wins 13, losses 8) | +0.0095 [-0.0132, 0.0314] (wins 12, losses 5) | +0.0136 [-0.0087, 0.0350] (wins 12, losses 7) | no |
| without the 15 zero-distance rows | 3460 | 23 | 0.9399 | +0.0405 [0.0083, 0.0780] (wins 14, losses 8) | +0.0040 [-0.0148, 0.0199] (wins 13, losses 6) | +0.0039 [-0.0131, 0.0184] (wins 15, losses 5) | no |
| rows where clay_percent is present | 2899 | 22 | 0.9111 | +0.0453 [0.0007, 0.0951] (wins 14, losses 7) | +0.0193 [-0.0056, 0.0429] (wins 11, losses 7) | +0.0143 [-0.0096, 0.0359] (wins 11, losses 8) | no |

### Buffered-neighbour split (16.5, 1,000 m)

- logistic:con per-event mean 0.8726 (minimum training rows in a fold: 371); against its own selection-split result: -0.0646 [-0.1165, -0.0245] (wins 1, losses 19).
- Against elevation only: -0.0249 [-0.0921, 0.0350] (wins 10, losses 12). The value beyond elevation is not demonstrated under the buffered split.
- Against the unconstrained logistic: +0.0023 [-0.0003, 0.0051] (wins 11, losses 3).

## Limitations that stay

- The labels come from GFM: 945 of the 2,450 controls (38.6%) lie inside its exclusion mask, so "control" means "not detected as flooded" (protocol Section 12).
- 23 scorable events give paired intervals about 0.015 wide on each side; an interval that includes 0 means "not shown", not "equal".
- The boosters' tuning chose the edge of the committed grids; the grids were not widened after seeing scores (protocol Section 13).
- The cause of the land_cover contribution is not established (register row D14).
- The slope coefficient of the registered model is held at 0 by its bound, so the model ignores slope.

## Open items and owners

- Serving the registered model (D9): a deploy pull request with onnxruntime in the image, a SHA-256 check before loading and a new input contract.
- The OSM check of the 15 zero-distance rows (D11) and the train-check, drift-monitor and deploy workflows (D13).
- Confirmatory replication on new GFM scene dates, champion against challengers (D20).
- Dependency pinning in `ci.yml`, a `permissions:` block and SHA-pinned actions (hardening pull request).

## Reproducibility

| Run | MLflow run ID | Commit |
|---|---|---|
| logistic:con scoring | `54afe7e812fd4c4985280283d52f3793` | `7a25c17d128eccce8ab8b0a72fc28d9ba398e916` |
| robustness battery | `7582915498fb46b4be850fe0440ed149` | `ae2cbc8a2df172fc1aca516b9bc2e47cf8ff501d` |
| boosters, mappable frame | `620e9d6219b64bc5818ba0fba82d92d7` | `35578998ecae90f00fb0d24c61da0731e7b46df0` |
| boosters, full frame | `b74d0c183b914b298a6b404e3d49718a` | `35578998ecae90f00fb0d24c61da0731e7b46df0` |

Regenerate the results and this record with `python scripts/build_phase_c_results.py` (needs a DagsHub token).

## Update after D20 (2026-10-05)

The confirmatory replication scored 27 new events once with the frozen Phase C models (`docs/D20_PROTOCOL.md` Section 13, `docs/D20_RESULTS.json`, MLflow run `2aa7d3b9d810426181425379a97f2828`). It supersedes parts of this record; the registered model is unchanged by it.
- Value beyond elevation only: champion minus elevation only on the new events +0.0117 [-0.0107, 0.0381] (wins 11, losses 16); the value beyond elevation only does not replicate.
- land_cover contribution: +0.0157 [-0.0028, 0.0378] (wins 13, losses 13); the contribution is not shown on the new events. Leakage is not excluded either way.
- hgb:con minus the champion: new events +0.0206 [0.0094, 0.0309] (wins 21, losses 5); pooled over 50 events +0.0138 [0.0036, 0.0230] (wins 33, losses 12); zero violations: yes; it passes the replacement rule.
- xgb:con minus the champion: new events +0.0204 [0.0097, 0.0306] (wins 22, losses 5); pooled over 50 events +0.0139 [0.0043, 0.0227] (wins 34, losses 11); zero violations: yes; it passes the replacement rule.
- The statement above that no booster beat logistic:con describes the 23 Phase C events only. Registering a booster is a separate decision under rules R2 to R4.
