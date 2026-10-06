# Model card: Nyando Flood AI V2 (registered model)

Structured after Mitchell et al. (2019), "Model Cards for Model Reporting". The numbers below are checked against `docs/PHASE_C_RESULTS.json` by `tests/test_data_docs.py`; the full record is `docs/PHASE_C_CLOSURE.md`.

## Model details
- File: `nyando_logcon_7a909898d4f6.onnx` (ONNX, CPU), SHA-256 `7a909898d4f67efc3291f656d2aa9e7c2559418f5e421eaaa68a158838f1cf8d`. Registered in MLflow run `54afe7e812fd4c4985280283d52f3793`.
- Type: logistic regression, L2 penalty (C = 1), with sign constraints on four features (rainfall up, elevation down, distance to river down, slope down). The slope coefficient is held at 0 by its bound, so the model ignores slope.
- Inputs: elevation, slope, rainfall_3day, distance_river, clay_percent (may be missing) and land_cover (WorldCover class).
- Training data: the mappable frame (3,475 rows) of the 35-date training file, SHA-256 `0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a` (see `data/DATA_SOURCES.md`).
- Serving: `POST /v2/score` returns a ranking score; `GET /v2/metrics` returns the stored results below.
- Developed by James Koero, Kisumu, Kenya, 2026.

## Intended use
Rank locations, inside the areas GFM can map, by flood susceptibility, for research and planning support. The score is a ranking score, not a flood probability: the sample is case-control.
Out of scope: flood probabilities, warnings or safety decisions; locations inside the GFM exclusion mask; floods GFM cannot detect; other basins.

## Evaluation data and method
Leave-one-event-out over 23 scorable events (5 event scenes and 18 other dates with flood pixels), with locations shared with the held-out event removed. The mappable frame is the primary frame. Intervals are 97.5% paired per-event bootstrap intervals.

## Metrics
- Per-event mean AUC: 0.9372 (mappable frame) and 0.9390 (full frame).
- Against elevation only, mappable frame: +0.0397 [0.0076, 0.0774]. The advantage holds on the selection split.
- With neighbours within 1,000 m removed: -0.0249 [-0.0921, 0.0350] against elevation only.
- Out of time (trained on events up to 2021, 16 scorable test events): +0.0246 [-0.0150, 0.0758] against elevation only.
The advantage is not demonstrated under the buffered split or out of time (the intervals include 0).
- On new events (D20: 27 events scored once, `docs/D20_PROTOCOL.md` Section 13): +0.0117 [-0.0107, 0.0381] against elevation only. The value beyond elevation only does not replicate there, so it is shown on the selection split only.

## Quantitative analyses
- land_cover has the largest permutation drop within events: +0.1045 [0.0853, 0.1239]. Its contribution survives out of time: +0.0398 [0.0144, 0.0706]. On the 27 new D20 events the contribution is not shown (+0.0157 [-0.0028, 0.0378]). Whether it carries leaked label information is not excluded (register row D14).
- Gradient-boosting and XGBoost models with the same constraints did not beat this model on both frames on the 23 Phase C events. On the 27 new D20 events both beat it (hgb:con minus this model +0.0206 [0.0094, 0.0309]; xgb:con minus this model +0.0204 [0.0097, 0.0306]) and pooled over 50 events (hgb:con +0.0138 [0.0036, 0.0230]; xgb:con +0.0139 [0.0043, 0.0227]). Which model is registered is decided separately: this card describes the registered model.

## Ethical considerations
- No personal data is used.
- Labels come from one product, and 38.6% of controls lie inside its exclusion mask.
- No thresholds were validated, so there are no risk classes. The score is not calibrated.
- Per-ward evaluation is not done yet (roadmap Phase K).

## Caveats and recommendations
- Do not use for safety decisions.
- The advantage over elevation alone is not robust beyond the selection split: the D20 replication on new events did not confirm it.
- A blank clay value is informative in the training data, so scores without clay are not comparable with scores that include it.

## Legacy model
`models/nyando_xgb_v1.pkl` (SHA-256 `de0e721c808b72730658880337f5f40cb88172f01186eb9e3908e527d6e31bb5`) is the pre-V2 model behind `POST /predict`. Its published figures were retracted (circular evaluation). No validated performance exists for it, and its risk classes were never validated. It is retired in Phase F.

## Citation
Koero, J. O. (2026). Nyando Flood AI. https://github.com/jameskoero/nyando-flood-ai. Model card format: Mitchell et al. (2019).
