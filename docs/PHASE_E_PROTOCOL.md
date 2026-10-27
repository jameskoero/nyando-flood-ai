# Phase E protocol: validated risk levels and the LLM advisory

Committed before any level is computed and before any advisory is served (register row D49). Later changes are dated amendments, as in the Phase D protocol.

## 1. Scope

- E1: low, medium and high levels for the registered model's score, fixed in this protocol and validated once on held-out data.
- E2 and E3: an advisory module and `POST /v2/advisory`. The roadmap (Section 4, Phase E) asks for a Gemini free-tier advisory with a Groq free-tier fallback, never on the scoring path, timeout-isolated, cached, and degrading to a template message.
- Out of scope: SMS (Phase G), the dashboard (Phase F) and any calibration of scores to probabilities.

## 2. What a level is

- The registered `hgb:con` returns a ranking score from a case-control model, not a flood probability (its `claim_limit`). A level is a rank band: where a score sits among the scores of a reference set.
- A level is not a probability and carries no lead time. Any text that shows a level says so. Levels apply only inside GFM-mappable areas, as the model's claim limit does.

## 3. Reference and cutpoints

- Reference scores: out-of-fold scores of `hgb:con` at the registered parameters (max_depth 4, max_iter 100, min_samples_leaf 20) on the mappable training frame, leave-one-event-out with shared locations removed, as in Phase C.
- Cutpoints are the 1/3 and 2/3 quantiles of the reference scores. LOW is below the first, MEDIUM from the first to below the second, HIGH from the second up (`src/advisory/rules.py`).
- The cutpoints are fixed here. They are never tuned on Block A or on any later data.

## 4. Validation (E1)

- Held-out data: Block A, mappable rows, scored once by the registered artifact. Block A takes no part in choosing a cutpoint.
- Reported per level: rows, flood rows and flood share, and the HIGH minus LOW difference with a 97.5% interval from an event-clustered bootstrap (resample the Block A events with replacement; 10,000 draws; seed 42).
- Gates: V1 every level has at least 100 Block A rows; V2 the flood share rises strictly from LOW to MEDIUM to HIGH; V3 the HIGH minus LOW difference has a 97.5% interval whose lower end is above 0.
- Outcomes: "LEVELS VALIDATED" when V1 to V3 pass; otherwise "LEVELS NOT VALIDATED", and the service shows no level (the advisory then uses the score and its attributions only). The outcome is recorded either way.
- Block A is a case-control sample, so a flood share there measures ranking, not risk.

## 5. Advisory contract (E2 and E3)

- Input: ward name, score, level when validated, the top three attributions, warnings and the claim limit. No coordinates and no personal data.
- Attribution: reference substitution. For each of the six features, the change in score when that feature is replaced by its training median (the training mode for land_cover), computed with the ONNX session. It is not SHAP.
- Providers in order: Gemini, then Groq, then a deterministic template. Model names and keys come from environment variables; no key is stored in the repository.
- Each provider call is limited to 8 s and the whole request to 12 s; results are cached in memory for 1 h by input hash. `/v2/score` never calls a provider.
- Output guard: a text with a probability, a percent chance or an evacuation instruction is rejected and the template is used. The advisory says it is generated from a ranking score and is not an official warning.

## 6. Evidence

- `tests/test_phase_e.py` checks these rules on known values. The E1 run will be logged to one MLflow run and read back, and its results file will be recomputed by tests, as in Phase D.

## 7. Deviations from the roadmap (register row D49)

1. The endpoint is `/v2/advisory`, beside `/v2/score` and `/v2/metrics`, not `/advisory`.
2. The advisory takes the ranking score and its level, not a probability.
3. Attributions use reference substitution computed from the ONNX model, not SHAP: the production image has no SHAP, and SHAP on the exported tree model is not verified.
4. A validated level layer is added. The roadmap names HIGH in Phase G but never defines or validates levels.
