# V2 foundation: closure record

Stages covered: Hotfix, Phase A (data and artifact integrity), Phase B (experiment tracking) and the CI gates stage, from the V2 Engineering Hardening Roadmap (25 Sep 2026). Closed 2026-10-01.

Status key: **Done** (evidence cited), **Partial** (what is missing is stated), **Deferred** (owner phase and reason). Evidence is limited to merged pull requests, CI results, files in this repository and live API responses recorded on or before the closing date.

## Hotfix (roadmap Section 0)

| Item | Status | Evidence |
|---|---|---|
| Identify and log the served model binary | Done | Live /health on both Render hosts reports model file nyando_xgb_v1.pkl and SHA-256 de0e721c808b72730658880337f5f40cb88172f01186eb9e3908e527d6e31bb5. Both repository copies (backend/models and models) have that hash. nyando_gb_v1.pkl is a different binary (SHA-256 starts e024d9db92b7), is not served, and is recorded as legacy in models/MANIFEST.json. |
| /metrics must not return a hardcoded figure | Done (interim) | Live /metrics returns 503 and no figures. The cached read from a logged MLflow run is deferred to Phase C, when a model with a logged run exists. |
| /health reports file, model hash and training-data hash | Partial | File and model hash are live. The training-data fields are null on the live API. models/PROVENANCE.json names a legacy training file and hash, while models/MANIFEST.json records no training-data hash for the same binary. The two disagree, so neither is published. Resolved in Phase C, when the first V2 model carries a logged run. |
| /docs and /redoc disabled | Done | Live: /docs, /redoc and /openapi.json return 404 on the API host; /openapi.json returns 404 on the second host. |
| Rate limits on /predict | Partial | A shared cap of 15 per minute holds (30 Sep audit, live: bursts of 25 admitted 15, 15, 15, 13, 15). The per-client limit of 10 per minute does not hold (bursts admitted 17 to 21 of 25); the cause is not established. /advisory does not exist until Phase E. |

## Phase A: data and artifact integrity (roadmap Section 1)

| Item | Status | Evidence |
|---|---|---|
| 4,420 rows from 35 dated Sentinel-1 scenes, flood labels from Copernicus GFM | Done | data/MANIFEST.json and tests/test_data_gate.py. |
| A-Gate enforced in CI | Done | Row-count, flood-rate, repeated-value, event-date and provenance checks pass in the required data-gate check. The bare 4-feature leave-one-event-out AUC is 0.8412 against a limit of 0.90. The roadmap's clay-flatness rule was replaced by a rule that no single feature reaches AUC 0.90. |
| distance_river == 0 annotation | Partial | river_adjacent_verified exists and CI checks that it equals (distance_river == 0): 15 of 4,420 rows, all in Kabonyo/Kanyagwal. The value is derived from the distance, not an independent check; the one-time comparison with a waterway layer has not been done. Owner: before the first training run in Phase C. |
| Monotonicity test; permutation-importance audit on the full feature set | Deferred to Phase C | Both need a trained model. |
| Hash manifests recomputed independently in CI; naming lint | Done | manifest-check recomputes every recorded hash. scripts/check_manifests.py rejects _v<number> model names and requires an algorithm plus hash-prefix name. Green on the closing pull requests. |

## CI gates (roadmap Section 2)

| Item | Status | Evidence |
|---|---|---|
| data-gate, manifest-check, test | Done | Required checks on main; all three green on pull requests 14 and 15. The test job runs pytest -v with no filter and includes the retraction guard. |
| Branch protection, public repository | Done | The pull request page marks the three checks Required; the repository is public. |
| Runner and Python alignment | Done (this change) | ubuntu-24.04 is pinned in all three workflows, and CI now uses Python 3.11 like the other gates and the production Dockerfile. |
| Fork pull request safety | Partial | No workflow uses pull_request_target. The test job reads one secret and fails if it is absent. The platform behaviour of withholding secrets from fork pull requests was not tested here. |
| train-check, drift-monitor | Deferred to Phase C and F | They need a trained model and live inputs. |
| deploy workflow | Deferred | No deploy workflow exists; the hosting platforms trigger deploys. Before the first model ships, verify that Render deploys only after CI passes. |

## Phase B: experiment tracking (roadmap Section 3)

| Item | Status | Evidence |
|---|---|---|
| Tracking wrapper that refuses to log on credential or manifest mismatch | Done | src/tracking.py and tests/test_tracking.py (pull request 7). |
| First logged run, read back from the server | Done | DagsHub run smoke-bare4-logreg-loeo (run ID 4f588525288d4cb682a0d03220378aae): AUC 0.8412 on 4,420 rows at commit 3a811dc, read back with fetch_run. See docs/PHASE_B.md. |

## Corrections made during the stage

README rewritten with tests tying its claims to the repository (pull request 4). Retracted figures and unsupported wording removed from the live dashboard (5, 6), the legacy dashboard folder, the model card, the data-source notes, the concept note, two legacy metrics files and the concept-note PDF (14, 15). Correction notices added to releases v1.0.0 and v1.1.0. tests/test_retracted_claims.py now fails the suite if a retracted figure appears outside the files that record the retraction, or if a published score lacks an MLflow run ID and a training-data hash.

## Open items and owners

1. Per-client rate limit does not hold; cause unknown. Owner: Phase F; the fix must include a test.
2. Verify the 15 zero-distance rows against a waterway layer. Owner: start of Phase C.
3. models/PROVENANCE.json and models/MANIFEST.json disagree about the legacy model's training data. Owner: Phase C.
4. Dependency sets differ. Dockerfile pins fastapi 0.110.0, pydantic 2.7.1, slowapi 0.1.9, pandas 2.2.2 and scikit-learn 1.6.1. backend/requirements.txt pins fastapi 0.111.0 and pandas 2.1.4 and lists no slowapi. Root requirements.txt, which CI installs, uses version ranges for the web libraries. Which file Render installs from is unverified. Owner: before the first model ships.
5. Render runtime, build command, start command and auto-deploy setting are unverified, and there is no render.yaml. Owner: maintainer.
6. Vercel security headers (frontend/vercel.json) broke the preview build; cause unresolved. Needs the project's Build and Deployment settings. Owner: Phase F.
7. Zenodo erratum for the earlier record. Owner: Phase J.

## Constraints carried into Phase C

- Train under scikit-learn 1.6.1 and Python 3.11 (the Dockerfile pins), or prove that each pickle loads under those versions before it ships. The Colab default observed during this stage was Python 3.13.
- Any published score needs an MLflow run ID and a training-data hash, or the suite fails.
