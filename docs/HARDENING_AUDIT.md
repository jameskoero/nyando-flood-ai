# Hardening audit, 30 Sep 2026

Scope: Hotfix, Phase A, Phase B and the CI gates. Tested from a clean clone in Colab, with live requests, and from GitHub history. The served model is the legacy model and is not validated.

## Verified
- Live /health model SHA-256 de0e721c808b72730658880337f5f40cb88172f01186eb9e3908e527d6e31bb5 equals the SHA-256 of the model file in the repo.
- /docs and /openapi.json return 404 on both Render hosts; /redoc returns 404 on nyando-flood-api.onrender.com. /metrics returns 503 with an explicit unavailable body.
- /predict includes a legacy-model notice. Live: a NaN input and a 65-character ward return 422. In tests: Infinity returns 422 and a missing model returns 503.
- /predict has a per-client limit of 10 per minute and a shared limit of 15 per minute across all clients (in code). Live: five bursts of 25 requests admitted 15, 15, 15, 13 and 15.
- A pattern-based scan found no key-like strings in 120 commits of history or in the live JavaScript bundle. .env variants are gitignored (tested).
- Local full run on main: 103 passed, 1 expected failure, 4 failed. The 4 failures are Earth Engine not being authenticated in Colab; the CI test check passes.
- Vercel production builds succeeded for the merges in this audit. A preview build failed only for the commit that added frontend/vercel.json and passed on the next commit without it.
- The legacy dashboard/ folder, which still held retracted claims, was removed, and the claims test now scans every frontend folder.

## Data (V2 file)
- 4,420 rows in 35 scene groups: 1,970 flood rows and 2,450 control rows by sampling design, so the 44.6% flood share is not a real flood rate.
- Labels match the sampling design, every ward label lies inside its ward polygon, and the river flag is consistent for the 15 zero-distance rows.
- Bare 4-feature logistic baseline, leave-one-event-out AUC: pooled 0.841 (the value the A-Gate checks), per-event mean 0.910 (above the 0.90 limit), pooled 0.826 with every held-out location removed from training, leave-one-ward-out pooled 0.770.
- Label shuffle within events: mean per-event AUC 0.497 (maximum 0.533), so the pipeline does not leak labels.
- Strongest single feature is elevation, AUC 0.824. 18.9% of rows sit at a location used in more than one scene.
- clay_percent is blank on 28.1% of flood rows against 1.8% of controls. Awasi/Onjiko has only 2 flood points.

## Open (cause not established)
- Before the shared cap, bursts of 25 requests admitted 17 to 21 against a per-client limit of 10 per minute; setting X-Forwarded-For did not raise the count. The shared cap holds under the same test.
- The production Dockerfile pins fastapi 0.110.0, pydantic 2.7.1 and slowapi 0.1.9 on Python 3.11. The Colab test environment used fastapi 0.141.1, pydantic 2.13.5 and slowapi 0.1.10 on Python 3.13. The versions CI and Render use were not observed, and Render's Runtime and Start Command were not seen (no render.yaml in the repo).
- The root requirements.txt, which CI installs, has 19 packages with only minimum versions.
- Frontend security headers are not added: frontend/vercel.json made the Vercel preview build fail with a cd frontend error, and the reason is not established.
- CORS allows all origins. It was left as is because the API is public with no cookies or logins.
