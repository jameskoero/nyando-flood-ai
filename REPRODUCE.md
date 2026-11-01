# Reproducing Nyando Flood AI V2

This file lists what can be re-run and what needs accounts or keys. It makes no claim about results; those are in docs/ and in MODEL_CARD.md.

## 1. Checks anyone can run (no accounts)

    git clone https://github.com/jameskoero/nyando-flood-ai.git
    cd nyando-flood-ai
    python -m pip install -r requirements-gate.txt
    python -m pytest tests/test_data_gate.py -q --noconftest -rx
    python scripts/check_manifests.py

The README states the expected output: 13 passed, 1 xfailed, then manifest-check OK.

## 2. Result runs (GitHub Actions, started by hand)

The workflows e1.yml, k1.yml and l1.yml have a manual trigger (workflow_dispatch). Each installs requirements.txt and requirements-onnx.txt under constraints-ci.txt, then runs scripts/run_phase_e1.py, scripts/run_phase_k.py or scripts/run_phase_l.py with --log-mlflow, using the repository secret DAGSHUB_TOKEN. The stored results are in docs/PHASE_E_RESULTS.json, docs/PHASE_K_RESULTS.json and docs/PHASE_L_RESULTS.json.

## 3. Rebuilding the data (needs Earth Engine)

scripts/build_initial_dataset.py needs an Earth Engine account and a Cloud project. It assumes Google Colab paths (/content/nyando-flood-ai), resumes after an interruption and takes about 10 minutes. The full test suite (pytest tests/) also needs Earth Engine credentials.

## 4. Files that identify the model and data

The registered model file is models/nyando_hgbcon_5ae81ad8b030.onnx. See models/MANIFEST.json and docs/REGISTRATION.json, and run scripts/check_manifests.py. The deployed service reports the model it loaded at GET /health.

## 5. Production image

The Dockerfile installs from requirements-image.lock with pip --require-hashes (wheels only), at the versions in constraints-ci.txt. The workflow image.yml builds the image when the Dockerfile, the lock, the constraints or backend/ change. render.yaml defines the Render service nyando-flood-v2. CI test installs are version-pinned but not hash-verified (register row D32).

## 6. Advisory text (needs your own keys)

The service reads GEMINI_API_KEY, GEMINI_MODEL, GEMINI_BASE, GROQ_API_KEY, GROQ_MODEL and GROQ_URL from the environment. Supply your own values to use the advisory feature.
