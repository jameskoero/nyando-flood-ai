# Phase B: experiment tracking (MLflow on DagsHub)

Status: complete as of 2026-09-29.

What exists
- src/tracking.py logs a fixed set of fields on every run: algorithm, hyperparameters, training-file path and SHA-256, git commit, origin, timestamp and package versions.
- Credentials come from environment variables. Nothing is logged if they are missing or if the training file no longer matches models/MANIFEST.json.
- Tracking server: the DagsHub MLflow endpoint for jmskoero/nyando-flood-ai, experiment "nyando-flood-ai".

First logged run (smoke test, 2026-09-29)
- Run name: smoke-bare4-logreg-loeo, run ID 4f588525288d4cb682a0d03220378aae
- Model: StandardScaler + LogisticRegression, leave-one-event-out CV grouped by event_id
- Features: elevation, slope, rainfall_3day, distance_river
- Data: 4,420 rows
- Metric: auc_roc_bare4_loeo = 0.8412, recomputed from the committed dataset, not carried over from an earlier number
- Code state: commit 3a811dc14bff4f5c0b544a379dd3c47d861f477b (main)
- Round trip: the run was read back from the server with fetch_run(); the returned metrics matched what was logged.

How to read this run
- It is a leakage check on a deliberately simple baseline, not a model score. The data gate requires this AUC to stay below 0.90; a bare four-feature model that scored higher would suggest the labels were derived from the features.
- It says nothing about deployment readiness. No V2 model has been trained yet (Phase C).
