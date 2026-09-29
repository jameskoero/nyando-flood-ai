"""src/tracking.py -- Phase B: MLflow experiment tracking on DagsHub.

Wraps MLflow so every training run (Phase C onward) logs itself the same way: algorithm,
hyperparameters, metrics, the model artifact, the exact training file's name and an independently
recomputed SHA-256, the git commit, whether the run happened on Colab or Termux, a timestamp, and
key package versions.

mlflow is imported lazily, inside functions, never at module level. Importing this module never
requires mlflow to be installed, so Termux-side tooling that only needs the rest of this repository
is never forced to build it (mlflow, and the C-extension dependencies it pulls in, are the class of
package the project's roadmap keeps off Termux/ARM64 by design).

Configuration is entirely through environment variables -- no token or URI is ever hardcoded here:
  MLFLOW_TRACKING_URI       e.g. https://dagshub.com/<user>/<repo>.mlflow (falls back to DEFAULT_TRACKING_URI)
  MLFLOW_TRACKING_USERNAME  a DagsHub username
  MLFLOW_TRACKING_PASSWORD  a DagsHub access token (never an account password)
  NYANDO_RUN_ORIGIN         optional override for "colab" / "termux" detection

The training file tracked is whichever entry in data/MANIFEST.json carries a `label_source` key --
the same convention tests/test_data_gate.py and tests/test_readme.py already use to mean "the
current, non-legacy training dataset."
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = "https://dagshub.com/jmskoero/nyando-flood-ai.mlflow"
DEFAULT_EXPERIMENT_NAME = "nyando-flood-ai"
_PACKAGES_OF_INTEREST = ("numpy", "pandas", "scikit-learn", "xgboost", "mlflow")
REQUIRED_RUN_PARAM_KEYS = (
    "algorithm", "training_data_file", "training_data_sha256",
    "git_commit_sha", "origin", "logged_at_utc",
)


class TrackingConfigError(RuntimeError):
    """Raised when tracking credentials, the manifest, or the training file are missing or invalid.
    Always raised before any network call or mlflow import is attempted, so the failure is cheap
    and the message names exactly what to fix."""


def sha256_file(path):
    """Independently recompute a file's SHA-256. Never trust a value read from a manifest."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cannot hash {path}: file does not exist")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_training_entry(repo_root=None):
    """Return (relative_path, manifest_entry, recomputed_sha256) for the one training file that
    carries a `label_source` key in data/MANIFEST.json. Raises TrackingConfigError if there is not
    exactly one such entry, or if the file's real hash no longer matches what the manifest says."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    manifest_path = root / "data" / "MANIFEST.json"
    if not manifest_path.exists():
        raise TrackingConfigError(f"{manifest_path} does not exist")
    manifest = json.loads(manifest_path.read_text())
    entries = [(k, v) for k, v in manifest.items() if isinstance(v, dict) and "label_source" in v]
    if len(entries) != 1:
        raise TrackingConfigError(
            f"expected exactly one training entry with a label_source in data/MANIFEST.json, found {len(entries)}"
        )
    rel_path, entry = entries[0]
    real_hash = sha256_file(root / rel_path)
    if real_hash != entry.get("sha256"):
        raise TrackingConfigError(
            f"{rel_path}: recomputed SHA-256 {real_hash} does not match data/MANIFEST.json's "
            f"{entry.get('sha256')}; the training file has changed since the manifest was written"
        )
    return rel_path, entry, real_hash


def git_commit_sha(repo_root=None):
    """Best-effort git commit SHA. Returns None -- never raises -- if git or the repo is unavailable."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def detect_origin():
    """"colab", "termux", or "unknown". NYANDO_RUN_ORIGIN in the environment always wins."""
    override = os.environ.get("NYANDO_RUN_ORIGIN")
    if override:
        return override
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "colab"
    if "TERMUX_VERSION" in os.environ or os.environ.get("PREFIX", "").endswith("com.termux/files/usr"):
        return "termux"
    return "unknown"


def environment_snapshot():
    """Best-effort package versions. A package that is not installed is simply omitted, never a crash."""
    versions = {"python": platform.python_version()}
    for name in _PACKAGES_OF_INTEREST:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _resolve_tracking_uri():
    """Checked first, and needs no import: a missing-credentials error must never depend on
    mlflow being installed."""
    if not os.environ.get("MLFLOW_TRACKING_USERNAME") or not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        raise TrackingConfigError(
            "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must both be set in the environment "
            "(the password is a DagsHub access token, never an account password). Nothing was logged."
        )
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _require_mlflow():
    try:
        import mlflow
    except ImportError as e:
        raise TrackingConfigError(
            "mlflow is not installed. Install it with `pip install dagshub` in Colab -- this is expected "
            "to fail on Termux/ARM64 and is meant to run in Colab instead."
        ) from e
    return mlflow


@contextlib.contextmanager
def tracked_run(run_name, algorithm, params, tags=None, experiment_name=DEFAULT_EXPERIMENT_NAME, repo_root=None):
    """Context manager: opens an MLflow run against DagsHub and logs the fixed Phase B fields
    automatically, then yields (mlflow, run) so the caller logs metrics and the model artifact
    themselves inside the `with` block.

        with tracked_run("baseline-lr", "LogisticRegression", {"C": 1.0}) as (mlflow, run):
            mlflow.log_metric("auc_roc", 0.841)
            mlflow.sklearn.log_model(model, "model")
            print(run.info.run_id)

    Raises TrackingConfigError, before anything is logged or any network call is made, if
    credentials are missing, the manifest entry cannot be found, or the training file's hash no
    longer matches the manifest.
    """
    uri = _resolve_tracking_uri()
    rel_path, entry, real_hash = current_training_entry(repo_root)
    mlflow = _require_mlflow()

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_params(params or {})
        mlflow.log_param("training_data_file", rel_path)
        mlflow.log_param("training_data_sha256", real_hash)
        mlflow.log_param("git_commit_sha", git_commit_sha(repo_root) or "unknown")
        mlflow.log_param("origin", detect_origin())
        mlflow.log_param("logged_at_utc", datetime.now(timezone.utc).isoformat())
        for k, v in environment_snapshot().items():
            mlflow.log_param(f"pkg_{k}", v)
        if tags:
            mlflow.set_tags(tags)
        yield mlflow, run


def log_run(run_name, algorithm, params, metrics, model=None, model_flavor="sklearn",
            artifact_path="model", tags=None, experiment_name=DEFAULT_EXPERIMENT_NAME, repo_root=None):
    """One-call convenience wrapper over tracked_run: logs params, metrics, and (if given) a model
    artifact in a single call, then returns the MLflow run ID."""
    with tracked_run(run_name, algorithm, params, tags=tags, experiment_name=experiment_name, repo_root=repo_root) as (mlflow, run):
        mlflow.log_metrics(metrics or {})
        if model is not None:
            getattr(mlflow, model_flavor).log_model(model, artifact_path)
        run_id = run.info.run_id
    return run_id


def fetch_run(run_id):
    """Fetch a logged run back from the tracking server via the MLflow client -- proves the round
    trip actually reached the server and came back, not just that the log_* calls didn't raise."""
    mlflow = _require_mlflow()
    return mlflow.tracking.MlflowClient().get_run(run_id)
