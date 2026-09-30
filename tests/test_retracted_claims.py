"""Retraction guard: retracted figures and unverified claims must not reappear,
and any published score must be traceable to a logged MLflow run."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "node_modules", "__pycache__", "dist", "build", ".venv", ".pytest_cache"}
TEXT_SUFFIXES = {".md", ".py", ".json", ".txt", ".yml", ".yaml", ".jsx", ".js",
                 ".html", ".ipynb", ".toml", ".cfg", ".ini"}
SKIP_NAMES = {"package-lock.json"}

RETRACTED = ["0.9717", "0.9022", "0.9727", "Kenya DPA", "72-hour",
             "Residents Protected", "EARLY WARNING", "UNDP/USAID",
             "0.9915", "0.9504", "0.9905", "Fully compliant", "Data Protection Act"]

# Files that may quote retracted figures because they record the retraction.
ALLOWED = {"README.md", "CHANGES.md", "docs/HARDENING_AUDIT.md",
           "tests/test_readme.py", "tests/test_retracted_claims.py"}

SCORE_KEYS = {"auc", "auc_roc", "roc_auc", "f1", "f1_score", "precision",
              "recall", "brier_score", "cv_auc_mean", "cv_mean"}


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
        paths = [ROOT / line for line in out]
    except Exception:
        paths = [p for p in ROOT.rglob("*")
                 if not set(p.relative_to(ROOT).parts) & SKIP_DIRS]
    for p in paths:
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and p.name not in SKIP_NAMES:
            yield p


def test_no_retracted_claims_outside_the_record():
    offenders = []
    for p in _files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        offenders += [rel + ": " + s for s in RETRACTED if s in text]
    assert not offenders, "retracted claims found: " + "; ".join(offenders)


def test_published_scores_are_traceable_to_a_logged_run():
    for p in _files():
        if p.name != "metrics.json":
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and SCORE_KEYS & {k.lower() for k in data}:
            assert data.get("mlflow_run_id") and data.get("training_data_sha256"), (
                p.relative_to(ROOT).as_posix()
                + " publishes scores without mlflow_run_id and training_data_sha256")
