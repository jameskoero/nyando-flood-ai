"""tests/test_tracking.py -- Phase B: offline tests for src/tracking.py.

Every test here runs with no network access, no Earth Engine, and no mlflow package installed.
mlflow's own behaviour is stood in for with a small in-memory fake, injected via sys.modules,
which is what tracking.tracked_run's lazy `import mlflow` binds to. This is deliberate: these
tests must pass on a fork PR and on Termux, where mlflow is never expected to be installed.
"""

import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.tracking as tracking  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this test never touches Earth Engine."""
    yield


class FakeRun:
    def __init__(self, run_id):
        self.info = types.SimpleNamespace(run_id=run_id)


class FakeClient:
    def __init__(self, runs_by_id):
        self._runs_by_id = runs_by_id

    def get_run(self, run_id):
        return self._runs_by_id[run_id]


class FakeMlflow:
    """Records every call a real mlflow client would receive, without touching the network."""

    def __init__(self):
        self.calls = {"tracking_uri": None, "experiment": None, "params": {}, "metrics": {},
                      "tags": {}, "models": []}
        self._runs_by_id = {}
        self._counter = 0
        self.sklearn = types.SimpleNamespace(log_model=self._log_model)
        self.tracking = types.SimpleNamespace(MlflowClient=lambda: FakeClient(self._runs_by_id))

    def set_tracking_uri(self, uri):
        self.calls["tracking_uri"] = uri

    def set_experiment(self, name):
        self.calls["experiment"] = name

    def start_run(self, run_name=None):
        self._counter += 1
        run = FakeRun(f"run-{self._counter}")
        self._runs_by_id[run.info.run_id] = run
        return _FakeRunContext(run)

    def log_param(self, k, v):
        self.calls["params"][k] = v

    def log_params(self, d):
        self.calls["params"].update(d or {})

    def log_metric(self, k, v):
        self.calls["metrics"][k] = v

    def log_metrics(self, d):
        self.calls["metrics"].update(d or {})

    def set_tags(self, d):
        self.calls["tags"].update(d or {})

    def _log_model(self, model, artifact_path):
        self.calls["models"].append((model, artifact_path))


class _FakeRunContext:
    def __init__(self, run):
        self._run = run

    def __enter__(self):
        return self._run

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_repo(tmp_path):
    """A minimal repo layout: one training CSV and a data/MANIFEST.json entry naming it, with a
    real, independently computed SHA-256 -- the same shape current_training_entry expects."""
    (tmp_path / "data" / "training").mkdir(parents=True)
    csv_path = tmp_path / "data" / "training" / "nyando_training_v2_multidate.csv"
    content = b"lon,lat,flooded\n34.8,-0.2,1\n34.9,-0.3,0\n"
    csv_path.write_bytes(content)
    real_hash = hashlib.sha256(content).hexdigest()
    manifest = {"data/training/nyando_training_v2_multidate.csv": {"sha256": real_hash, "label_source": "Copernicus GFM"}}
    (tmp_path / "data" / "MANIFEST.json").write_text(json.dumps(manifest))
    return tmp_path, real_hash


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "test-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "test-token")


def test_sha256_file_matches_an_independent_computation(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"some bytes to hash" * 100)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert tracking.sha256_file(p) == expected


def test_sha256_file_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tracking.sha256_file(tmp_path / "does-not-exist.csv")


def test_current_training_entry_finds_the_right_file_and_hash(fake_repo):
    root, real_hash = fake_repo
    rel_path, entry, computed_hash = tracking.current_training_entry(repo_root=root)
    assert rel_path == "data/training/nyando_training_v2_multidate.csv"
    assert computed_hash == real_hash
    assert entry["label_source"] == "Copernicus GFM"


def test_current_training_entry_detects_a_changed_file(fake_repo):
    root, _ = fake_repo
    (root / "data/training/nyando_training_v2_multidate.csv").write_bytes(b"tampered content")
    with pytest.raises(tracking.TrackingConfigError, match="does not match"):
        tracking.current_training_entry(repo_root=root)


def test_current_training_entry_requires_exactly_one_label_source(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "MANIFEST.json").write_text(json.dumps({"a.csv": {"sha256": "x"}}))
    with pytest.raises(tracking.TrackingConfigError, match="found 0"):
        tracking.current_training_entry(repo_root=tmp_path)


def test_git_commit_sha_never_raises_even_outside_a_repo(tmp_path):
    result = tracking.git_commit_sha(repo_root=tmp_path)
    assert result is None


def test_detect_origin_respects_explicit_override(monkeypatch):
    monkeypatch.setenv("NYANDO_RUN_ORIGIN", "termux")
    assert tracking.detect_origin() == "termux"


def test_detect_origin_falls_back_to_unknown(monkeypatch):
    monkeypatch.delenv("NYANDO_RUN_ORIGIN", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    assert tracking.detect_origin() == "unknown"


def test_environment_snapshot_always_has_python_and_never_crashes():
    snap = tracking.environment_snapshot()
    assert "python" in snap and isinstance(snap["python"], str)


def test_missing_credentials_fail_clearly_and_safely(monkeypatch, fake_repo):
    root, _ = fake_repo
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    with pytest.raises(tracking.TrackingConfigError, match="MLFLOW_TRACKING_USERNAME"):
        with tracking.tracked_run("x", "LogisticRegression", {}, repo_root=root):
            pytest.fail("must not reach the body of the with-block without credentials")


def test_missing_mlflow_package_fails_clearly(monkeypatch, creds, fake_repo):
    if importlib.util.find_spec("mlflow") is not None:
        pytest.skip("mlflow is genuinely installed in this environment; this path can't be exercised")
    root, _ = fake_repo
    with pytest.raises(tracking.TrackingConfigError, match="mlflow is not installed"):
        with tracking.tracked_run("x", "LogisticRegression", {}, repo_root=root):
            pass


def test_a_run_can_be_created_and_logs_the_required_fields(fake_mlflow, creds, fake_repo, monkeypatch):
    monkeypatch.setenv("NYANDO_RUN_ORIGIN", "colab")
    root, real_hash = fake_repo
    with tracking.tracked_run("smoke-test", "LogisticRegression", {"C": 1.0}, repo_root=root) as (mlflow, run):
        assert run.info.run_id
        mlflow.log_metric("auc_roc", 0.841)
    for key in tracking.REQUIRED_RUN_PARAM_KEYS:
        assert key in fake_mlflow.calls["params"], f"required field {key!r} was not logged"
    assert fake_mlflow.calls["params"]["algorithm"] == "LogisticRegression"
    assert fake_mlflow.calls["params"]["training_data_file"] == "data/training/nyando_training_v2_multidate.csv"
    assert fake_mlflow.calls["params"]["training_data_sha256"] == real_hash
    assert fake_mlflow.calls["params"]["origin"] == "colab"
    assert fake_mlflow.calls["params"]["C"] == 1.0
    assert fake_mlflow.calls["metrics"]["auc_roc"] == 0.841
    assert fake_mlflow.calls["tracking_uri"] == tracking.DEFAULT_TRACKING_URI
    assert fake_mlflow.calls["experiment"] == tracking.DEFAULT_EXPERIMENT_NAME


def test_log_run_logs_metrics_and_the_model_artifact_in_one_call(fake_mlflow, creds, fake_repo, monkeypatch):
    monkeypatch.setenv("NYANDO_RUN_ORIGIN", "colab")
    root, real_hash = fake_repo
    dummy_model = object()
    run_id = tracking.log_run(
        run_name="one-call-test", algorithm="GradientBoosting",
        params={"n_estimators": 100}, metrics={"auc_roc": 0.9, "f1": 0.8},
        model=dummy_model, model_flavor="sklearn", repo_root=root,
    )
    assert run_id == "run-1"
    assert fake_mlflow.calls["metrics"] == {"auc_roc": 0.9, "f1": 0.8}
    assert fake_mlflow.calls["models"] == [(dummy_model, "model")]
    assert fake_mlflow.calls["params"]["training_data_sha256"] == real_hash


def test_fetch_run_round_trips_through_the_client(fake_mlflow, creds, fake_repo, monkeypatch):
    monkeypatch.setenv("NYANDO_RUN_ORIGIN", "colab")
    root, _ = fake_repo
    run_id = tracking.log_run("rt", "LogisticRegression", {}, {"auc_roc": 0.5}, repo_root=root)
    fetched = tracking.fetch_run(run_id)
    assert fetched.info.run_id == run_id


def test_no_hardcoded_credentials_in_module_source():
    src = Path(tracking.__file__).read_text(encoding="utf-8")
    assert "@" not in tracking.DEFAULT_TRACKING_URI, "the default tracking URI must not embed credentials"
    import re
    suspicious = re.findall(r'(MLFLOW_TRACKING_PASSWORD|dagshub_token|access_token)\s*=\s*["\'][^"\']{6,}["\']', src, re.I)
    assert not suspicious, f"looks like a hardcoded credential literal: {suspicious}"
