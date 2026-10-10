"""Security baseline (docs/SECURITY_BASELINE.md; register row D48)."""
import fnmatch
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKER = (ROOT / "Dockerfile").read_text(encoding="utf-8")
BASE = (ROOT / "docs" / "SECURITY_BASELINE.md").read_text(encoding="utf-8")
NINE = {"fastapi", "uvicorn", "pydantic", "slowapi", "onnxruntime", "numpy", "pandas", "scikit-learn", "joblib"}
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _pins():
    out = {}
    for line in (ROOT / "constraints-ci.txt").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.!+\-]+)", line.strip())
        if m: out[norm(m.group(1))] = m.group(2)
    return out


def test_the_image_installs_the_ci_pins_and_runs_as_non_root():
    assert DOCKER.startswith("FROM python:3.11-slim")
    assert DOCKER.index("COPY constraints-ci.txt") < DOCKER.index("pip install") and "-c constraints-ci.txt" in DOCKER
    assert not re.search(r"[A-Za-z0-9_.\-]+==\d", DOCKER) and "python-multipart" not in DOCKER and "torch" not in DOCKER
    assert re.search(r"^USER\s+(?!root\b)\S+\s*$", DOCKER, re.M) and DOCKER.index("USER ") < DOCKER.index("CMD")


def test_the_served_packages_are_pinned_and_past_the_audited_advisories():
    block = DOCKER.split("-c constraints-ci.txt", 1)[1].split("COPY . .", 1)[0]
    names = {norm(n) for n in re.findall(r"^[ ]{4}([A-Za-z0-9_.\-]+)[ ]*\\?[ ]*$", block, re.M)}
    pins = _pins()
    assert names == NINE and names <= set(pins) and {"starlette", "anyio", "h11", "limits"} <= set(pins)
    assert tuple(int(x) for x in pins["starlette"].split(".")[:3]) >= (1, 3, 1)


def test_dockerignore_keeps_the_constraints_file_in_the_build_context():
    p = ROOT / ".dockerignore"
    rules = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")] if p.exists() else []
    assert not [r for r in rules if not r.startswith("!") and fnmatch.fnmatch("constraints-ci.txt", r.rstrip("/"))]


def test_every_response_carries_the_security_headers():
    from fastapi.testclient import TestClient
    from backend.main import SECURITY_HEADERS, app
    client = TestClient(app)
    for path, status in (("/health", 200), ("/does-not-exist", 404)):
        r = client.get(path)
        assert r.status_code == status and all(r.headers.get(k) == v for k, v in SECURITY_HEADERS.items()), path
    assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff" and SECURITY_HEADERS["X-Frame-Options"] == "DENY"
    assert "default-src 'none'" in SECURITY_HEADERS["Content-Security-Policy"]


def test_the_weekly_audit_workflow_audits_the_served_set_and_the_ci_pins():
    w = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
    assert "schedule:" in w and "workflow_dispatch:" in w and "-c constraints-ci.txt" in w and w.count("pip-audit") >= 3
    assert "timeout-minutes" in w and 'python-version: "3.11"' in w


def test_the_baseline_points_only_at_files_that_exist_and_names_its_open_items():
    controls = BASE.split("## Controls", 1)[1].split("\n## ", 1)[0]
    paths = set(re.findall(r"`([A-Za-z0-9_./\-]+\.(?:py|yml|md|txt)|Dockerfile)`", controls))
    assert len(paths) >= 8 and all((ROOT / p).exists() for p in paths), sorted(p for p in paths if not (ROOT / p).exists())
    items = BASE.split("## Open items", 1)[1]
    assert all(w in items for w in ("rate limit", "CORS", "D32", "dashboard")) and "7 advisories" in BASE and "0.0.31" in BASE and "1.3.1" in BASE


def test_the_policy_and_the_register_follow_the_baseline():
    s = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "| 1.x |" not in s and "V2" in s and "docs/SECURITY_BASELINE.md" in s
    rows = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D48 |")]
    assert len(rows) == 1 and "tests/test_security_baseline.py::test_the_image_installs_the_ci_pins_and_runs_as_non_root" in rows[0]
