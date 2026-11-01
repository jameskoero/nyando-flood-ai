"""D32 (docs/ROADMAP_DEVIATIONS.md): the production image installs from a hash-locked file, the lock matches the CI pins, and CI builds the image. Standard library only."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKER = (ROOT / "Dockerfile").read_text(encoding="utf-8")
REGISTER = (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8")
NINE = {"fastapi", "uvicorn", "pydantic", "slowapi", "onnxruntime", "numpy", "pandas", "scikit-learn", "joblib"}
PIN = r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.!+\-]+)"


def _norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def _ci_pins():
    out = {}
    for line in (ROOT / "constraints-ci.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(PIN, line)
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


def _lock():
    entries, cur = {}, None
    for line in (ROOT / "requirements-image.lock").read_text(encoding="utf-8").splitlines():
        m = re.match(PIN, line)
        if m:
            cur = _norm(m.group(1))
            entries[cur] = {"version": m.group(2), "hashes": []}
            continue
        h = re.match(r"^\s+--hash=sha256:([0-9a-f]{64})", line)
        if h:
            assert cur, line
            entries[cur]["hashes"].append(h.group(1))
    return entries


def test_the_lock_is_hash_complete_and_matches_the_ci_pins():
    lock, ci = _lock(), _ci_pins()
    names = {_norm(n) for n in re.findall(r"^([A-Za-z0-9_.\-]+)[ ]*$", (ROOT / "requirements-image.in").read_text(encoding="utf-8"), re.M)}
    assert names == NINE and NINE <= set(lock)
    assert all(v["hashes"] for v in lock.values())
    assert {k: v["version"] for k, v in lock.items()} == {k: ci[k] for k in lock}
    assert not {"torch", "python-multipart"} & set(lock)
    assert {"starlette", "anyio", "h11", "limits"} <= set(lock)
    assert tuple(int(x) for x in lock["starlette"]["version"].split(".")[:3]) >= (1, 3, 1)
    text = (ROOT / "requirements-image.lock").read_text(encoding="utf-8")
    assert not any(bad in text for bad in ("--index-url", "--extra-index-url", "git+", " @ "))


def test_the_dockerfile_installs_only_from_the_lock_with_hash_checking():
    assert DOCKER.index("COPY constraints-ci.txt requirements-image.lock") < DOCKER.index("pip install")
    lines = re.findall(r"^RUN pip install (.*)$", DOCKER, re.M)
    assert len(lines) == 1
    opts = lines[0].split()
    for need in ("--require-hashes", "--only-binary=:all:", "-c", "constraints-ci.txt", "-r", "requirements-image.lock"):
        assert need in opts, need
    assert not [o for o in opts if re.match(r"^[A-Za-z]", o) and o not in ("constraints-ci.txt", "requirements-image.lock")]


def test_ci_builds_the_image_whenever_its_inputs_change():
    w = (ROOT / ".github" / "workflows" / "image.yml").read_text(encoding="utf-8")
    for must in ("docker build", "pull_request:", "workflow_dispatch:", "Dockerfile", "requirements-image.lock", "requirements-image.in", "constraints-ci.txt", "backend/**"):
        assert must in w, must


def test_the_register_closes_d32_on_this_file():
    rows = [l for l in REGISTER.splitlines() if l.startswith("| D32 |")]
    assert len(rows) == 1
    cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    assert len(cells) == 7 and cells[6] == "closed"
    assert cells[5] == "tests/test_image_lock.py::test_the_dockerfile_installs_only_from_the_lock_with_hash_checking"
    for must in ("requirements-image.lock", "--require-hashes", "D48", "CI installs"):
        assert must in rows[0], must
