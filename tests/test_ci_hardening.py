"""CI hardening (docs/ROADMAP_DEVIATIONS.md D30 and D31): read-only token permissions, actions pinned by commit SHA, exact version constraints."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
CONSTRAINTS = ROOT / "constraints-ci.txt"


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins():
    out = {}
    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        s = line.split(";")[0].strip()
        if s and not s.startswith("#"):
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.!+\-]+", s), "not an exact pin: " + s
            name, version = s.split("==")
            out[_norm(name)] = version
    return out


def _names(path):
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].strip()
        if s and not s.startswith("-"):
            out.add(_norm(re.split(r"[<>=!~\[; ]", s)[0]))
    return out


def test_every_workflow_declares_read_only_token_permissions():
    assert WORKFLOWS
    for w in WORKFLOWS:
        text = w.read_text(encoding="utf-8")
        assert re.search(r"^permissions:\s*\n(?:[ ]{2}[a-z-]+: read\s*\n)+", text, re.M), w.name
        assert ": write" not in text, w.name


def test_every_action_is_pinned_to_a_full_commit_sha_with_a_version_comment():
    seen = 0
    for w in WORKFLOWS:
        for m in re.finditer(r"^\s*(?:-\s*)?uses:\s*(\S+)(.*)$", w.read_text(encoding="utf-8"), re.M):
            ref, rest = m.group(1), m.group(2)
            if ref.startswith("./"):
                continue
            seen += 1
            assert re.fullmatch(r"[\w.\-]+/[\w.\-]+(?:/[\w.\-/]+)?@[0-9a-f]{40}", ref), "%s: %s is not pinned to a commit SHA" % (w.name, ref)
            assert re.search(r"#\s*v\d", rest), "%s: %s has no version comment" % (w.name, ref)
    assert seen >= 6


def test_ci_installs_under_the_constraints_with_retries():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for s in ("-r requirements.txt", "-r requirements-onnx.txt", "-c constraints-ci.txt", "for attempt in 1 2 3", "--retries"):
        assert s in text, s


def test_constraints_pin_everything_the_requirements_name():
    pins = _pins()
    for req in ("requirements.txt", "requirements-onnx.txt"):
        names = _names(ROOT / req)
        assert names <= set(pins), "%s names packages the constraints do not pin: %s" % (req, sorted(names - set(pins)))
        for line in (ROOT / req).read_text(encoding="utf-8").splitlines():
            m = re.fullmatch(r"\s*([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.!+\-]+)\s*(?:#.*)?", line)
            if m:
                assert pins[_norm(m.group(1))] == m.group(2), "%s: %s is pinned differently in the constraints" % (req, m.group(1))


def test_constraints_carry_the_production_pins():
    pins = _pins()
    assert pins["numpy"] == "1.26.4" and pins["scikit-learn"] == "1.6.1"
    assert {"onnx", "onnxruntime", "protobuf", "pandas", "xgboost", "pytest"} <= set(pins)


def test_dependabot_covers_github_actions_and_only_existing_directories():
    text = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    ecosystems, dirs = re.findall(r'package-ecosystem:\s*"([^"]+)"', text), re.findall(r'directory:\s*"([^"]+)"', text)
    assert re.search(r"^version:\s*2\s*$", text, re.M) and "github-actions" in ecosystems
    assert len(dirs) == len(ecosystems) == len(re.findall(r'interval:\s*"(?:daily|weekly|monthly)"', text))
    assert all((ROOT / d.lstrip("/")).exists() for d in dirs)
    assert "npm" not in ecosystems or (ROOT / "frontend" / "package.json").exists()
