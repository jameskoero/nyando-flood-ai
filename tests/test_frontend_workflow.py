"""The frontend workflow (README, Phase F): the dashboard is linted and built on every pull request, measured with Lighthouse on mobile with the report kept as an artifact, and the workflow stays read-only with every action pinned by commit SHA."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / ".github" / "workflows" / "frontend.yml").read_text(encoding="utf-8")


def accepts(version, spec):
    for clause in spec.split("||"):
        clause = clause.strip()
        m = re.fullmatch(r"(\^|>=)(\d+)(?:\.(\d+))?(?:\.(\d+))?", clause)
        if not m:
            continue
        op, major, minor, patch = m.group(1), int(m.group(2)), int(m.group(3) or 0), int(m.group(4) or 0)
        if op == ">=" and version >= (major, minor, patch):
            return True
        if op == "^" and version[0] == major and version >= (major, minor, patch):
            return True
    return False


def test_the_workflow_is_read_only_and_every_action_is_pinned_by_sha():
    assert ": write" not in TEXT and "permissions:\n  contents: read" in TEXT
    uses = [l.strip() for l in TEXT.splitlines() if "uses:" in l]
    assert len(uses) == 5
    for l in uses:
        assert re.search(r"uses: [\w./-]+@[0-9a-f]{40} # v\d+$", l), l


def test_the_frontend_job_installs_lints_and_builds_in_that_order():
    assert TEXT.index("npm ci") < TEXT.index("npm run lint") < TEXT.index("npm run build")
    assert "pull_request:" in TEXT and "branches: [main]" in TEXT


def test_lighthouse_runs_on_mobile_gates_performance_and_keeps_the_report():
    for must in ("lighthouse@13.5.0", "--form-factor=mobile", "!(s[k]>=90)", "'accessibility'", "'seo'", "upload-artifact", "if: always()", "name: lighthouse-report"):
        assert must in TEXT, must


def test_the_engine_check_accepts_and_refuses_correctly():
    assert accepts((24, 0, 0), "^20.19.0 || ^22.13.0 || >=24") and accepts((24, 0, 0), "^20.19.0 || >=22.12.0")
    assert not accepts((24, 0, 0), "^20.19.0 || ^22.13.0") and not accepts((22, 5, 0), ">=22.12.0") and not accepts((20, 5, 0), "^20.19.0")


def test_the_node_version_satisfies_the_engines_of_the_build_tools():
    assert 'node-version: "24"' in TEXT
    lock = json.loads((ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))["packages"]
    for name in ("vite", "eslint", "@vitejs/plugin-react"):
        spec = lock["node_modules/" + name]["engines"]["node"]
        assert accepts((24, 0, 0), spec), (name, spec)


def test_the_report_folder_exists_before_lighthouse_writes_to_it():
    assert TEXT.index("mkdir -p lh") < TEXT.index("lighthouse@13.5.0 http")
