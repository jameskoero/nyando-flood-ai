"""Register row D44: tests that call the live EODC GFM catalogue run in their own job, the required test job excludes them, and the client's requests have a timeout."""
import ast
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
LIVE = {"GFMClient", "sample_case_control_points", "get_peak_flood_extent", "search_items"}


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _names(n):
    return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} | {x.attr for x in ast.walk(n) if isinstance(x, ast.Attribute)}


def test_the_required_job_excludes_live_tests_and_a_separate_job_runs_them():
    head, tail = CI.split("\n  live-data:\n")
    assert re.search(r"^  test:\n", head, re.M) and '-m "not live"' in head and "-m live" in tail and '-m "not live"' not in tail
    assert "live: " in (ROOT / "pytest.ini").read_text(encoding="utf-8")


def test_every_test_that_uses_the_live_client_carries_the_live_marker():
    bad = []
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        if p.name == Path(__file__).name:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        isfix = lambda f: any("fixture" in ast.unparse(d) for d in f.decorator_list)
        fix = {f.name for f in fns if isfix(f) and _names(f) & LIVE}
        module = any(isinstance(n, ast.Assign) and "pytestmark" in ast.unparse(n) and "live" in ast.unparse(n) for n in tree.body)
        for f in fns:
            if f.name.startswith("test_") and not isfix(f) and (_names(f) & LIVE or {a.arg for a in f.args.args} & fix):
                if not module and not any("mark.live" in ast.unparse(d) for d in f.decorator_list):
                    bad.append("%s::%s" % (p.name, f.name))
    assert not bad, "tests that call the live client need @pytest.mark.live: %s" % bad


def test_the_gfm_client_sets_a_request_timeout():
    src = (ROOT / "src" / "data" / "gfm_client.py").read_text(encoding="utf-8")
    assert re.search(r"^GFM_TIMEOUT = \(\d+, \d+\)", src, re.M) and "Client.open(stac_url, timeout=timeout)" in src


def test_no_offline_check_runs_a_live_test_in_a_subprocess():
    assert "test_case_control_sampler" not in (ROOT / "tests" / "test_ee_gating.py").read_text(encoding="utf-8")
# END OF FILE
