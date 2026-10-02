"""Source integrity: no tracked Python file may contain a pasted notebook directive or fail to parse, and a script
may define main() and its __main__ guard at most once (a duplicated notebook paste breaks all three)."""
import ast
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MARK = "%%" + "writefile"


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _py_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        return [ROOT / p for p in out]
    except Exception:
        return [p for p in ROOT.rglob("*.py") if ".git" not in p.parts and "node_modules" not in p.parts]


def test_every_python_file_parses_and_has_no_notebook_directive():
    bad = []
    for p in _py_files():
        text = p.read_text(encoding="utf-8")
        name = p.relative_to(ROOT).as_posix()
        if MARK in text or any(line.lstrip().startswith("%%") for line in text.splitlines()):
            bad.append(name + ": notebook directive")
            continue
        try:
            ast.parse(text)
        except SyntaxError as e:
            bad.append(name + ": " + str(e))
    assert not bad, "; ".join(bad)


def test_each_script_defines_main_and_its_guard_at_most_once():
    bad = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
        guards = [n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.dump(n.test)]
        if len(mains) > 1 or len(guards) > 1:
            bad.append("%s: %d main definitions, %d guards" % (p.name, len(mains), len(guards)))
    assert not bad, "; ".join(bad)


def test_run_mappable_imports_and_logs_the_documented_metric_set():
    from src.models.cv import EVENT, LABEL, load_layer_flags, load_training_frame, restrict_to_mappable
    spec = importlib.util.spec_from_file_location("run_mappable", ROOT / "scripts" / "run_mappable.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df, _ = load_training_frame()
    mapp = restrict_to_mappable(df, load_layer_flags(df))
    g = mapp.groupby(EVENT)[LABEL].agg(["size", "sum"])
    both = sorted(g[(g["sum"] > 0) & (g["sum"] < g["size"])].index)[:6]
    sub = mapp[mapp[EVENT].isin(both)].reset_index(drop=True)
    metrics = {}
    mod.frame_metrics("t", sub, metrics)
    assert len(metrics) == 16
    assert all(np.isfinite(v) for v in metrics.values())
    assert "t/six/sel/per_event_mean" in metrics
