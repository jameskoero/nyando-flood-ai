"""Repository hygiene (docs/ROADMAP_DEVIATIONS.md D25 to D27): removed v1 leftovers, a README layout that covers the tracked tree, a truthful sampler docstring."""
import ast
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _tracked():
    r = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        pytest.skip("not a git checkout")
    return r.stdout.splitlines()


def _layout():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## Repository layout\s*$", text, re.M)
    assert m, "the README has no Repository layout section"
    nxt = re.search(r"^## ", text[m.end():], re.M)
    return text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]


def test_v1_figures_and_the_stray_test_package_are_gone():
    assert not (REPO / "tests" / "tests").exists()
    figures = REPO / "reports" / "figures"
    assert not figures.exists() or not list(figures.glob("*.png"))


def test_readme_layout_covers_every_tracked_folder_and_root_file():
    layout, need = _layout(), set()
    for f in _tracked():
        parts = f.split("/")
        if len(parts) == 1:
            if not f.startswith("."):
                need.add(f)
            continue
        need.add(parts[0] + "/")
        if parts[0] in ("src", "data", "docs") and len(parts) > 2:
            need.add("%s/%s/" % (parts[0], parts[1]))
    missing = sorted(p for p in need if p not in layout)
    assert not missing, "the README Repository layout does not mention: %s" % missing


def test_sampler_docstring_matches_the_exclusion_mask_audit():
    from src.models.cv import LABEL, load_layer_flags, load_training_frame, mappable_mask
    doc = ast.get_docstring(ast.parse((REPO / "src" / "data" / "case_control_sampler.py").read_text(encoding="utf-8"))) or ""
    df, _ = load_training_frame()
    mappable = np.asarray(mappable_mask(df, load_layer_flags(df)))
    controls = df[LABEL].to_numpy() == 0
    inside, total = int((controls & ~mappable).sum()), int(controls.sum())
    assert inside > 0
    assert "%d of %s controls (%.1f%%)" % (inside, format(total, ","), 100.0 * inside / total) in doc
    assert "(non-excluded)" not in doc and "independently-validated" not in doc and "not detected as flooded" in doc
