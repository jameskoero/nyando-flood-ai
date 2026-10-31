"""D56 (docs/ROADMAP_DEVIATIONS.md): the two figures for the registered model's mean per-event AUC on the mappable frame, the evaluation behind each, and the README sentence that states them. Standard library only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
REGISTER = (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8")


def _find(obj, key):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out += _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out += _find(v, key)
    return out


def _all(rel, key):
    vals = _find(json.loads((ROOT / rel).read_text(encoding="utf-8")), key)
    assert vals, (rel, key)
    return vals


def _one(rel, key):
    vals = _all(rel, key)
    assert len(set(vals)) == 1, (rel, key, vals)
    return vals[0]


def test_the_two_figures_are_the_stored_values_and_differ_slightly():
    c = _one("docs/PHASE_C_RESULTS.json", "mappable/hgb_con/per_event_mean")
    k = _one("docs/PHASE_K_RESULTS.json", "reference_mean_event_auc")
    assert _one("docs/PHASE_L_RESULTS.json", "reference_mean_auc") == k
    assert c in _all("docs/D20_RESULTS.json", "stored")
    assert round(c, 3) == 0.943 and round(k, 3) == 0.945
    assert 0.002 < k - c < 0.0025


def test_each_figure_is_tied_to_the_code_that_produces_it():
    rb = (ROOT / "scripts" / "run_boosters.py").read_text(encoding="utf-8")
    assert "nested_scores(make, space_for(kind), df, feats" in rb and 'arm.replace(":", "_")' in rb
    assert "def nested_scores" in (ROOT / "src" / "models" / "boosters.py").read_text(encoding="utf-8")
    pr = (ROOT / "src" / "models" / "promotion.py").read_text(encoding="utf-8")
    assert 'FIXED_PARAMS = {"max_depth": 4, "max_iter": 100, "min_samples_leaf": 20}' in pr
    pp = (ROOT / "docs" / "PROMOTION_PROTOCOL.md").read_text(encoding="utf-8")
    assert "the modal grid point in `docs/D20_RESULTS.json`" in pp and "Nothing is tuned under this protocol" in pp


def test_the_readme_labels_each_figure_with_its_evaluation():
    lines = [l for l in README.splitlines() if l.startswith("- **Registered model.**")]
    assert len(lines) == 1
    line = lines[0]
    for must in ("0.943 over the 23 existing events", "(`docs/REGISTRATION.json`).",
                 "The 0.943 is the figure stored by the Phase C boosters run, which tunes hyperparameters inside each leave-one-event-out fold",
                 "the fixed registered configuration gives 0.945", "(E1, K1 and L1)", "register row D56"):
        assert must in line, must


def test_the_register_closes_d56_on_this_file():
    rows = [l for l in REGISTER.splitlines() if l.startswith("| D56 |")]
    assert len(rows) == 1
    cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    assert len(cells) == 7 and cells[6] == "closed"
    assert cells[5] == "tests/test_d56_figures.py::test_each_figure_is_tied_to_the_code_that_produces_it"
    for must in ("nested_scores", "0.9430801061503482", "0.9452450434782609", "FIXED_PARAMS", "not recomputed"):
        assert must in cells[4], must
