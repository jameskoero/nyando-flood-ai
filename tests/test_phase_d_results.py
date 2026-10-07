"""Phase D Stage 2 results (docs/PHASE_D_RESULTS.json; protocol Section 9; register row D46): every mean and statistic is recomputed from the stored per-event AUCs (millionths), the selected points follow the stored choices, and the record matches the data and the README. Stage 2 informs selection; it decides nothing."""
import json
import re
from pathlib import Path
import pytest
from src.models import phase_d as D
from src.models.boosters import expand, space_for
from src.models.cv import load_training_frame
from src.models.phase_d_stats import stat

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "docs" / "PHASE_D_RESULTS.json").read_text(encoding="utf-8"))
A = R["auc_micro"]


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_record_belongs_to_the_training_file_and_to_the_logged_run():
    assert R["training_data_sha256"] == load_training_frame(ROOT)[1] and re.fullmatch(r"[0-9a-f]{40}", R["commit"])
    assert re.fullmatch(r"[0-9a-f]{32}", R["mlflow"]["run_id"]) and R["mlflow"]["metrics_logged"] > 0


def test_the_means_follow_the_stored_per_event_aucs():
    for fr, arms in A.items():
        for arm, d in arms.items():
            assert R["mean_micro"][fr][arm] == round(sum(d.values()) / len(d)), (fr, arm)


def test_the_statistics_are_recomputed_from_the_stored_aucs():
    for fr, refs in R["stats"].items():
        for ref, s in refs.items():
            a, b = (A[fr]["control"], A[fr]["mlp_fixed"]) if (fr, ref) == ("mappable", "control") else (A[fr][ref], A[fr]["mlp"])
            t = stat(a, b)
            assert (s["n_events"], s["wins"], s["losses"], s["ties"]) == (t["n_events"], t["wins"], t["losses"], t["ties"]) and s["mean_diff"] == pytest.approx(t["mean_diff"], abs=1e-9), (fr, ref)
            assert abs(s["ci_low"] - t["ci_low"]) <= 0.003 and abs(s["ci_high"] - t["ci_high"]) <= 0.003, (fr, ref)  # Monte Carlo tolerance, as in D20


def test_the_selected_points_follow_the_stored_choices():
    rt = lambda x: json.loads(json.dumps(x))
    assert rt(D.modal_point(R["chosen"]["mlp"], list(D.GRID))) == R["modal_points"]["mlp"]
    assert rt(D.modal_point(R["chosen"]["hgb"], expand(space_for("hgb")))) == R["modal_points"]["hgb"]
    assert len(R["chosen"]["mlp"]) == len(R["chosen"]["hgb"]) == R["smoke"]["outer_folds"]


def test_the_reproduction_checks_state_what_they_found_and_the_readme_tick_follows_the_file():
    for arm, c in R["checks"].items():
        if arm == "scorable_events_mappable":
            assert c == len(A["mappable"]["mlp"])
            continue
        want, tol = D.REPRODUCE[arm]
        assert (c["stored"], c["tolerance"], c["recomputed_micro"]) == (want, tol, R["mean_micro"]["mappable"][arm]) and c["reproduces"] == (abs(c["recomputed_micro"] / 1e6 - want) <= tol)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "- [x] **D. Stage 2 and conclusion:**" in readme and "docs/PHASE_D_RESULTS.json" in readme and "not promoted" in readme


def test_the_decision_follows_the_rules_and_block_b_stays_sealed():
    v, text = R["violations"], (ROOT / "docs" / "PHASE_D_PROTOCOL.md").read_text(encoding="utf-8")
    assert R["decision"]["outcome"] == D.early_decision(v["mlp"]) == D.NOT_PROMOTED and v["hgb"] == {k: 0 for k in v["hgb"]}
    f = lambda m: tuple(v[m][k] for k in ("rainfall_3day", "elevation", "distance_river", "slope"))
    assert "rainfall_3day %d, elevation %d, distance_river %d, slope %d of 200" % f("mlp") in text and "(%d, %d, %d, %d)" % f("mlp_control") in text
    assert R["modal_points"]["mlp"] == {"hidden": [64, 32], "lam": 1.0} and R["modal_points"]["hgb"] == {"max_depth": 4, "max_iter": 100, "min_samples_leaf": 20}
    assert not (ROOT / "docs" / "PHASE_D_FREEZE.json").exists() and not [p for p in (ROOT / "data").rglob("*") if "block_b" in p.name.lower()]
