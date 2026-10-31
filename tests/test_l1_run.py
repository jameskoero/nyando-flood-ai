"""Phase L, step L1: the run script and its workflow are read-only, hand-started and wired to the protocol's rules; the statistics module touches no file and no network. File checks only, so these run on the phone too."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / ".github" / "workflows" / "l1.yml").read_text(encoding="utf-8")
SCRIPT = (ROOT / "scripts" / "run_phase_l.py").read_text(encoding="utf-8")
STATS = (ROOT / "src" / "models" / "phase_l_stats.py").read_text(encoding="utf-8")


def test_the_l1_workflow_is_manual_read_only_and_pinned():
    assert "workflow_dispatch:" in WORKFLOW
    for trigger in ("push:", "pull_request:", "schedule:"):
        assert trigger not in WORKFLOW
    assert re.search(r"^permissions:\n  contents: read$", WORKFLOW, re.M) and ": write" not in WORKFLOW
    uses = re.findall(r"uses:\s*(\S+)", WORKFLOW)
    assert uses and all(re.fullmatch(r".+@[0-9a-f]{40}", u) for u in uses)
    assert WORKFLOW.count("secrets.DAGSHUB_TOKEN") == 1 and "python scripts/run_phase_l.py --log-mlflow" in WORKFLOW


def test_the_l1_script_is_wired_to_the_protocol_rules():
    ast.parse(SCRIPT)
    for needle in ("phase_l_stats", "L.d14_outcome", "L.d55_outcome", "L.class_strata", "L.CLAIMS", "L.LIMITS", "cv.paired_event_bootstrap", "build_hgb(fs, True, **params)", "RESULT_JSON ", "MLFLOW_RUN ",
                   "registered_parity_max_abs_diff", 'if __name__ == "__main__":'):
        assert needle in SCRIPT, needle


def test_the_l1_script_stops_instead_of_guessing():
    for message in ("the recomputed cutpoints", "Block A keeps", "the training file is not the one E1 used", "differs from the registered artifact on Block A", "the registered model file is not the artifact named in docs/REGISTRATION.json", "is not the registered",
                    "the six features are not the ones the protocol names", "the out-of-fold arms do not cover the same rows", "DAGSHUB_TOKEN is not set"):
        assert message in SCRIPT, message
    assert "PARITY_TOLERANCE = 1e-9" in SCRIPT


def test_the_l1_script_and_statistics_write_nothing_and_the_statistics_read_nothing():
    for src in (SCRIPT, STATS):
        assert "write_text(" not in src and "write_bytes(" not in src and "open(" not in src.replace("urlopen(", "")
    for banned in ("urllib", "requests", "read_csv", "read_text(", "os.environ"):
        assert banned not in STATS, banned


def test_the_audit_arms_remove_exactly_one_feature_each():
    assert '"no_land_cover": [f for f in six if f != "land_cover"]' in SCRIPT and '"no_clay": [f for f in six if f != "clay_percent"]' in SCRIPT
