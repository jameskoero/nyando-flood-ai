"""Phase K, step K1: the run script and its workflow are read-only, hand-started and wired to the K0 rules; the statistics module touches no file and no network. No numpy here, so these run on the phone too."""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / ".github" / "workflows" / "k1.yml").read_text(encoding="utf-8")
SCRIPT = (ROOT / "scripts" / "run_phase_k.py").read_text(encoding="utf-8")
STATS = (ROOT / "src" / "models" / "phase_k_stats.py").read_text(encoding="utf-8")


def test_the_k1_workflow_is_manual_read_only_and_pinned():
    assert "workflow_dispatch:" in WORKFLOW
    for trigger in ("push:", "pull_request:", "schedule:"):
        assert trigger not in WORKFLOW
    assert re.search(r"^permissions:\n  contents: read$", WORKFLOW, re.M)
    assert ": write" not in WORKFLOW
    uses = re.findall(r"uses:\s*(\S+)", WORKFLOW)
    assert uses and all(re.fullmatch(r".+@[0-9a-f]{40}", u) for u in uses)
    assert WORKFLOW.count("secrets.DAGSHUB_TOKEN") == 1
    assert "python scripts/run_phase_k.py --log-mlflow" in WORKFLOW


def test_the_k1_script_is_wired_to_the_k0_rules_and_the_e1_frames():
    ast.parse(SCRIPT)
    for needle in ("phase_k_stats", "K.OUTCOME", "K.WARDS", "RESULT_JSON ", "MLFLOW_RUN ", "cutpoints", "rows_kept", "training_data_sha256", "protocol_sha256", 'if __name__ == "__main__":'):
        assert needle in SCRIPT, needle


def test_the_k1_script_stops_instead_of_guessing():
    for message in ("the recomputed cutpoints", "Block A keeps", "the training file is not the one E1 used", "ward names the protocol does not list", "DAGSHUB_TOKEN is not set"):
        assert message in SCRIPT, message


def test_the_k1_script_and_statistics_write_nothing_and_the_statistics_read_nothing():
    for src in (SCRIPT, STATS):
        assert "write_text(" not in src and "write_bytes(" not in src and "open(" not in src.replace("urlopen(", "")
    for banned in ("urllib", "requests", "read_csv", "read_text(", "os.environ"):
        assert banned not in STATS, banned


def test_the_statistics_use_the_e1_constants_and_the_e1_bootstrap():
    for needle in ("R.N_BOOT", "R.SEED", "R.ALPHA", "R.cluster_bootstrap_diff", "K.estimability", "K.flags", "K.ward_counts"):
        assert needle in STATS, needle
