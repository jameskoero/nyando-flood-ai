"""D20 Block A scoring (docs/D20_PROTOCOL.md Sections 5 to 7, 12.1 and 13; register row D20): the verdicts follow from the stored per-event AUCs and the committed rules; the files and hashes are the committed ones."""
import hashlib
import json
import re
from pathlib import Path

from src.models import d20

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "docs" / "D20_RESULTS.json").read_text(encoding="utf-8"))
SEL = json.loads((ROOT / "data" / "derived" / "d20_selection.json").read_text(encoding="utf-8"))
MAN = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
MODELS = json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
SEC13 = (ROOT / "docs" / "D20_PROTOCOL.md").read_text(encoding="utf-8").split("## 13.")[1]
SENS = RES["registered_artifact_sensitivity"]


def test_results_name_the_committed_block_a_models_and_training_data():
    b = RES["block_a"]
    assert hashlib.sha256((ROOT / b["file"]).read_bytes()).hexdigest() == b["sha256"] == MAN[b["file"]]["sha256"]
    c = RES["models"]["champion"]
    assert c["file"] in MODELS and MODELS[c["file"]]["sha256"] == c["sha256"] and MODELS[c["file"]]["status"] in ("active", "retired")
    assert RES["training_data_sha256"] in [v["sha256"] for v in MAN.values() if isinstance(v, dict) and "label_source" in v]
    assert b["scorable_mappable"] >= d20.MIN_BLOCK_A_EVENTS
    assert b["shared_locations"] > 0 and b["training_rows_dropped"] >= b["shared_locations"] and b["training_rows_used"] + b["training_rows_dropped"] == b["training_rows_total"]


def test_scored_events_are_block_a_events_only_and_block_b_is_untouched():
    a, b = {"date-" + d for d in SEL["block_a"]}, {"date-" + d for d in SEL["block_b"]}
    for fr, arms in RES["block_a_auc"].items():
        for arm, ev in arms.items():
            assert set(ev) <= a and not set(ev) & b, (fr, arm)
    for fr, ev in SENS["block_a_auc"].items():
        assert set(ev) <= a and not set(ev) & b, fr
    assert not set(RES["old_events"]["champion"]) & a


def test_every_arm_is_scored_on_the_same_events():
    for fr, arms in RES["block_a_auc"].items():
        assert set(arms) == set(d20.ARMS) and len({frozenset(e) for e in arms.values()}) == 1
        assert set(SENS["block_a_auc"][fr]) == set(arms["champion"])
    assert len(set.intersection(*[set(e) for e in RES["old_events"].values()])) >= 20


def test_the_phase_c_evaluation_was_reproduced_before_use():
    for arm, r in RES["reproduction"].items():
        assert abs(r["stored"] - r["recomputed"]) <= (1e-5 if arm == "champion" else 1e-3), arm


def test_the_environment_and_the_preflight_are_recorded():
    assert {"python", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "onnxruntime"} <= set(RES["environment"]) and all(RES["environment"].values())
    assert RES["preflight"]["onnx_vs_refit_training_rows"] <= 1e-3 and RES["preflight"]["contract"]["events"] >= RES["block_a"]["scorable_full"]


def test_verdicts_follow_from_the_stored_per_event_aucs():
    old = {a: d20.from_micro(e) for a, e in RES["old_events"].items()}
    blk = {fr: {arm: d20.from_micro(e) for arm, e in arms.items()} for fr, arms in RES["block_a_auc"].items()}
    viol = {c: RES["models"][c]["violations"] for c in d20.CHALLENGERS}
    got, want = d20.analyse(old, blk, viol), RES["analysis"]
    sens = d20.sensitivity(blk, {fr: d20.from_micro(e) for fr, e in SENS["block_a_auc"].items()})
    pairs = [(got["frames"][fr][k], want["frames"][fr][k]) for fr in got["frames"] for k in got["frames"][fr]]
    pairs += [(got["pooled"][c], want["pooled"][c]) for c in got["pooled"]]
    pairs += [(sens[fr][k], SENS["analysis"][fr][k]) for fr in sens for k in sens[fr]]
    for s, w in pairs:
        assert abs(s["mean_diff"] - w["mean_diff"]) < 1e-9 and (s["wins"], s["losses"]) == (w["wins"], w["losses"])
        assert all(abs(s[k] - w[k]) <= d20.TOLERANCE for k in ("ci_low", "ci_high", "month_ci_low", "month_ci_high"))
    v, fr = want["verdicts"], want["frames"]["mappable"]
    assert v["estimable"] == d20.block_a_estimable(v["n_scorable_block_a"]) and v["n_scorable_block_a"] == RES["block_a"]["scorable_mappable"]
    assert v["R-A"]["replicates"] == d20.replicates_beyond_elevation(fr["champion_vs_elevation"]["ci_low"])
    assert v["R-C"]["survives"] == d20.land_cover_survives(fr["land_cover_contribution"]["ci_low"])
    for c in d20.CHALLENGERS:
        ok, f, p = all(int(x) == 0 for x in viol[c].values()), fr[c + "_vs_champion"], want["pooled"][c]
        assert v["R-B"][c]["replaces"] == bool(ok and d20.challenger_replaces(f["ci_low"], f["mean_diff"], p["ci_low"], p["mean_diff"]))
    assert v["champion_stays"] == (not any(v["R-B"][c]["replaces"] for c in d20.CHALLENGERS))


def test_the_protocol_record_states_the_verdicts_that_follow_from_the_results():
    v = RES["analysis"]["verdicts"]
    assert ("the value beyond elevation only replicates" in SEC13) == bool(v["R-A"]["replicates"])
    assert ("the land_cover contribution survives" in SEC13) == bool(v["R-C"]["survives"])
    assert ("the champion stays" in SEC13) == bool(v["champion_stays"])
    assert RES["mlflow"]["run_id"] in SEC13 and "Block B" in SEC13 and "untouched" in SEC13
    assert "Section 12.1" in SEC13 and "registered artifact" in SEC13 and "sensitivity" in SEC13


def test_the_run_is_logged_and_the_script_is_the_committed_one():
    m = RES["mlflow"]
    assert re.fullmatch(r"[0-9a-f]{32}", m["run_id"]) and m["tracking_uri"].endswith("nyando-flood-ai.mlflow")
    assert hashlib.sha256((ROOT / "scripts" / "run_d20.py").read_bytes()).hexdigest() == RES["script_sha256"]


def test_register_row_d20_is_closed_with_this_test():
    row = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D20 |")][0]
    assert row.rstrip().endswith("| closed |") and "tests/test_d20_results.py::" in row
