"""Promotion gate run (docs/PROMOTION_PROTOCOL.md, docs/PROMOTION_RESULTS.json; register row D41): every statistic follows from the stored per-event AUCs, every gate from the stored statistics, and the outcome from the gates."""
import hashlib
import json
import re
from pathlib import Path

from src.models import promotion as pr
from src.models.robustness import paired

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "docs" / "PROMOTION_RESULTS.json").read_text(encoding="utf-8"))
MAN = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
TOL = 0.003


def un(d):
    return {k: v / 1e6 for k, v in d.items()}


def other_beats(s):
    return s["ci_low"] > 0 and s["mean_diff"] > 0


def ref_beats(s):   # the interval of ref minus other is the mirror of the stored one
    return s["ci_high"] < 0 and s["mean_diff"] < 0


def test_the_results_name_the_committed_inputs():
    assert RES["protocol_sha256"] == hashlib.sha256((ROOT / RES["protocol"]).read_bytes()).hexdigest()
    for p, h in RES["inputs"].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
    assert RES["training_data_sha256"] in [v["sha256"] for v in MAN.values() if isinstance(v, dict) and "label_source" in v]
    assert RES["fixed_params"] == pr.FIXED_PARAMS


def test_the_phase_c_and_d20_values_reproduced_before_use():
    f, b = RES["reproduction"]["full_frame"], RES["reproduction"]["block_a_max_per_event_difference"]
    assert abs(f["champion"]["stored"] - f["champion"]["recomputed"]) <= 1e-6 and abs(f["hgb:con"]["stored"] - f["hgb:con"]["recomputed"]) <= 1e-3
    assert b["champion"] <= 1e-6 and b["hgb:con"] <= 1e-3


def test_every_statistic_follows_from_the_stored_per_event_aucs():
    assert set(RES["events"]) == set(RES["stats"])
    for k, ev in RES["events"].items():
        s, w = paired(un(ev["ref"]), un(ev["other"])), RES["stats"][k]
        assert abs(s["mean_diff"] - w["mean_diff"]) < 1e-9 and (s["wins"], s["losses"]) == (w["wins"], w["losses"]), k
        assert abs(s["ci_low"] - w["ci_low"]) <= TOL and abs(s["ci_high"] - w["ci_high"]) <= TOL, k


def test_every_gate_follows_from_the_stored_statistics():
    st, g = RES["stats"], RES["gates"]
    assert g["S1"] == (other_beats(st["new/mappable"]) and other_beats(st["new/full"]))
    assert g["S2"] == (other_beats(st["pooled/mappable"]) and other_beats(st["pooled/full"]))
    unres = {k: ref_beats(st[k]) or (k.split("/")[0] == pr.FLOOR_SUBSET and not st[k]["mean_diff"] > 0) for k in pr.REQUIRED_RUNS}
    assert all(RES["runs"][k]["unresolved"] == unres[k] for k in pr.REQUIRED_RUNS)
    assert g["S3"] == (not any(unres[k] for k in pr.REQUIRED_RUNS if k.split("/")[0] in pr.SUBSETS))
    assert g["S4"] == (not unres["buffered"]) and g["S5"] == (not unres["temporal"])
    assert g["S6"] == other_beats(st["S6/pooled_mappable"])
    assert RES["limitations"] == sorted(k for k in pr.REQUIRED_RUNS if not st[k]["mean_diff"] > 0)


def test_the_outcome_follows_the_gates():
    g = RES["gates"]
    assert set(g) == set(pr.STATISTICAL_GATES)
    if pr.build_exporter(g):
        assert RES["outcome"] is None and "exporter" in RES["next"]
    else:
        assert RES["outcome"] == pr.NOT_PROMOTED == pr.decision(g, None) and "no exporter" in RES["next"]


def test_the_run_is_logged_and_the_script_is_the_committed_one():
    m = RES["mlflow"]
    assert re.fullmatch(r"[0-9a-f]{32}", m["run_id"]) and m["tracking_uri"].endswith("nyando-flood-ai.mlflow")
    assert hashlib.sha256((ROOT / "scripts" / "run_promotion.py").read_bytes()).hexdigest() == RES["script_sha256"]


def test_register_row_d41_records_the_gate_run():
    row = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D41 |")][0]
    assert "Gate run on" in row and "docs/PROMOTION_RESULTS.json" in row and RES["mlflow"]["run_id"] in row
