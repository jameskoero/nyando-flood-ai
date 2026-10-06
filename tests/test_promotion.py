"""Promotion rules for hgb:con (docs/PROMOTION_PROTOCOL.md; register row D41). The rule logic is tested on hand-written vectors, as tests/test_d20.py does; no data file or reported number comes from them."""
import hashlib
import json
from pathlib import Path

import pytest

from src.models import promotion as pr

ROOT = Path(__file__).resolve().parent.parent
EV = ["e%02d" % i for i in range(20)]
BASE = {e: 0.90 + 0.001 * i for i, e in enumerate(EV)}
BETTER = {e: v + 0.02 for e, v in BASE.items()}
WORSE = {e: v - 0.02 for e, v in BASE.items()}
NOISE = {e: v + (0.01 if i % 2 else -0.011) for i, (e, v) in enumerate(BASE.items())}
ZERO = {f: 0 for f in ("rainfall_3day", "elevation", "distance_river", "slope")}


def all_runs():
    return {k: (BASE, BETTER) for k in pr.REQUIRED_RUNS}


def test_r3_needs_the_challenger_to_beat_the_champion_in_every_pair():
    assert pr.r3({"a": (BASE, BETTER), "b": (BASE, BETTER)})[0]
    assert not pr.r3({"a": (BASE, BETTER), "b": (BASE, NOISE)})[0]
    with pytest.raises(ValueError):
        pr.r3({})


def test_reading_a_reverses_only_when_the_champion_beats_the_challenger():
    assert pr.run_verdict("no_zero_distance/new", BASE, WORSE)["unresolved"]
    v = pr.run_verdict("no_zero_distance/new", BASE, NOISE)
    assert not v["unresolved"] and v["limitation"] and not v["reading_B_holds"] and not v["reading_C_holds"]


def test_the_floor_check_blocks_a_win_without_a_positive_mean_advantage():
    assert pr.run_verdict("no_floor_rows/pooled", BASE, NOISE)["unresolved"]
    assert not pr.run_verdict("no_floor_rows/pooled", BASE, BETTER)["unresolved"]


def test_sensitivity_needs_every_declared_run():
    assert pr.sensitivity(all_runs())[0]
    bad = all_runs()
    bad["temporal"] = (BASE, WORSE)
    assert not pr.sensitivity(bad)[0]
    part = all_runs()
    del part["buffered"]
    with pytest.raises(ValueError):
        pr.sensitivity(part)


def test_r6_and_the_export_gates():
    assert pr.r6(BASE, BETTER) and not pr.r6(BASE, NOISE)
    assert pr.export_gates(1e-6, 1e-6, ZERO, True)
    assert not pr.export_gates(1.1e-6, 0.0, ZERO, True) and not pr.export_gates(0.0, 1.1e-6, ZERO, True)
    assert not pr.export_gates(0.0, 0.0, dict(ZERO, slope=1), True) and not pr.export_gates(0.0, 0.0, ZERO, False)
    with pytest.raises(ValueError):
        pr.export_gates(0.0, 0.0, {"slope": 0}, True)


def test_exactly_one_outcome_and_no_exporter_after_a_failed_gate():
    ok = {g: True for g in pr.STATISTICAL_GATES}
    assert pr.build_exporter(ok) and pr.decision(ok, True) == pr.PROMOTED
    assert pr.decision(ok, None) == pr.NOT_PROMOTED and pr.decision(ok, False) == pr.NOT_PROMOTED
    for g in pr.STATISTICAL_GATES:
        one = dict(ok, **{g: False})
        assert not pr.build_exporter(one) and pr.decision(one, True) == pr.NOT_PROMOTED
    assert pr.PROMOTED == "D20 PROMOTED" and pr.NOT_PROMOTED == "D20 NOT PROMOTED — EXISTING CHAMPION RETAINED"


def test_the_candidate_is_the_model_d20_tested():
    d20 = json.loads((ROOT / "docs" / "D20_RESULTS.json").read_text(encoding="utf-8"))
    assert d20["models"]["hgb:con"]["modal"] == pr.FIXED_PARAMS


def test_the_rules_are_fixed_before_any_gate_result():
    text = (ROOT / "docs" / "PROMOTION_PROTOCOL.md").read_text(encoding="utf-8")
    for s in ("S1.", "S2.", "S3.", "S4.", "S5.", "S6.", "E1.", "E2.", "E3.", "E4.", "reading A plus the floor check", "1e-6", pr.PROMOTED, pr.NOT_PROMOTED):
        assert s in text, s
    res = ROOT / "docs" / "PROMOTION_RESULTS.json"
    if res.exists():
        assert json.loads(res.read_text(encoding="utf-8"))["protocol_sha256"] == hashlib.sha256((ROOT / "docs" / "PROMOTION_PROTOCOL.md").read_bytes()).hexdigest()


def test_register_row_d41_records_the_rules():
    row = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D41 |")]
    assert len(row) == 1 and "tests/test_promotion.py::" in row[0]
