"""Phase K protocol (register row D52; docs/PHASE_K_PROTOCOL.md): the rules in the protocol are the rules in src/models/phase_k.py, the per-ward counts in the protocol are the counts of the training file, and a ward that cannot meet the estimability rule is not estimated."""
import csv
import re
from pathlib import Path

from src.models import phase_k as K

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "PHASE_K_PROTOCOL.md").read_text(encoding="utf-8")


def _training_counts():
    rows = list(csv.DictReader(open(ROOT / "data" / "training" / "nyando_training_v2_multidate.csv", newline="", encoding="utf-8")))
    out = {}
    for ward in K.WARDS:
        sub = [r for r in rows if r["ward"] == ward]
        flooded = [r["flooded"] in ("1", "1.0") for r in sub]
        out[ward] = (
            len(sub) - sum(flooded),
            sum(flooded),
            len({r["event_id"] for r, f in zip(sub, flooded) if not f}),
            len({r["event_id"] for r, f in zip(sub, flooded) if f}),
        )
    return out


def test_the_protocol_states_the_frozen_rules():
    for must in ("MIN_ROWS_PER_CLASS = %d" % K.MIN_ROWS_PER_CLASS, "MIN_EVENTS_BOTH = %d" % K.MIN_EVENTS_BOTH, K.OUTCOME, "before any ward-level metric", "not tuned"):
        assert must in DOC, must
    for flag in K.FLAGS:
        assert "`%s`" % flag in DOC, flag
    for ward in K.WARDS:
        assert ward in DOC, ward


def test_the_counts_in_the_protocol_are_the_counts_of_the_training_file():
    table = {m.group(1): tuple(int(x) for x in m.groups()[1:]) for m in re.finditer(r"^\| ([A-Za-z/ ]+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", DOC, re.M)}
    assert set(table) == set(K.WARDS), table
    assert table == _training_counts()


def test_a_ward_that_cannot_meet_the_rule_in_the_whole_file_is_never_estimable():
    controls, floods, events_c, events_f = _training_counts()["Awasi/Onjiko"]
    assert floods < K.MIN_ROWS_PER_CLASS and events_f < K.MIN_EVENTS_BOTH
    ok, reason = K.estimability(floods, controls, min(events_c, events_f))
    assert not ok and "flood rows" in reason


def test_the_estimability_rule_checks_each_condition():
    assert K.estimability(30, 30, 5) == (True, "")
    assert not K.estimability(29, 500, 9)[0] and not K.estimability(500, 29, 9)[0] and not K.estimability(500, 500, 4)[0]


def test_ward_counts_count_the_events_that_hold_both_classes():
    assert K.ward_counts([1, 1, 0, 0, 1], ["a", "b", "a", "c", "c"]) == (3, 2, 2)


def test_the_flags_follow_the_protocol_and_a_ward_that_is_not_estimable_gets_only_one():
    assert K.flags(False, 0.9, 0.9, (0.1, 0.2)) == ["not_estimable"]
    assert K.flags(True, 0.2, 0.7, (-0.05, 0.05)) == []
    assert K.flags(True, 0.0, 0.5, (-0.3, -0.1)) == ["bands_not_separated", "auc_not_above_chance", "differs_from_pooled"]
    assert K.flags(True, None, None, None) == ["bands_not_separated", "auc_not_above_chance"]


def test_the_register_holds_row_d52_with_this_test_as_its_verification():
    rows = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D52 |")]
    assert len(rows) == 1 and "tests/test_phase_k.py::test_the_protocol_states_the_frozen_rules" in rows[0]
