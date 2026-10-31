"""Phase L protocol (register rows D14 and D55; docs/PHASE_L_PROTOCOL.md): the rules in the protocol are the rules in src/models/phase_l.py, every cited figure is in its source, the counts are generated from the training file, and the worked example and the decision tables are pinned. Standard library only, so these run on the phone too."""
import json
import re
from pathlib import Path

from src.models import phase_k as K
from src.models import phase_l as L

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "PHASE_L_PROTOCOL.md").read_text(encoding="utf-8")
CSV = ROOT / "data" / "training" / "nyando_training_v2_multidate.csv"


def test_the_protocol_states_the_frozen_rules():
    for must in ("MIN_ROWS_PER_CLASS = %d" % K.MIN_ROWS_PER_CLASS, "MIN_EVENTS_BOTH = %d" % K.MIN_EVENTS_BOTH, "N_BOOT = 10000", "SEED = 42", "2.5th to 97.5th percentile", "no lead-time or early-warning claim"):
        assert must in DOC, must
    for name in L.D14_OUTCOMES + L.D55_OUTCOMES:
        assert "`%s`" % name in DOC, name
    for arm in L.ARMS:
        assert "`%s`" % arm in DOC, arm
    assert "{{" not in DOC and "}}" not in DOC
    sha = json.loads((ROOT / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8"))["model_sha256"]
    assert sha in DOC


def test_every_cited_figure_is_in_its_source():
    for rel, text in L.CITED:
        assert text in (ROOT / rel).read_text(encoding="utf-8"), (rel, text)
        assert text in DOC, text


def test_the_claims_in_the_protocol_are_the_claims_in_the_code():
    for key, sentence in L.CLAIMS.items():
        assert sentence in DOC, key
    assert L.LIMITS in DOC and "no causal reading" in L.LIMITS and "lead-time" in L.LIMITS


def test_the_counts_block_is_generated_from_the_training_file():
    assert DOC.count(L.BEGIN) == 1 and DOC.count(L.END) == 1
    block = DOC.split(L.BEGIN)[1].split(L.END)[0].strip("\n")
    tables = L.count_tables(CSV)
    assert block == L.render_counts(tables)
    controls, floods = tables["rows"]
    assert (controls + floods, floods, controls) == (4420, 1970, 2450)
    assert sum(sum(v) for v in tables["land_cover"].values()) == 4420 and sum(sum(v) for v in tables["clay"].values()) == 4420


def test_the_worked_example_is_five_sixths_and_ties_count_one_half():
    scores = [0.9, 0.4, 0.6, 0.5, 0.3, 0.7, 0.2]
    flooded = [1, 0, 1, 0, 1, 0, 0]
    classes = ["A", "A", "A", "A", "B", "B", "B"]
    w, pairs = L.within_class_concordance(scores, flooded, classes)
    assert pairs == 6 and abs(w - 5 / 6) < 1e-12 and "5/6" in DOC
    assert L.within_class_concordance(scores, flooded, classes, use=["B"]) == (0.5, 2)
    assert L.within_class_concordance([0.5, 0.5], [1, 0], ["A", "A"]) == (0.5, 1)
    assert L.within_class_concordance([0.5], [1], ["A"]) == (None, 0)


def test_classes_follow_the_phase_k_estimability_rule():
    classes = ["A"] * 60 + ["B"] * 59
    flooded = [1] * 30 + [0] * 30 + [1] * 29 + [0] * 30
    events = [("e%d" % (i % 5)) for i in range(60)] + [("e%d" % (i % 5)) for i in range(59)]
    s = L.class_strata(classes, flooded, events)
    assert s["A"]["estimable"] is True and s["B"]["estimable"] is False and "flood rows 29 below 30" in s["B"]["reason"]


def test_the_decision_rules_follow_the_protocol():
    assert L.d14_outcome(0.01, 0.05) == "within_class_gain" and L.d14_outcome(-0.01, 0.05) == "between_class_only"
    assert L.d14_outcome(-0.05, -0.01) == "within_class_loss" and L.d14_outcome(0.0, 0.04) == "between_class_only" and L.d14_outcome(None, 0.1) == "not_estimable"
    assert L.d55_outcome((0.01, 0.05), (0.01, 0.04)) == "value_gain"
    assert L.d55_outcome((0.01, 0.05), (-0.01, 0.02)) == "missingness_only"
    assert L.d55_outcome((-0.01, 0.05), (-0.02, 0.02)) == "no_contribution"
    assert L.d55_outcome((0.01, 0.05), (-0.05, -0.01)) == "other" and L.d55_outcome((-0.05, -0.01), (-0.02, 0.02)) == "other"
    assert L.d55_outcome(None, (0.0, 0.1)) == "not_estimable"
    assert set(L.CLAIMS) == set(L.D14_OUTCOMES) | set(L.D55_OUTCOMES)


def test_the_register_holds_d14_and_d55_with_the_protocol():
    rows = (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines()
    d14 = [l for l in rows if l.startswith("| D14 |")]
    d55 = [l for l in rows if l.startswith("| D55 |")]
    assert len(d14) == 1 and len(d55) == 1
    c14, c55 = [[c.strip() for c in l.strip().strip("|").split("|")] for l in (d14[0], d55[0])]
    assert c14[1] == "Open" and c14[6] in ("open", "closed") and "docs/PHASE_L_PROTOCOL.md" in " ".join(c14)
    assert c55[1] == "Open" and c55[6] in ("open", "closed") and "clay" in c55[2].lower()


def test_the_register_claim_about_blank_clay_holds_in_the_training_file():
    t = L.count_tables(CSV)
    (blank_ctrl, blank_flood), (_, _) = t["clay"]["blank"], t["clay"]["observed"]
    controls, floods = t["rows"]
    assert blank_flood / floods > blank_ctrl / controls
