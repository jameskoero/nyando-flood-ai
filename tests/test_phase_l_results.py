"""Phase L results (register rows D14 and D55; docs/PHASE_L_PROTOCOL.md Section 7): the stored L1 results pin the protocol, the model and the training file; every estimability decision and outcome follows from the stored numbers; the Model Card and Datasheet sections are the generated ones."""
import hashlib
import json
import re
from pathlib import Path

from src.models import phase_k as K
from src.models import phase_l as L
from src.models import phase_l_report as RP

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "docs" / "PHASE_L_RESULTS.json").read_text(encoding="utf-8"))
E1 = json.loads((ROOT / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8"))
K1 = json.loads((ROOT / "docs" / "PHASE_K_RESULTS.json").read_text(encoding="utf-8"))
CARD = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
SHEET = (ROOT / "data" / "DATA_SOURCES.md").read_text(encoding="utf-8")
FRAMES = RES["frames"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _between(text):
    assert text.count(RP.BEGIN) == 1 and text.count(RP.END) == 1
    return text.split(RP.BEGIN)[1].split(RP.END)[0].strip("\n")


def test_the_results_pin_the_protocol_the_model_and_the_training_file():
    assert RES["schema"] == 1 and RES["step"] == "L1" and RES["outcome"] == L.OUTCOME
    assert RES["protocol_sha256"] == _sha(ROOT / "docs" / "PHASE_L_PROTOCOL.md")
    onnx = sorted((ROOT / "models").glob("nyando_hgbcon_*.onnx"))
    assert len(onnx) == 1 and RES["model_sha256"] == _sha(onnx[0]) and onnx[0].name == "nyando_hgbcon_%s.onnx" % RES["model_sha256"][:12]
    assert RES["training_data_sha256"] == _sha(ROOT / "data" / "training" / "nyando_training_v2_multidate.csv")
    assert RES["model_sha256"] == E1["model_sha256"] and RES["training_data_sha256"] == E1["training_data_sha256"]


def test_the_results_carry_the_run_ids():
    assert re.fullmatch(r"[0-9a-f]{32}", RES["mlflow"]["run_id"]) and RES["mlflow"]["metrics_logged"] > 0
    assert isinstance(RES["github_run_id"], int) and re.fullmatch(r"[0-9a-f]{40}", RES["commit"])


def test_the_checks_hold_and_the_frames_are_the_e1_frames():
    c = RES["checks"]
    assert c["cutpoints_match"] is True and c["rows_kept_match"] is True
    assert c["registered_parity_max_abs_diff"] <= c["parity_tolerance"] <= 1e-9
    assert abs(c["reference_mean_auc"] - K1["e1"]["reference_mean_event_auc"]) <= 1e-9 and abs(c["reference_recorded_by_k1"] - K1["e1"]["reference_mean_event_auc"]) <= 1e-9
    assert FRAMES["primary"]["rows"] == E1["block_a"]["rows_kept"]
    assert FRAMES["secondary"]["rows"] == E1["reference"]["rows"]
    assert RES["arms"]["no_land_cover"] == [f for f in RES["arms"]["six"] if f != "land_cover"] and RES["arms"]["no_clay"] == [f for f in RES["arms"]["six"] if f != "clay_percent"]
    assert RES["fixed_params"] == {"max_depth": 4, "max_iter": 100, "min_samples_leaf": 20}


def test_every_class_decision_and_every_outcome_follows_from_the_stored_numbers():
    for frame in FRAMES.values():
        for k, e in frame["classes"].items():
            assert tuple(K.estimability(e["flood_rows"], e["control_rows"], e["events_both"])) == (e["estimable"], e["reason"]), k
        assert frame["classes_used"] == [int(k) for k in sorted(frame["classes"], key=int) if frame["classes"][k]["estimable"]]
        dw = frame["delta_w_land_cover"]
        assert frame["d14_outcome"] == (L.d14_outcome(dw["ci_low"], dw["ci_high"]) if dw else "not_estimable")
        if frame["clay"]["observed_estimable"]:
            a, p = frame["delta_t_clay_all"], frame["delta_t_clay_present"]
            assert frame["d55_outcome"] == L.d55_outcome((a["ci_low"], a["ci_high"]), (p["ci_low"], p["ci_high"]))
        else:
            assert frame["d55_outcome"] == "not_estimable"
        assert frame["d14_outcome"] in L.D14_OUTCOMES and frame["d55_outcome"] in L.D55_OUTCOMES


def test_the_counts_add_up_and_the_intervals_are_ordered():
    for frame in FRAMES.values():
        assert sum(e["flood_rows"] + e["control_rows"] for e in frame["classes"].values()) == frame["rows"]
        c = frame["clay"]
        assert c["rows_blank"] + c["rows_observed"] == frame["rows"] and c["observed_flood_rows"] + c["observed_control_rows"] == c["rows_observed"]
        dw = frame["delta_w_land_cover"]
        if dw:
            assert abs(dw["delta"] - (dw["w_a"] - dw["w_b"])) < 1e-12 and dw["ci_low"] <= dw["ci_high"] and dw["draws_used"] <= 10000
        for k in ("delta_t_land_cover", "delta_t_clay_all", "delta_t_clay_present"):
            if k in frame:
                t = frame[k]
                assert t["ci_low"] <= t["ci_high"] and t["wins"] + t["losses"] + t["ties"] == t["n_events"], k


def test_the_stored_claims_are_the_protocol_claims_for_the_primary_outcomes():
    p = FRAMES["primary"]
    assert RES["claims"] == {"d14": L.CLAIMS[p["d14_outcome"]], "d55": L.CLAIMS[p["d55_outcome"]], "limits": L.LIMITS}


def test_the_model_card_and_datasheet_sections_are_the_generated_ones():
    assert _between(CARD) == RP.render_card(RES)
    assert _between(SHEET) == RP.render_datasheet(RES)


def test_the_card_states_the_claims_the_limits_and_which_classes_the_statement_covers():
    section = _between(CARD)
    p = FRAMES["primary"]
    assert L.CLAIMS[p["d14_outcome"]] in section and L.CLAIMS[p["d55_outcome"]] in section
    assert "`%s`" % p["d14_outcome"] in section and "`%s`" % p["d55_outcome"] in section
    assert L.LIMITS in section.replace("What this does not show: the audit", "The audit") and "no lead-time or early-warning claim follows" in section and "not flood probabilities" in section and "a two-sided 95% interval" in section
    assert "Only the land-cover classes marked yes below enter W" in section
    for k in p["classes"]:
        assert "| %s |" % k in section


def test_the_bootstrap_constants_in_the_text_are_the_code_constants():
    try:
        from src.advisory import rules
    except ImportError:
        return
    assert (rules.N_BOOT, rules.ALPHA, rules.SEED) == (10000, 0.025, 42)
    assert "2.5th to 97.5th percentile of 10,000 resamples of events with seed 42" in CARD


def _row(register, key):
    rows = [l for l in register.splitlines() if l.startswith("| %s |" % key)]
    assert len(rows) == 1, key
    return rows[0]


def test_the_register_closes_d14_and_d55_on_this_file_and_opens_d56():
    register = (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8")
    p = FRAMES["primary"]
    d14, d55, d56 = _row(register, "D14"), _row(register, "D55"), _row(register, "D56")
    assert d14.rstrip().endswith("| closed |") and "outcome %s" % p["d14_outcome"] in d14 and "tests/test_phase_l_results.py" in d14 and "no such claim is made" in d14
    assert d55.rstrip().endswith("| closed |") and "outcome %s" % p["d55_outcome"] in d55 and "tests/test_phase_l_results.py" in d55
    assert d56.rstrip().endswith("| open |") and repr(K1["e1"]["reference_mean_event_auc"]) in d56 and "never a gate" in d56


def test_the_readme_ticks_phase_l_and_keeps_the_paper_open():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "- [x] **L (leakage audit):**" in readme and "- [ ] **J (remainder):**" in readme
    assert readme.index("**L (leakage audit):**") < readme.index("**J (remainder):**")
