"""Phase K results (register rows D52 and D53; docs/PHASE_K_PROTOCOL.md Section 8): the stored K1 results pin the protocol, the model and the training file; every estimability decision and flag follows from the stored numbers; the Model Card and Datasheet sections are the generated ones. Standard library only, so these run on the phone too."""
import csv
import hashlib
import json
import re
from pathlib import Path

from src.models import phase_k as K
from src.models import phase_k_report as RP

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "docs" / "PHASE_K_RESULTS.json").read_text(encoding="utf-8"))
E1 = json.loads((ROOT / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8"))
PROTOCOL = (ROOT / "docs" / "PHASE_K_PROTOCOL.md").read_text(encoding="utf-8")
CARD = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
SHEET = (ROOT / "data" / "DATA_SOURCES.md").read_text(encoding="utf-8")
FRAMES = RES["frames"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _between(text):
    assert text.count(RP.BEGIN) == 1 and text.count(RP.END) == 1
    return text.split(RP.BEGIN)[1].split(RP.END)[0].strip("\n")


def test_the_results_pin_the_protocol_the_model_and_the_training_file():
    assert RES["schema"] == 1 and RES["step"] == "K1" and RES["outcome"] == K.OUTCOME
    assert RES["protocol_sha256"] == _sha(ROOT / "docs" / "PHASE_K_PROTOCOL.md")
    onnx = sorted((ROOT / "models").glob("nyando_hgbcon_*.onnx"))
    assert len(onnx) == 1 and RES["model_sha256"] == _sha(onnx[0]) and onnx[0].name == "nyando_hgbcon_%s.onnx" % RES["model_sha256"][:12]
    assert RES["training_data_sha256"] == _sha(ROOT / "data" / "training" / "nyando_training_v2_multidate.csv")
    assert RES["model_sha256"] == E1["model_sha256"] and RES["training_data_sha256"] == E1["training_data_sha256"]


def test_the_results_carry_the_run_ids():
    assert re.fullmatch(r"[0-9a-f]{32}", RES["mlflow"]["run_id"]) and RES["mlflow"]["metrics_logged"] > 0
    assert isinstance(RES["github_run_id"], int) and re.fullmatch(r"[0-9a-f]{40}", RES["commit"])


def test_the_frames_are_the_e1_frames():
    assert RES["e1"]["cutpoints_match"] is True and RES["e1"]["rows_kept_match"] is True
    for k in ("low", "high"):
        assert abs(RES["e1"]["cutpoints"][k] - E1["cutpoints"][k]) < 1e-9
    assert FRAMES["primary"]["rows"] == E1["block_a"]["rows_kept"]
    assert FRAMES["secondary"]["rows"] == E1["reference"]["rows"]


def test_every_estimability_decision_and_every_flag_follows_from_the_stored_numbers():
    assert list(FRAMES["primary"]["wards"]) == list(K.WARDS) == list(FRAMES["secondary"]["wards"])
    for frame in FRAMES.values():
        for ward, e in frame["wards"].items():
            assert K.estimability(e["flood_rows"], e["control_rows"], e["events_both"]) == (e["estimable"], e["reason"]), ward
            if e["estimable"]:
                gap = (e["gap"]["ci_low"], e["gap"]["ci_high"]) if e["gap"]["ci_low"] is not None else None
                assert e["flags"] == K.flags(True, e["bands"]["ci_low"], e["auc"]["ci_low"], gap), ward
            else:
                assert e["flags"] == ["not_estimable"], ward


def test_the_counts_add_up_and_the_secondary_flood_rows_are_the_training_file_flood_rows():
    for frame in FRAMES.values():
        assert sum(e["flood_rows"] + e["control_rows"] for e in frame["wards"].values()) == frame["rows"]
    table = {m.group(1): int(m.group(3)) for m in re.finditer(r"^\| ([A-Za-z/ ]+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", PROTOCOL, re.M)}
    assert set(table) == set(K.WARDS)
    assert {w: e["flood_rows"] for w, e in FRAMES["secondary"]["wards"].items()} == table


def test_a_ward_that_is_not_estimable_carries_no_metric():
    for frame in FRAMES.values():
        for ward, e in frame["wards"].items():
            if not e["estimable"]:
                assert not ({"rates", "bands", "auc", "gap"} & set(e)), ward
    assert not FRAMES["primary"]["wards"]["Awasi/Onjiko"]["estimable"] and not FRAMES["secondary"]["wards"]["Awasi/Onjiko"]["estimable"]


def test_the_intervals_are_ordered_and_the_levels_cover_every_row():
    for frame in FRAMES.values():
        for ward, e in frame["wards"].items():
            if not e["estimable"]:
                continue
            for k, lo, hi in (("bands", "ci_low", "ci_high"), ("auc", "ci_low", "ci_high"), ("gap", "ci_low", "ci_high")):
                if e[k][lo] is not None:
                    assert e[k][lo] <= e[k][hi], (ward, k)
            assert sum(r["rows"] for r in e["rates"].values()) == e["flood_rows"] + e["control_rows"], ward


def test_the_model_card_and_datasheet_sections_are_the_generated_ones():
    assert _between(CARD) == RP.render_card(RES)
    assert _between(SHEET) == RP.render_datasheet(RES)


def test_every_flag_raised_in_the_primary_frame_is_reported_and_a_ward_that_is_not_estimable_is_stated():
    section = _between(CARD)
    for ward, e in FRAMES["primary"]["wards"].items():
        assert ward in section
        for flag in e["flags"]:
            assert "`%s`" % flag in section, (ward, flag)
        if not e["estimable"]:
            assert "%s: no claim possible" % ward in section
    assert "no early-warning claim follows" in section and "not flood probabilities" in section


def test_the_bootstrap_constants_in_the_text_are_the_code_constants():
    try:
        from src.advisory import rules
    except ImportError:
        return
    assert (rules.N_BOOT, rules.ALPHA, rules.SEED) == (10000, 0.025, 42)
    assert "2.5th to 97.5th percentile of 10,000 resamples of events with seed 42" in CARD


def test_phase_k_states_its_intervals_as_95_percent_and_the_model_card_path_is_recorded():
    assert "a two-sided 95% interval" in _between(CARD)
    assert (ROOT / "MODEL_CARD.md").is_file() and not (ROOT / "docs" / "MODEL_CARD.md").exists()
    assert "`docs/MODEL_CARD.md`" in PROTOCOL
    rows = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D53 |")]
    assert len(rows) == 1 and "tests/test_phase_k_results.py::test_phase_k_states_its_intervals_as_95_percent_and_the_model_card_path_is_recorded" in rows[0] and rows[0].rstrip().endswith("| closed |")


def test_the_register_closes_d52_and_the_readme_ticks_phase_k():
    rows = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D52 |")]
    assert len(rows) == 1 and rows[0].rstrip().endswith("| closed |")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "- [x] **K (fairness slices):**" in readme and "- [ ] **J (remainder):**" in readme and "J-K (remainder)" not in readme
