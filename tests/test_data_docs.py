"""Datasheet and model card (docs/ROADMAP_DEVIATIONS.md D37 and D38): the documents match the code and the stored results."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEET = (ROOT / "data" / "DATA_SOURCES.md").read_text(encoding="utf-8")
CARD = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
RES = json.loads((ROOT / "docs" / "PHASE_C_RESULTS.json").read_text(encoding="utf-8"))
REG, ROB = RES["runs"]["registration"]["metrics"], RES["runs"]["robustness"]["metrics"]


def _consts(path):
    out = {}
    for n in ast.parse((ROOT / path).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            out[n.targets[0].id] = n.value.value
    return out


def _iv(m, p):
    return "%+.4f [%.4f, %.4f]" % (m[p + "/mean_diff"], m[p + "/ci_low"], m[p + "/ci_high"])


def test_data_sources_ids_match_the_code():
    ids = [v for p in ("src/data/raw_features.py", "src/data/terrain_features.py") for k, v in _consts(p).items() if k.endswith("_ASSET")]
    ids += [_consts("src/data/gfm_client.py")[k] for k in ("GFM_STAC_URL", "GFM_COLLECTION")]
    assert len(ids) >= 7
    for v in ids:
        assert v in SHEET, v + " is used by the code but missing from data/DATA_SOURCES.md"
    assert "WWF/HydroSHEDS/v1/Basins/hybas_8" in SHEET


def test_data_sources_has_no_v1_leftovers_and_cites_the_datasheet_standard():
    for bad in ("2,308", "SAR flood confirmed", "COPERNICUS/S1_GRD", "NASADEM", "WorldPop", "Independent Validation", "2512.13710", "ESA/WorldCover/v200"):
        assert bad not in SHEET, bad
    assert "Gebru" in SHEET and "WorldCover 10 m 2020" in SHEET and "ODbL" in SHEET


def test_model_card_numbers_equal_the_stored_results():
    want = ["%.4f" % REG["mappable/con/per_event_mean"], "%.4f" % REG["full/con/per_event_mean"], _iv(REG, "mappable/con/vs_elevation"), _iv(ROB, "buffer/con_minus_elevation"),
            _iv(ROB, "temporal/con_minus_elevation"), _iv(ROB, "perm/land_cover/within_drop"), _iv(ROB, "temporal/land_cover_contribution"), "%d" % ROB["temporal/scorable_test_events"],
            RES["registered_model"]["sha256"], RES["registered_model"]["file"].split("/")[-1], RES["runs"]["registration"]["training_data_sha256"], RES["runs"]["registration"]["id"]]
    for w in want:
        assert w in CARD, w


def test_model_card_statements_agree_with_the_signs_of_the_results():
    assert REG["mappable/con/vs_elevation/ci_low"] > 0 and "The advantage holds on the selection split" in CARD
    assert ROB["buffer/con_minus_elevation/ci_low"] <= 0 and ROB["temporal/con_minus_elevation/ci_low"] <= 0
    assert "not demonstrated under the buffered split or out of time" in CARD


def test_model_card_has_no_retracted_or_unverified_claims():
    for bad in ("AUC > 0.85", "Prepare evacuation", "Immediate action", "Physically sensible", "No significant spatial bias"):
        assert bad not in CARD, bad
    assert "Mitchell" in CARD and "not a flood probability" in CARD and "retracted" in CARD


def test_register_records_the_data_licence_decision():
    rows = {l.split("|")[1].strip(): l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D")}
    assert "MERIT" in rows["D37"] and rows["D37"].rstrip().endswith("| open |") and rows["D38"].rstrip().endswith("| closed |")


def test_every_four_decimal_figure_in_the_model_card_is_a_stored_result():
    stored = ["%.4f" % REG["mappable/con/per_event_mean"], "%.4f" % REG["full/con/per_event_mean"], _iv(REG, "mappable/con/vs_elevation"), _iv(ROB, "buffer/con_minus_elevation"),
              _iv(ROB, "temporal/con_minus_elevation"), _iv(ROB, "perm/land_cover/within_drop"), _iv(ROB, "temporal/land_cover_contribution")]
    allowed = set(re.findall(r"\b0\.\d{4}\b", " ".join(stored))) | _d20_figures() | _promotion_figures()
    found = set(re.findall(r"\b0\.\d{4}\b", CARD))
    assert found and found <= allowed, "figures in the model card that are not stored results: %s" % sorted(found - allowed)


def _d20_figures():
    """Every four-decimal figure of the stored D20 statistics (absolute values), for the model-card guard."""
    p = ROOT / "docs" / "D20_RESULTS.json"
    if not p.exists():
        return set()
    a = json.loads(p.read_text(encoding="utf-8"))["analysis"]
    stats = [s for fr in a["frames"].values() for s in fr.values()] + list(a["pooled"].values())
    return {"%.4f" % abs(s[k]) for s in stats for k in ("mean_diff", "ci_low", "ci_high")}


def _promotion_figures():
    """Every four-decimal figure of the stored promotion statistics (absolute values), for the model-card guard."""
    p = ROOT / "docs" / "PROMOTION_RESULTS.json"
    if not p.exists():
        return set()
    return {"%.4f" % abs(s[k]) for s in json.loads(p.read_text(encoding="utf-8"))["stats"].values() for k in ("mean_diff", "ci_low", "ci_high")}
