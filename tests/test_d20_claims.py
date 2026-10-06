"""D20 claims (docs/ROADMAP_DEVIATIONS.md D40): the README, the model card, the closure record and the datasheet state what D20 found, with the stored numbers."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "docs" / "D20_RESULTS.json").read_text(encoding="utf-8"))
SEL = json.loads((ROOT / "data" / "derived" / "d20_selection.json").read_text(encoding="utf-8"))
CARD, README = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8"), (ROOT / "README.md").read_text(encoding="utf-8")
CLOSURE, SHEET = (ROOT / "docs" / "PHASE_C_CLOSURE.md").read_text(encoding="utf-8"), (ROOT / "data" / "DATA_SOURCES.md").read_text(encoding="utf-8")
V, FR, POOL = RES["analysis"]["verdicts"], RES["analysis"]["frames"]["mappable"], RES["analysis"]["pooled"]
BOTH = all(V["R-B"][c]["replaces"] for c in ("hgb:con", "xgb:con"))


def _iv(s):
    return "%+.4f [%.4f, %.4f]" % (s["mean_diff"], s["ci_low"], s["ci_high"])


def test_the_model_card_cites_the_d20_numbers_it_states():
    for s in (FR["champion_vs_elevation"], FR["land_cover_contribution"], FR["hgb:con_vs_champion"], FR["xgb:con_vs_champion"], POOL["hgb:con"], POOL["xgb:con"]):
        assert _iv(s) in CARD, _iv(s)
    assert "D20: %d events scored once" % RES["block_a"]["scorable_mappable"] in CARD


def test_the_wording_follows_the_verdicts():
    rep = bool(V["R-A"]["replicates"])
    assert ("does not replicate" in CARD) == (not rep) and ("does not replicate" in CLOSURE) == (not rep) and ("did not replicate" in README) == (not rep)
    assert ("the contribution is not shown" in CARD) == (not V["R-C"]["survives"])
    assert ("both beat it" in CARD) == BOTH and ("both boosters beat" in README) == BOTH
    assert "did not beat this model on both frames, and no sensitivity subset reversed that" not in CARD


def test_the_closure_record_carries_the_d20_update():
    assert "## Update after D20" in CLOSURE and RES["mlflow"]["run_id"] in CLOSURE and _iv(FR["champion_vs_elevation"]) in CLOSURE


def test_the_datasheet_describes_block_a_from_its_files():
    blk = pd.read_csv(ROOT / "data" / "confirmatory" / "nyando_block_a.csv")
    for s in ("{:,} rows".format(len(blk)), "{:,} floods".format(int(blk["flooded"].sum())), "%d points per class" % SEL["per_class"], "%d dates (Block B)" % len(SEL["block_b"])):
        assert s in SHEET, s
