"""Phase E E1 results (docs/PHASE_E_RESULTS.json; docs/PHASE_E_PROTOCOL.md Sections 3, 4 and 8; register row D49): every figure is recomputed from the stored per-event counts, and the record is bound to the model, the data and the protocol it was produced with."""
import hashlib
import json
from pathlib import Path
import pytest
from src.advisory import e1
from src.advisory import rules as R

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8"))
REG = json.loads((ROOT / "docs" / "REGISTRATION.json").read_text(encoding="utf-8"))
PHASE_D = json.loads((ROOT / "docs" / "PHASE_D_RESULTS.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_record_is_bound_to_the_model_the_data_and_the_protocol_it_was_made_with():
    assert RES["schema"] == 1
    assert RES["model_sha256"] == REG["model"]["sha256"]
    assert RES["training_data_sha256"] == PHASE_D["training_data_sha256"]
    assert RES["protocol_sha256"] == hashlib.sha256((ROOT / "docs" / "PHASE_E_PROTOCOL.md").read_bytes()).hexdigest()
    assert RES["registered_params"] == REG["fixed_params"]
    assert len(RES["mlflow"]["run_id"]) == 32 and RES["provenance"]["run_id"] > 0


def test_every_figure_recomputes_from_the_per_event_counts():
    s = e1.summarize(*e1.rebuild(RES["per_event"]))
    assert s["rates"] == RES["rates"] and s["gates"] == RES["gates"] and s["outcome"] == RES["outcome"]
    assert s["bootstrap"]["draws_used"] == RES["bootstrap"]["draws_used"] == R.N_BOOT
    for k in ("diff", "ci_low", "ci_high"):
        assert s["bootstrap"][k] == pytest.approx(RES["bootstrap"][k], abs=1e-9), k


def test_the_outcome_follows_the_protocol_rules_and_is_the_recorded_one():
    assert RES["outcome"] == R.outcome(R.gates(RES["rates"], RES["bootstrap"]["ci_low"]))
    assert RES["outcome"] == R.VALIDATED and all(RES["gates"].values())


def test_the_counts_add_up():
    b = RES["block_a"]
    assert sum(v["rows"] for v in RES["rates"].values()) == b["rows_kept"] == b["rows_mappable"] - b["rows_excluded"]
    assert b["locations_mappable"] == b["locations_excluded"] + b["locations_kept"] and b["rows_total"] >= b["rows_mappable"]
    ref = RES["reference"]
    assert sum(ref["rows_per_level"].values()) == ref["rows"]
    assert max(ref["rows_per_level"].values()) - min(ref["rows_per_level"].values()) <= 0.01 * ref["rows"]
    assert RES["cutpoints"]["low"] < RES["cutpoints"]["high"]


def test_the_reference_flag_is_recorded_as_observed():
    ref = RES["reference"]
    assert ref["reproduces"] == (abs(ref["mean_event_auc_micro"] / 1e6 - ref["stored"]) <= ref["tolerance"])
