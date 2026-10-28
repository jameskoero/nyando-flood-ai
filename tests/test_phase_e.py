"""Phase E rules (docs/PHASE_E_PROTOCOL.md; register row D49): the cutpoints, the levels, the gates and the clustered bootstrap on known values, and the protocol text."""
from pathlib import Path
import numpy as np
import pytest
from src.advisory import rules as R

TEXT = (Path(__file__).resolve().parent.parent / "docs" / "PHASE_E_PROTOCOL.md").read_text(encoding="utf-8")
REGISTER = (Path(__file__).resolve().parent.parent / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_cutpoints_are_the_terciles_and_the_levels_follow_them():
    cuts = R.cutpoints([0, 1, 2, 3, 4, 5])
    assert cuts == pytest.approx((5 / 3, 10 / 3))
    assert [R.level_of(x, cuts) for x in (1.66, cuts[0], 3.3, cuts[1], 9)] == ["LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]


def test_level_rates_count_rows_and_flood_shares():
    r = R.level_rates(["LOW", "LOW", "HIGH", "HIGH", "HIGH"], [0, 1, 1, 1, 0])
    assert r["LOW"] == {"rows": 2, "flood": 1, "share": 0.5} and r["HIGH"]["share"] == pytest.approx(2 / 3)
    assert r["MEDIUM"] == {"rows": 0, "flood": 0, "share": None}


def test_the_gates_and_the_outcome():
    ok = {lv: {"rows": 100, "share": s} for lv, s in zip(R.LEVELS, (0.2, 0.5, 0.8))}
    assert R.gates(ok, 0.1) == {"V1": True, "V2": True, "V3": True} and R.outcome(R.gates(ok, 0.1)) == R.VALIDATED
    assert not R.gates(ok, 0.0)["V3"] and R.outcome(R.gates(ok, 0.0)) == R.NOT_VALIDATED
    flat = {lv: {"rows": 100, "share": s} for lv, s in zip(R.LEVELS, (0.2, 0.2, 0.8))}
    assert not R.gates(flat, 0.1)["V2"]
    thin = {lv: {"rows": 99 if lv == "MEDIUM" else 100, "share": s} for lv, s in zip(R.LEVELS, (0.2, 0.5, 0.8))}
    assert not R.gates(thin, 0.1)["V1"] and R.outcome(R.gates(thin, 0.1)) == R.NOT_VALIDATED


def test_the_clustered_bootstrap_is_deterministic_and_separates_clear_cases():
    ev = [e for e in "abcdefgh" for _ in range(4)]
    lv = ["HIGH", "HIGH", "LOW", "LOW"] * 8
    sep = R.cluster_bootstrap_diff(lv, [1, 1, 0, 0] * 8, ev, n_boot=2000)
    assert sep["diff"] == 1.0 and sep["ci_low"] == 1.0 and sep["draws_used"] == 2000
    mixed = [1, 1, 0, 0] * 4 + [0, 0, 1, 1] * 4      # four events favour HIGH, four favour LOW: pooled difference 0, events differ
    same = R.cluster_bootstrap_diff(lv, mixed, ev, n_boot=2000)
    assert same["diff"] == 0.0 and same["ci_low"] < 0 < same["ci_high"]
    assert R.cluster_bootstrap_diff(lv, mixed, ev, n_boot=500) == R.cluster_bootstrap_diff(lv, mixed, ev, n_boot=500)


def test_the_protocol_states_the_frozen_rules_and_the_four_deviations():
    for s in ("1/3 and 2/3 quantiles", "never tuned", "at least 100", "97.5%", "10,000 draws", "seed 42", "LEVELS VALIDATED", "LEVELS NOT VALIDATED", "reference substitution", "It is not SHAP", "ranking score", "not a flood probability"):
        assert s in TEXT or s.lower() in TEXT.lower(), s
    assert (R.N_BOOT, R.SEED, R.ALPHA, R.MIN_ROWS_PER_LEVEL, R.PROVIDERS) == (10000, 42, 0.025, 100, ("gemini", "groq"))
    assert (R.PROVIDER_TIMEOUT_S, R.TOTAL_TIMEOUT_S, R.CACHE_TTL_S) == (8, 12, 3600) and "8 s" in TEXT and "12 s" in TEXT and "1 h" in TEXT
    dev = TEXT.split("## 7. Deviations")[1]
    assert all(("%d. " % i) in dev for i in (1, 2, 3, 4)) and "/v2/advisory" in dev


def test_the_register_row_points_at_this_test_file():
    rows = [l for l in REGISTER.splitlines() if l.startswith("| D49 |")]
    assert len(rows) == 1 and "tests/test_phase_e.py::test_the_protocol_states_the_frozen_rules_and_the_four_deviations" in rows[0]


def test_the_protocol_excludes_block_a_rows_at_training_locations():
    s8 = TEXT.split("## 8. Amendment r1")[1]
    assert "Section 12.1" in s8 and "excluded from the validation" in s8 and "kept rows only" in s8 and "no E1 number was computed before" in s8
