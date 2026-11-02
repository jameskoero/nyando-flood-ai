"""Phase H core: selection rules on the real Block A file, the service contract,
and the control flow of the chain with a transport stub (the stub returns
labelled STUB values and its numbers are never used in a report)."""
import json
from pathlib import Path

from src.agent import core

ROOT = Path(__file__).resolve().parents[1]
ROWS = core.load_rows()
PRIMARY, N_MAPPABLE = core.load_primary()
STUB_MODEL = {"id": "STUB"}


def fails(fn):
    try:
        fn()
    except core.ApiError:
        return True
    return False


def stub_post(calls, score=0.25, advisory_score=None, fail_first=0, status=429):
    left = {"n": fail_first}

    def post(url, payload):
        calls.append((url, payload))
        if left["n"] > 0:
            left["n"] -= 1
            return status, {"error": "stub"}
        if url.endswith("/v2/score"):
            return 200, {"score": score, "model": STUB_MODEL}
        return 200, {
            "score": score if advisory_score is None else advisory_score,
            "level": "STUB", "level_basis": "STUB BASIS", "advisory": "STUB TEXT",
            "advisory_source": "stub", "advisory_model": None, "notice": "STUB NOTICE",
            "attempts": [{"provider": "STUB"}], "providers_configured": ["stub"],
            "claim_limit": "STUB CLAIM", "score_meaning": "STUB MEANING", "model": STUB_MODEL,
        }
    return post


def test_block_a_has_the_expected_shape():
    assert len(ROWS) == 3780 and len({r["event_id"] for r in ROWS}) == 27


def test_the_primary_frame_reproduces_the_recorded_phase_e_counts():
    # recorded in the Phase E level validation: 3,062 mappable rows, 300 excluded, 2,762 kept
    assert N_MAPPABLE == 3062 and N_MAPPABLE - len(PRIMARY) == 300 and len(PRIMARY) == 2762


def test_the_primary_frame_matches_the_repo_functions():
    try:
        import numpy as np
        import pandas as pd
        from src.advisory import e1
        from src.models import cv
    except ImportError:  # no pandas in Termux; CI runs this
        return
    df, _sha = cv.load_training_frame(ROOT)
    m = cv.restrict_to_mappable(df, cv.load_layer_flags(df, cv.FLAGS_PATH))
    blk = pd.read_csv(ROOT / "data" / "confirmatory" / "nyando_block_a.csv")
    flags = cv.load_layer_flags(blk, ROOT / "data" / "confirmatory" / "nyando_block_a_layer_flags.csv")
    b = cv.restrict_to_mappable(blk, flags)
    keep = np.array(e1.keep_mask(list(cv.location_keys(b)), list(cv.location_keys(m))))
    k = b[keep]
    assert len(b) == N_MAPPABLE and len(k) == len(PRIMARY)
    assert list(zip(k["event_id"].astype(str), k["lon"], k["lat"])) == [
        (r["event_id"], float(r["lon"]), float(r["lat"])) for r in PRIMARY]
    assert list(cv.location_keys(k)) == [core.location_key(r["lon"], r["lat"]) for r in PRIMARY]


def test_the_event_is_the_one_with_the_highest_mean_rainfall():
    groups = {}
    for r in PRIMARY:
        groups.setdefault(r["event_id"], []).append(float(r["rainfall_3day"]))
    means = {e: sum(v) / len(v) for e, v in groups.items()}
    chosen = core.pick_event(PRIMARY)
    assert means[chosen] == max(means.values())


def test_each_ward_row_is_the_lower_median_of_its_event_rows():
    ev = core.pick_event(PRIMARY)
    picked = core.pick_ward_rows(PRIMARY, ev)
    assert picked
    for ward, row in picked.items():
        vals = [float(r["rainfall_3day"]) for r in PRIMARY if r["event_id"] == ev and r["ward"] == ward]
        x = float(row["rainfall_3day"])
        assert row["ward"] == ward and row["event_id"] == ev
        assert sum(v < x for v in vals) <= (len(vals) - 1) // 2 < sum(v <= x for v in vals)


def test_every_block_a_row_is_accepted_by_the_service_schema():
    try:
        from backend.main import ScoreInput
    except ImportError:  # no pydantic in Termux; CI runs this
        return
    for r in ROWS:
        ScoreInput(**core.to_payload(r))


def test_block_a_wards_are_wards_the_advisory_accepts():
    geo = json.loads((ROOT / "data" / "external" / "nyando_wards.geojson").read_text(encoding="utf-8"))
    names = {f["properties"]["ward"] for f in geo["features"]}
    assert {r["ward"] for r in ROWS} <= names


def test_the_chain_calls_each_endpoint_once_per_ward_and_says_it_is_a_replay():
    calls = []
    pl = core.plan(PRIMARY, N_MAPPABLE)
    report = core.run_replay(pl, post=stub_post(calls), generated="TEST")
    n = len(pl["picked"])
    assert [u.rsplit("/v2/", 1)[1] for u, _ in calls] == ["score"] * n + ["advisory"] * n
    assert "Replay of recorded data" in report and "not a validation" in report
    assert "not a forecast" in report and "STUB NOTICE" in report
    assert "primary frame" in report and "Providers tried" in report
    for ward in pl["picked"]:
        assert "### " + ward in report


def test_disagreeing_scores_stop_the_chain():
    assert fails(lambda: core.run_replay(core.plan(PRIMARY), post=stub_post([], advisory_score=0.5)))


def test_retries_wait_and_then_succeed_or_give_up():
    waits = []
    body = core.call(stub_post([], fail_first=2), "/v2/score", {}, sleep=waits.append, wait=7.0)
    assert body["score"] == 0.25 and waits == [7.0, 7.0]
    waits.clear()
    assert fails(lambda: core.call(stub_post([], fail_first=9), "/v2/score", {}, tries=3, sleep=waits.append))
    assert len(waits) == 2


def test_a_client_error_is_not_retried():
    waits = []
    assert fails(lambda: core.call(stub_post([], fail_first=9, status=422), "/v2/score", {}, sleep=waits.append))
    assert waits == []
