"""D20 Block A (docs/D20_PROTOCOL.md Sections 6 and 11; register rows D20 and D39): the build follows the committed dates, every date is accounted for, Block B is untouched, and the files match the manifest."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import block_a as B

ROOT = Path(__file__).resolve().parent.parent
SEL = B.load_selection()
MAN = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
DATA = ROOT / "data" / "confirmatory"


def test_jobs_are_the_committed_block_a_dates_in_order():
    jobs = B.block_a_jobs(SEL)
    assert [j["anchor"].isoformat() for j in jobs] == SEL["block_a"] and all(j["mode"] == "date_mosaic" for j in jobs)


def test_a_block_b_date_is_refused():
    with pytest.raises(ValueError):
        B.block_a_jobs(dict(SEL, block_a=SEL["block_a"] + [SEL["block_b"][0]]))


def test_accounting_on_arithmetic_vectors():
    a = SEL["block_a"]
    assert B.account_for(SEL, a[:20], a[20:]) == (a[:20], a[20:])
    for built, failed in ((a[:20], a[21:]), (a, [a[0]]), (a + [SEL["block_b"][0]], []), (a[:-1], [])):
        with pytest.raises(ValueError):
            B.account_for(SEL, built, failed)


def test_scorable_events_on_arithmetic_vectors():
    df = pd.DataFrame({"event_id": ["e1"] * 60 + ["e2"] * 40, "flooded": [0] * 30 + [1] * 30 + [0] * 10 + [1] * 30})
    assert B.scorable_events(df) == ["e1"]
    assert B.scorable_events(df, [True] * 29 + [False] + [True] * 70) == []


def test_block_a_files_match_the_manifest_and_are_not_training_entries():
    mine = {k: v for k, v in MAN.items() if k.startswith("data/confirmatory/")}
    assert "data/confirmatory/nyando_block_a.csv" in mine
    for rel, e in mine.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == e["sha256"], rel
        assert "label_source" not in e, rel + " must not look like a training file"


def test_every_block_a_date_is_built_or_failed_and_block_b_is_untouched():
    df = pd.read_csv(DATA / "nyando_block_a.csv")
    fails = pd.read_csv(DATA / "nyando_block_a_failures.csv")
    built = sorted({s[len("date-"):] for s in df["sample_set"]})
    B.account_for(SEL, built, list(fails["date"]))
    assert set(df["event_date"]) == set(built) and not set(df["event_date"]) & set(SEL["block_b"])


def test_rows_follow_the_sample_size_rule_and_flags_align_with_rows():
    df = pd.read_csv(DATA / "nyando_block_a.csv")
    fl = pd.read_csv(DATA / "nyando_block_a_layer_flags.csv")
    per = df.groupby("sample_set")["flooded"].agg(["sum", "count"])
    assert set(df["flooded"].unique()) <= {0, 1} and set(df["sample_mode"]) == {"date_mosaic"}
    assert (per["sum"] <= SEL["per_class"]).all() and ((per["count"] - per["sum"]) <= SEL["per_class"]).all()
    assert len(fl) == len(df) and list(fl["row"]) == list(range(len(df))) and list(fl["flooded"]) == list(df["flooded"])
    assert set(fl["exclusion_mask"].unique()) <= {0, 1}


def test_protocol_records_the_block_a_build():
    assert "## 12. Block A build record" in (ROOT / "docs" / "D20_PROTOCOL.md").read_text(encoding="utf-8")
