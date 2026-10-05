"""D20 (docs/D20_PROTOCOL.md; register row D39): the committed selection follows from the committed scan and the declared rule, and the decision rules behave as declared."""
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.models import d20
from src.models.cv import FLAGS_PATH, load_layer_flags, load_training_frame, mappable_mask

ROOT = Path(__file__).resolve().parent.parent
SCAN = ROOT / "data" / "derived" / "d20_scan.csv"
SEL = json.loads((ROOT / "data" / "derived" / "d20_selection.json").read_text(encoding="utf-8"))


def _rows():
    with open(SCAN, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _existing():
    with open(ROOT / "data" / "training" / "nyando_training_v2_multidate_sets.csv", newline="", encoding="utf-8") as f:
        return [date.fromisoformat(r["scene_date"]) for r in csv.DictReader(f)]


def test_the_scan_file_is_the_one_the_selection_names():
    assert hashlib.sha256(SCAN.read_bytes()).hexdigest() == SEL["scan_sha256"] and len(_rows()) == SEL["scan_rows"]


def test_the_committed_selection_follows_from_the_scan_and_the_rule():
    elig = d20.eligible_dates(_rows(), _existing())
    chosen = d20.select_dates(elig, _existing())
    a, b = d20.split_blocks(chosen)
    assert [str(d) for d in elig] == SEL["eligible"] and [str(d) for d in chosen] == SEL["selected"]
    assert [str(d) for d in a] == SEL["block_a"] and [str(d) for d in b] == SEL["block_b"]


def test_selected_dates_respect_the_spacing_the_window_and_the_blocks():
    chosen = [date.fromisoformat(d) for d in SEL["selected"]]
    pool = sorted(chosen + _existing())
    assert all((y - x).days >= d20.MIN_GAP_DAYS for x, y in zip(pool, pool[1:]))
    assert all(d20.WINDOW[0] <= d <= d20.WINDOW[1] for d in chosen) and 0 < len(chosen) <= d20.MAX_NEW_DATES
    assert set(SEL["block_a"]).isdisjoint(SEL["block_b"]) and len(SEL["block_a"]) + len(SEL["block_b"]) == len(chosen)


def test_the_per_class_sample_size_follows_from_the_existing_date_sets():
    df, _ = load_training_frame(ROOT)
    df = df.reset_index(drop=True)
    m = np.asarray(mappable_mask(df, load_layer_flags(df, FLAGS_PATH)), dtype=bool)
    ctl = df[(df["flooded"] == 0) & (df["sample_mode"] == "date_mosaic")]
    assert d20.per_class_sample_size([float(m[g.index].mean()) for _, g in ctl.groupby("sample_set")]) == SEL["per_class"]


def test_sample_size_rule_on_arithmetic_vectors():
    assert d20.per_class_sample_size([0.9] * 10) == 40
    assert d20.per_class_sample_size([0.5] * 10) == 70
    assert d20.per_class_sample_size([1.0] + [0.6] * 9) == 60
    with pytest.raises(ValueError):
        d20.per_class_sample_size([0.2] * 10)


def test_decision_rules_on_arithmetic_vectors():
    assert d20.replicates_beyond_elevation(0.001) and not d20.replicates_beyond_elevation(0.0) and not d20.replicates_beyond_elevation(-0.02)
    assert d20.challenger_replaces(0.002, 0.01, 0.001, 0.008) and not d20.challenger_replaces(0.002, 0.01, -0.001, 0.008)
    assert not d20.challenger_replaces(-0.001, 0.01, 0.003, 0.009) and not d20.challenger_replaces(0.002, -0.01, 0.001, 0.008)
    assert d20.land_cover_survives(0.0001) and not d20.land_cover_survives(0.0)


def test_month_cluster_bootstrap_on_arithmetic_vectors():
    mean, lo, hi = d20.month_cluster_interval([0.01] * 6, ["2021-01", "2021-01", "2021-02", "2021-02", "2021-03", "2021-03"], n_boot=500)
    assert abs(mean - 0.01) < 1e-12 and abs(lo - 0.01) < 1e-12 and abs(hi - 0.01) < 1e-12
    mean, lo, hi = d20.month_cluster_interval([0.05, 0.05, -0.05, -0.05], ["a", "a", "b", "b"], n_boot=2000)
    assert mean == 0.0 and lo < 0.0 < hi


def test_protocol_exists_names_the_sealed_block_and_the_scan_hash():
    text = (ROOT / "docs" / "D20_PROTOCOL.md").read_text(encoding="utf-8")
    assert "Block B" in text and "sealed" in text and SEL["scan_sha256"] in text


def test_register_records_the_d20_protocol_and_keeps_d20_open():
    rows = {l.split("|")[1].strip(): l.strip().strip("|").split("|")[-1].strip() for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D")}
    assert rows["D39"] == "closed" and rows["D20"].startswith("open")


def test_failed_block_a_dates_are_listed_and_never_replaced_from_block_b():
    a, b = SEL["block_a"], SEL["block_b"]
    built = d20.built_block_a(a, [a[0], a[5]])
    assert built == [d for d in a if d not in (a[0], a[5])] and len(built) == len(a) - 2 and not set(built) & set(b)
    assert d20.built_block_a(a, []) == a
    with pytest.raises(ValueError):
        d20.built_block_a(a, [b[0]])


def test_block_a_below_the_floor_is_not_estimable():
    assert d20.MIN_BLOCK_A_EVENTS == 15 and d20.block_a_estimable(15) and not d20.block_a_estimable(14)


def test_protocol_states_the_failure_rule():
    text = (ROOT / "docs" / "D20_PROTOCOL.md").read_text(encoding="utf-8")
    assert "## 11. Dates that fail to build" in text and "never replaced" in text and "fewer than 15 scorable" in text


def test_shared_training_rows_on_arithmetic_vectors():
    assert d20.shared_training_rows(["a", "b", "a", "c"], ["a", "z"]) == [True, False, True, False]
    assert d20.shared_training_rows(["a", "b"], []) == [False, False]


def test_protocol_states_the_shared_location_rule():
    text = (ROOT / "docs" / "D20_PROTOCOL.md").read_text(encoding="utf-8")
    assert "## 12.1 Locations shared with Block A" in text and "refit by the same pipeline" in text and "registered artifact itself" in text
