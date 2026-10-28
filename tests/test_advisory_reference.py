"""Phase E reference values (docs/PHASE_E_ADVISORY.md Section 7; register row D49): the builder's logic on a tiny table made only for the test, and the stored reference rebuilt two independent ways from the real training file."""
import hashlib
import json
import math
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.advisory import reference
from src.advisory.attribution import FEATURES
from src.advisory.reference import REFERENCE_PATH, TRAINING_SHA256, build, load
from src.models.cv import restrict_to_mappable
from src.tracking import current_training_entry

ROOT = Path(__file__).resolve().parent.parent
HEAD = "lon,lat,flooded,elevation,slope,rainfall_3day,distance_river,clay_percent,land_cover"
ROWS = [
    ("34.0,-0.1,1,1100,1,50,100,20,40", 1), ("34.1,-0.1,1,1120,2,60,200,,40", 0), ("34.2,-0.1,0,1140,3,10,300,30,10", 0),
    ("34.3,-0.1,0,1160,4,20,400,40,10", 1), ("34.4,-0.1,0,1180,5,30,500,50,40", 0), ("34.5,-0.1,0,1200,6,60,600,60,10", 1),
]
EXPECTED = {"rows": 6, "floods": 2, "mappable": 4}


def toy_tree(root, rows=ROWS, flag_lon_shift=0.0, head=HEAD):
    (root / "data" / "training").mkdir(parents=True)
    (root / "data" / "derived").mkdir(parents=True)
    train = root / "data" / "training" / "t.csv"
    train.write_text(head + "\n" + "\n".join(r for r, _ in rows) + "\n", encoding="utf-8")
    lines = ["lon,lat,flooded,exclusion_mask"] + ["%s,%s,%s,%d" % (r.split(",")[0], r.split(",")[1], r.split(",")[2], mask) for r, mask in rows]
    flags = root / "data" / "derived" / "gfm_layer_flags.csv"
    flags.write_text("\n".join(lines).replace("34.0,", "%s," % (34.0 + flag_lon_shift), 1) + "\n", encoding="utf-8")
    return hashlib.sha256(train.read_bytes()).hexdigest()


def test_the_builder_takes_flood_rows_and_unmasked_controls_and_computes_the_reference():
    with tempfile.TemporaryDirectory() as d:
        sha = toy_tree(Path(d))
        out = build(d, training_sha256=sha, expected=EXPECTED)
    assert out["rows"] == 4 and out["median"] == {"elevation": 1130.0, "slope": 2.5, "rainfall_3day": 40.0, "distance_river": 250.0, "clay_percent": 30.0}
    assert out["n"]["clay_percent"] == 3 and out["n"]["elevation"] == 4 and out["land_cover_mode"] == 40 and out["land_cover_mode_rows"] == 3 and out["training_file"] == "data/training/t.csv"


def test_a_tie_in_land_cover_takes_the_smallest_class():
    rows = [(r.replace(",50,40", ",50,10") if r.startswith("34.4") else r, m) for r, m in ROWS]
    with tempfile.TemporaryDirectory() as d:
        sha = toy_tree(Path(d), rows)
        assert build(d, training_sha256=sha, expected=EXPECTED)["land_cover_mode"] == 10


def test_the_builder_stops_on_any_difference_from_the_verified_inputs():
    with tempfile.TemporaryDirectory() as d:
        sha = toy_tree(Path(d))
        with pytest.raises(ValueError):
            build(d, training_sha256="0" * 64, expected=EXPECTED)
        with pytest.raises(ValueError):
            build(d, training_sha256=sha, expected=dict(EXPECTED, mappable=5))
    with tempfile.TemporaryDirectory() as d:
        sha = toy_tree(Path(d), flag_lon_shift=0.5)
        with pytest.raises(ValueError):
            build(d, training_sha256=sha, expected=EXPECTED)
    with tempfile.TemporaryDirectory() as d:
        sha = toy_tree(Path(d), head=HEAD.replace("land_cover", "landcover"))
        with pytest.raises(ValueError):
            build(d, training_sha256=sha, expected=EXPECTED)


def test_the_stored_reference_is_rebuilt_exactly_from_the_real_training_file():
    stored = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert build(ROOT) == stored
    assert stored["training_sha256"] == TRAINING_SHA256 and stored["rows"] == reference.EXPECTED["mappable"] and stored["frame"] == "mappable"


def test_pandas_and_the_repositorys_own_mappable_code_agree_with_the_pure_python_build():
    stored = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    rel, entry, real_hash = current_training_entry(ROOT)
    assert real_hash == stored["training_sha256"] and Path(rel).as_posix() == stored["training_file"]
    m = restrict_to_mappable(pd.read_csv(ROOT / rel))
    assert len(m) == stored["rows"]
    for f in FEATURES:
        assert int(m[f].notna().sum()) == stored["n"][f], f
        if f != "land_cover":
            assert math.isclose(float(m[f].median()), stored["median"][f], rel_tol=1e-12, abs_tol=1e-12), f
    mode = int(m["land_cover"].mode().iloc[0])
    assert mode == stored["land_cover_mode"] and int((m["land_cover"] == mode).sum()) == stored["land_cover_mode_rows"]


def test_the_server_reads_the_reference_as_numbers_and_an_integer_class():
    ref, stored = load(), json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert set(ref) == set(FEATURES) and isinstance(ref["land_cover"], int) and ref["land_cover"] == stored["land_cover_mode"]
    assert all(isinstance(ref[f], float) and ref[f] == stored["median"][f] for f in FEATURES if f != "land_cover")


def test_the_advisory_document_names_the_reference_file_and_its_builder():
    doc = (ROOT / "docs" / "PHASE_E_ADVISORY.md").read_text(encoding="utf-8")
    assert "docs/PHASE_E_REFERENCE.json" in doc and "scripts/build_advisory_reference.py" in doc
