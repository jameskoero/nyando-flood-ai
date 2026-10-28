"""Phase E reference values (docs/PHASE_E_ADVISORY.md Section 7): the median of each numeric feature and the most common land-cover class of the mappable training frame, built from the real training file with the standard library only. build() refuses to write anything if the file's hash, its counts, its columns or its alignment with the layer-flag file differ from the verified ones; load() is what the server reads."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from src.advisory.attribution import FEATURES

TRAINING_SHA256 = "0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a"
FLAGS = "data/derived/gfm_layer_flags.csv"
EXPECTED = {"rows": 4420, "floods": 1970, "mappable": 3475}
REFERENCE_PATH = Path(__file__).resolve().parents[2] / "docs" / "PHASE_E_REFERENCE.json"
NUMERIC = tuple(f for f in FEATURES if f != "land_cover")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _number(text):
    t = (text or "").strip()
    if t == "":
        return None
    v = float(t)
    return None if math.isnan(v) else v


def build(root, training_sha256=TRAINING_SHA256, expected=EXPECTED, flags=FLAGS):
    root = Path(root)
    found = [p for p in sorted((root / "data" / "training").glob("*.csv")) if _sha256(p) == training_sha256]
    if len(found) != 1:
        raise ValueError("expected exactly one training file with hash %s, found %d" % (training_sha256[:16], len(found)))
    path = found[0]
    rows, fl = _read(path), _read(root / flags)
    if not rows:
        raise ValueError("the training file is empty")
    missing = [c for c in ("lon", "lat", "flooded") + FEATURES if c not in rows[0]]
    if missing:
        raise ValueError("the training file lacks columns %s; it has %s" % (missing, list(rows[0])))
    if len(rows) != len(fl):
        raise ValueError("layer-flag file and training file differ in length (%d and %d)" % (len(fl), len(rows)))
    mappable, floods = [], 0
    for r, f in zip(rows, fl):
        flood = int(float(r["flooded"])) == 1
        if abs(float(r["lon"]) - float(f["lon"])) > 1e-9 or abs(float(r["lat"]) - float(f["lat"])) > 1e-9 or flood != (int(float(f["flooded"])) == 1):
            raise ValueError("layer-flag file is not aligned with the training file")
        floods += flood
        if flood or float(f["exclusion_mask"]) == 0:
            mappable.append(r)
    got = {"rows": len(rows), "floods": floods, "mappable": len(mappable)}
    if got != expected:
        raise ValueError("counts differ from the verified ones: %s against %s" % (got, expected))
    out = {"schema": 1, "frame": "mappable", "training_file": path.relative_to(root).as_posix(), "training_sha256": training_sha256, "flags_file": flags, "rows": len(mappable), "n": {}, "median": {}}
    for feature in NUMERIC:
        vals = [v for v in (_number(r[feature]) for r in mappable) if v is not None]
        if not vals:
            raise ValueError("no values for " + feature)
        out["n"][feature], out["median"][feature] = len(vals), statistics.median(vals)
    classes = Counter(int(v) for v in (_number(r["land_cover"]) for r in mappable) if v is not None)
    if not classes:
        raise ValueError("no values for land_cover")
    top = max(classes.values())
    out["n"]["land_cover"], out["land_cover_mode"], out["land_cover_mode_rows"] = sum(classes.values()), min(k for k, c in classes.items() if c == top), top
    return out


def load(path=None):
    """The reference location the attribution substitutes: medians, and the land-cover mode as an integer."""
    data = json.loads(Path(path or REFERENCE_PATH).read_text(encoding="utf-8"))
    ref = {f: float(data["median"][f]) for f in NUMERIC}
    if not all(math.isfinite(v) for v in ref.values()):
        raise ValueError("a reference value is not finite")
    ref["land_cover"] = int(data["land_cover_mode"])
    return ref
