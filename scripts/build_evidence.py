"""scripts/build_evidence.py -- build frontend/public/evidence.json from the stored result files.

Every research figure the dashboard shows is read at runtime from that one generated file, so no
metric is typed into frontend code (roadmap Phase F: no hard-coded metric). tests/test_dashboard_evidence.py
fails if the committed file ever differs from a fresh build. Standard library only; no network.

    python scripts/build_evidence.py           # write frontend/public/evidence.json
    python scripts/build_evidence.py --check   # exit 1 if the committed file is stale
"""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("frontend") / "public" / "evidence.json"
TRAIN_CSV = "data/training/nyando_training_v2_multidate.csv"
BLOCK_A_CSV = "data/confirmatory/nyando_block_a.csv"
BANDS = ("LOW", "MEDIUM", "HIGH")

DISCLAIMER = ("Observed flooded shares within the case-control evaluation sample. These figures are not "
              "basin-wide flood probabilities, calibrated probabilities or estimates of flood prevalence.")

FEATURES = [
    {"key": "rainfall_3day", "label": "Three-day rainfall", "direction": "increases"},
    {"key": "elevation", "label": "Elevation", "direction": "decreases"},
    {"key": "slope", "label": "Slope", "direction": "decreases"},
    {"key": "distance_river", "label": "Distance to mapped surface water", "direction": "decreases"},
    {"key": "clay_percent", "label": "Clay content", "direction": "unconstrained"},
    {"key": "land_cover", "label": "Land cover", "direction": "unconstrained"},
]

DATASETS = [
    {"name": "Copernicus Global Flood Monitoring (GFM)", "org": "Copernicus Emergency Management Service, EODC",
     "role": "Flood extent labels"},
    {"name": "Sentinel-1", "org": "ESA, Copernicus programme", "role": "Radar Earth observation behind the GFM labels"},
    {"name": "CHIRPS", "org": "Climate Hazards Center, UC Santa Barbara", "role": "Rainfall"},
    {"name": "MERIT Hydro", "org": "University of Tokyo", "role": "Elevation, slope and terrain layers"},
    {"name": "JRC Global Surface Water", "org": "European Commission Joint Research Centre",
     "role": "Surface-water information (see the project datasheet)"},
    {"name": "SoilGrids", "org": "ISRIC World Soil Information", "role": "Clay content"},
    {"name": "ESA WorldCover (2020 map)", "org": "European Space Agency", "role": "Land cover"},
]

LIMITS = [
    "Scores are ranking scores from a case-control model. They are not flood probabilities.",
    "The advantage over elevation alone is not shown under a 1 km spatial buffer or an out-of-time split.",
    "All labels come from one satellite product and were not compared with an independent flood record.",
    "Points inside the GFM exclusion mask cannot be labelled flooded by construction, so the headline frame excludes them.",
    "Wards rest on few events, and one ward cannot be estimated.",
    "Land cover is a single 2020 map applied to every scene date.",
    "No forecast and no warning lead time is claimed. The output is not an official warning.",
]

NOT_DONE = [
    "Block B was never built or scored.",
    "The physics-constrained neural network was not promoted. The constrained gradient-boosting model stays registered.",
    "Early-warning skill is not demonstrated, and SMS alerts are not built (register row D45).",
]


def load(root, rel):
    return json.loads((Path(root) / rel).read_text(encoding="utf-8"))


def read_csv(root, rel):
    with open(Path(root) / rel, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def dates(rows):
    found = sorted({r["event_date"][:10] for r in rows if r.get("event_date")})
    return found


def flooded(row):
    return str(row["flooded"]).strip().lower() in ("1", "1.0", "true")


def quality(root):
    rows = read_csv(root, TRAIN_CSV)
    need = {"flooded", "clay_percent", "elevation", "event_date"}
    missing = need - set(rows[0])
    if missing:
        raise KeyError("training file lacks columns: " + ", ".join(sorted(missing)))
    flood_rows = [r for r in rows if flooded(r)]
    blank = [r for r in rows if r["clay_percent"].strip() == ""]
    floor = min(float(r["elevation"]) for r in rows if r["elevation"].strip() != "")
    at_floor = [r for r in rows if r["elevation"].strip() != "" and float(r["elevation"]) == floor]
    return rows, {
        "rows": len(rows),
        "clay_blank_rows": len(blank),
        "clay_blank_flood_rows": sum(flooded(r) for r in blank),
        "clay_blank_share_of_flood": len([r for r in blank if flooded(r)]) / max(1, len(flood_rows)),
        "clay_blank_share_of_controls": len([r for r in blank if not flooded(r)]) / max(1, len(rows) - len(flood_rows)),
        "elevation_floor_rows": len(at_floor),
        "elevation_floor_flooded_share": sum(flooded(r) for r in at_floor) / max(1, len(at_floor)),
    }


def geometry(root):
    gj = load(root, "data/external/nyando_wards.geojson")
    polys = {}
    for feat in gj["features"]:
        geom = feat["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        polys[feat["properties"]["ward"]] = parts
    pts = [pt for parts in polys.values() for poly in parts for ring in poly for pt in ring]
    lon0, lon1 = min(p[0] for p in pts), max(p[0] for p in pts)
    lat0, lat1 = min(p[1] for p in pts), max(p[1] for p in pts)
    k = math.cos(math.radians((lat0 + lat1) / 2))
    scale = 600.0 / ((lon1 - lon0) * k)
    height = (lat1 - lat0) * scale

    def xy(pt):
        return round((pt[0] - lon0) * k * scale, 1), round((lat1 - pt[1]) * scale, 1)

    wards = {}
    for name, parts in polys.items():
        d, xs, ys = [], [], []
        for poly in parts:
            for ring in poly:
                pr = [xy(p) for p in ring]
                xs += [p[0] for p in pr]
                ys += [p[1] for p in pr]
                d.append("M" + " L".join("%s,%s" % p for p in pr) + " Z")
        wards[name] = {"d": " ".join(d), "cx": round((min(xs) + max(xs)) / 2, 1), "cy": round((min(ys) + max(ys)) / 2, 1)}
    return {
        "viewBox": [600, round(height, 1)],
        "km_per_unit": round(111.32 / scale, 5),
        "bbox": {"lon": [round(lon0, 4), round(lon1, 4)], "lat": [round(lat0, 4), round(lat1, 4)]},
        "wards": wards,
    }


def pick(d, keys):
    return {k: d[k] for k in keys if k in d}


def build(root=ROOT):
    e = load(root, "docs/PHASE_E_RESULTS.json")
    k = load(root, "docs/PHASE_K_RESULTS.json")
    p = load(root, "docs/PROMOTION_RESULTS.json")
    reg = load(root, "docs/REGISTRATION.json")
    data_manifest = load(root, "data/MANIFEST.json")
    model_manifest = load(root, "models/MANIFEST.json")

    sha = e["model_sha256"]
    model_file = "models/nyando_hgbcon_%s.onnx" % sha[:12]
    if model_manifest[model_file]["sha256"] != sha:
        raise ValueError("the registered model hash in docs/PHASE_E_RESULTS.json is not the manifest hash")

    train = data_manifest[TRAIN_CSV]
    train_rows, qual = quality(root)
    if qual["rows"] != train["row_count"]:
        raise ValueError("the training file and its manifest disagree on the row count")
    block_dates = dates(read_csv(root, BLOCK_A_CSV))
    train_dates = dates(train_rows)

    stats = p["stats"]
    pooled, buffered, temporal = stats["S6/pooled_mappable"], stats["buffered"], stats["temporal"]
    fields = ("mean_diff", "ci_low", "ci_high", "wins", "losses", "ties", "n_events")
    events_total = pooled["n_events"]
    events_training = events_total - len(block_dates)
    if events_training != buffered["n_events"]:
        raise ValueError("the pooled event count and the Block A date count do not add up")

    primary = k["frames"]["primary"]
    wards = {}
    for name, w in primary["wards"].items():
        item = {"estimable": bool(w["estimable"]), "flood_rows": w["flood_rows"], "control_rows": w["control_rows"],
                "events_both": w["events_both"], "flags": w.get("flags", []), "reason": w.get("reason", "")}
        if "auc" in w:
            item["auc"] = pick(w["auc"], ("mean", "ci_low", "ci_high", "events"))
        if "gap" in w:
            item["gap"] = pick(w["gap"], ("gap", "ci_low", "ci_high"))
        if "rates" in w:
            item["rates"] = {b: pick(w["rates"][b], ("rows", "flood", "share")) for b in BANDS if b in w["rates"]}
        wards[name] = item
    geo = geometry(root)
    if set(geo["wards"]) != set(wards):
        raise ValueError("the ward polygons and the ward results name different wards")
    order = ["Kobura", "Kabonyo/Kanyagwal", "Ahero", "East Kano/Wawidhi", "Awasi/Onjiko"]
    order = [w for w in order if w in wards] + [w for w in sorted(wards) if w not in order]

    block_a = e["block_a"]
    return {
        "schema": 1,
        "model": {"arm": "hgb:con", "file": model_file, "sha256": sha, "format": "ONNX"},
        "counts": {
            "train_rows": train["row_count"],
            "flood": train["sample_types"]["flood_case"],
            "controls": train["sample_types"]["flood_control"],
            "train_dates": len(train_dates),
            "block_a_dates": len(block_dates),
            "events_pooled": events_total,
            "events_training_era": events_training,
            "block_a_rows": block_a["rows_total"],
            "block_a_rows_mappable": block_a["rows_mappable"],
            "block_a_rows_primary": block_a["rows_kept"],
            "block_a_rows_excluded": block_a["rows_excluded"],
            "date_range": {"train": [train_dates[0], train_dates[-1]], "block_a": [block_dates[0], block_dates[-1]]},
        },
        "performance": {"per_event_mean_auc": reg["metrics"]["per_event_mean_auc"]},
        "comparison": {
            "pooled": pick(pooled, fields),
            "buffered": pick(buffered, fields),
            "temporal": pick(temporal, fields),
        },
        "levels": {
            "cutpoints": pick(e["cutpoints"], ("low", "high")),
            "rates": {b: pick(e["rates"][b], ("rows", "flood", "share")) for b in BANDS},
            "bootstrap": {"diff": e["bootstrap"]["diff"], "ci_low": e["bootstrap"]["ci_low"],
                          "ci_high": e["bootstrap"]["ci_high"], "draws": e["bootstrap"]["draws_used"]},
            "outcome": e["outcome"],
            "gates": e["gates"],
        },
        "wards": {"order": order, "items": wards, "primary_rows": primary["rows"]},
        "geo": geo,
        "features": FEATURES,
        "datasets": DATASETS,
        "quality": qual,
        "disclaimer": DISCLAIMER,
        "limits": LIMITS,
        "not_done": NOT_DONE,
    }


def render(ev):
    return json.dumps(ev, indent=1, sort_keys=True) + "\n"


def main(argv):
    text = render(build())
    target = ROOT / OUT
    if "--check" in argv:
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            print("STALE: run python scripts/build_evidence.py and commit frontend/public/evidence.json")
            return 1
        print("evidence.json is current")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print("wrote", OUT, len(text), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
