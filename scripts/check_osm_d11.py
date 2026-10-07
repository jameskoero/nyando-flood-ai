"""scripts/check_osm_d11.py: the one-time comparison of the 15 zero-distance rows with OpenStreetMap (register row D11).
Per row: OSM waterway lines within 50 m (an item counts only if it has a waterway tag; any other item within 50 m is kept apart), and OSM water or wetland areas
containing the point. A failed or unattempted row is recorded as failed, never guessed. The public Overpass servers are shared and often return 504: queries run
one at a time (2 slots per IP), the step has a hard deadline, and rows already answered in the output file are kept (older layouts are converted on load).
Usage: check_osm_d11.py docs/D11_OSM_CHECK.json   |   check_osm_d11.py --selftest"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
RADIUS, DEADLINE, UA = 50, 150, {"User-Agent": "nyando-flood-ai-d11 (github.com/jameskoero/nyando-flood-ai)"}
KEYS = ("waterway", "name", "natural", "water", "wetland")
now = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm(r):
    """A waterway entry must carry a waterway tag; anything else found within the radius moves to other_feature_within_50m."""
    ww = r.get("waterway_within_50m", [])
    r["waterway_within_50m"] = [x for x in ww if "waterway" in x]
    r["other_feature_within_50m"] = r.get("other_feature_within_50m", []) + [x for x in ww if "waterway" not in x]
    return r


def legacy(c):
    """Answered rows of a saved file, keyed by position. A row saved without its own time gets the file's snapshot time and is marked."""
    out = {}
    for r in c.get("rows", []):
        if r.get("status") == "ok":
            if "queried_utc" not in r:
                r["queried_utc"], r["timestamp_is_run_snapshot"] = c.get("snapshot_utc", "unknown"), True
            out[(r["lon"], r["lat"])] = norm(r)
    return out


def record(rows, tsha):
    ok = [norm(r) for r in rows if r["status"] == "ok"]
    sm = {"rows": len(rows), "answered": len(ok), "failed": len(rows) - len(ok), "with_waterway_within_50m": sum(1 for r in ok if r["waterway_within_50m"]),
          "other_feature_within_50m": sum(1 for r in ok if r["other_feature_within_50m"]), "inside_water_or_wetland": sum(1 for r in ok if r["inside_water_or_wetland"])}
    qt = [r["queried_utc"] for r in ok] or [now()]
    return {"schema": 4, "snapshot_utc": max(qt), "source": "OpenStreetMap via Overpass API", "radius_m": RADIUS, "training_data_sha256": tsha,
            "status": "complete" if not sm["failed"] else "incomplete", "summary": sm, "rows": rows}


def selftest():
    old = {"snapshot_utc": "2026-10-06T11:46:17Z", "rows": [{"status": "ok", "lon": 1.0, "lat": 2.0, "waterway_within_50m": [{"natural": "wetland"}], "inside_water_or_wetland": [{"natural": "water"}]}]}
    prev = legacy(old)
    rec = record([prev[(1.0, 2.0)], {"status": "failed: x", "lon": 3.0, "lat": 4.0}], "sha")
    assert rec["summary"] == {"rows": 2, "answered": 1, "failed": 1, "with_waterway_within_50m": 0, "other_feature_within_50m": 1, "inside_water_or_wetland": 1}, rec["summary"]
    assert rec["status"] == "incomplete" and rec["snapshot_utc"] == "2026-10-06T11:46:17Z"
    print("selftest ok")


def get(q, t_end):
    last = "no answer"
    for rnd in range(3):
        for u in ENDPOINTS:
            if time.time() > t_end:
                return None, "deadline"
            try:
                x = requests.post(u, data={"data": q}, timeout=40, headers=UA)
                if x.status_code == 200:
                    return x.json(), None
                last = "HTTP %d" % x.status_code
            except Exception as e:
                last = type(e).__name__
        time.sleep(2 * (rnd + 1))
    return None, last


def tags(el, kind):
    return [{k: e["tags"][k] for k in KEYS if k in e.get("tags", {})} for e in el if e.get("type") == kind]


def main(out):
    from src.models.cv import load_training_frame
    df, tsha = load_training_frame(REPO)
    z = df[df.distance_river == 0].reset_index(drop=True)
    prev = {}
    if Path(out).exists():
        c = json.loads(Path(out).read_text())
        if c.get("training_data_sha256") == tsha:
            prev = legacy(c)
    rows = []
    for r in z.itertuples():
        key = (round(float(r.lon), 6), round(float(r.lat), 6))
        rows.append(prev.get(key) or {"event_id": str(r.event_id), "lon": key[0], "lat": key[1], "flooded": int(r.flooded), "status": "failed: not attempted"})
    print("rows already answered: %d of %d" % (len(prev), len(z)), flush=True)
    save = lambda: (Path(out).write_text(json.dumps(record(rows, tsha), indent=1, sort_keys=True) + "\n"), record(rows, tsha))[1]
    save()
    t_end = time.time() + DEADLINE
    for i, row in enumerate(rows):
        if row["status"] == "ok":
            continue
        if time.time() > t_end:
            row["status"] = "failed: deadline"; continue
        t, lat, lon = time.time(), row["lat"], row["lon"]
        w, e1 = get('[out:json][timeout:30];way(around:%d,%.6f,%.6f)["waterway"];out tags;' % (RADIUS, lat, lon), t_end)
        a, e2 = (None, None) if e1 else get('[out:json][timeout:30];is_in(%.6f,%.6f)->.a;(area.a["natural"~"^(water|wetland)$"];area.a["water"];);out tags;' % (lat, lon), t_end)
        if e1 or e2:
            row["status"] = "failed: %s" % (e1 or e2)
        else:
            row.update(status="ok", queried_utc=now(), waterway_within_50m=tags(w.get("elements", []), "way"), inside_water_or_wetland=tags(a.get("elements", []), "area"))
        save()
        print("row %2d/%d: %s (%.1fs)" % (i + 1, len(rows), row["status"], time.time() - t), flush=True)
    rec = save()
    print(json.dumps(rec["summary"]), flush=True)
    for r in rec["rows"]:
        if r["status"] == "ok" and r["waterway_within_50m"]:
            print("WATERWAY within 50 m:", r["event_id"], r["lat"], r["lon"], r["waterway_within_50m"], flush=True)
        if r["status"] == "ok" and r["other_feature_within_50m"]:
            print("OTHER feature within 50 m (no waterway tag):", r["event_id"], r["lat"], r["lon"], r["other_feature_within_50m"], flush=True)
    print("inside water/wetland, by type:", sorted({str(x.get("natural") or x.get("water")) for r in rec["rows"] if r["status"] == "ok" for x in r["inside_water_or_wetland"]}), flush=True)


if __name__ == "__main__":
    selftest() if sys.argv[1] == "--selftest" else main(sys.argv[1])
