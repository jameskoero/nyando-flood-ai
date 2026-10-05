"""scripts/d20_scan.py: the catalogue scan behind data/derived/d20_scan.csv (docs/D20_PROTOCOL.md). It reads label metadata only (valid pixels and flood pixels per GFM scene date), never model output. Colab paths: run it from a Colab notebook."""
import os, subprocess, sys, time
from datetime import date
R, OUT = "/content/nyando-flood-ai", "/content/d20/d20_scan.csv"
START, END, MIN_GAP, BUDGET_S = "2020-01-01", "2026-08-31", 14, 25 * 60
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, cwd="/content"); return r.returncode, (r.stdout + r.stderr).strip()
def setup():
    if not os.path.isdir(R + "/.git"): sh("git clone -q https://github.com/jameskoero/nyando-flood-ai.git " + R)
    sh("cd %s && git fetch -q origin && git checkout -q -B main origin/main && git reset -q --hard origin/main" % R)
    print("main:", sh("cd %s && git log -1 --format='%%h %%s'" % R)[1], flush=True)
    print("packages:", sh("pip -q install pystac-client rioxarray geopandas shapely 2>&1 | tail -2")[1] or "ok", flush=True)
    sys.path[:0] = [R, R + "/scripts"]
def main():
    setup()
    import pandas as pd
    import build_initial_dataset as B
    from src.data.gfm_client import GFMClient
    wards, wf, aoi = B.load_ward_geometry()
    ex = [date.fromisoformat(d) for d in pd.read_csv(R + "/data/training/nyando_training_v2_multidate_sets.csv")["scene_date"]]
    items = GFMClient().search_items(list(aoi.bounds), START + "/" + END)
    dates = sorted({it.datetime.date() for it in items if it.datetime})
    cands = [d for d in dates if all(abs((d - e).days) >= MIN_GAP for e in ex)]
    print("GFM scene dates %s to %s over the ward area: %d | existing sets: %d | candidates at least %d days from every existing date: %d" % (START, END, len(dates), len(ex), MIN_GAP, len(cands)), flush=True)
    done = set(pd.read_csv(OUT)["date"]) if os.path.exists(OUT) else set()
    client, t0, n = B.DateMosaicClient(), time.time(), 0
    for d in cands:
        if str(d) in done: continue
        if time.time() - t0 > BUDGET_S: print("time budget reached: run the run-cell again to resume", flush=True); break
        row = {"date": str(d), "n_slices": 0, "valid_px": 0, "flood_px": 0, "error": ""}
        try:
            res = client.get_peak_flood_extent(aoi, "EPSG:4326", d)
            if res is not None: row.update(n_slices=res.contributing_item_id.count(";") + 1, valid_px=int(res.valid_mask.sum()), flood_px=int(res.flood_mask.sum()))
        except Exception as e: row["error"] = type(e).__name__
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        n += 1
        if n % 10 == 0: print("scanned %d this run | %d of %d done | last %s valid %d flood %d" % (n, len(done) + n, len(cands), d, row["valid_px"], row["flood_px"]), flush=True)
    print("scan file rows: %d of %d candidates" % (len(pd.read_csv(OUT)) if os.path.exists(OUT) else 0, len(cands)))
if __name__ == "__main__": main()
