"""scripts/build_block_a.py: build D20 Block A (docs/D20_PROTOCOL.md Sections 6 and 11): the committed dates only, at the committed sample size, through the same pipeline as the training file. A date that fails is recorded with its reason and never replaced. Colab only: needs Earth Engine (your login) and network. Resumable."""
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
NAMES = ("nyando_block_a.csv", "nyando_block_a_dropped.csv", "nyando_block_a_sets.csv", "nyando_block_a_failures.csv")


def _read(p, col):
    import pandas as pd
    return set(pd.read_csv(p)[col]) if p.exists() else set()


def build(out_dir, budget_s=1500):
    """Returns True when every Block A date is built or failed; False means: run again to resume."""
    import pandas as pd
    import build_initial_dataset as B
    from src.data import block_a
    sel = block_a.load_selection()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_p, drop_p, sets_p, fail_p = (out / n for n in NAMES)
    B.DATE_N = sel["per_class"]
    done, failed = _read(sets_p, "sample_set"), _read(fail_p, "date")
    jobs = [j for j in block_a.block_a_jobs(sel) if j["set_id"] not in done and j["anchor"].isoformat() not in failed]
    print("Block A: %d dates committed | built %d | failed %d | to do %d | %d points per class" % (len(sel["block_a"]), len(done), len(failed), len(jobs), B.DATE_N), flush=True)
    if jobs:
        try:
            B.init_earth_engine()
        except (Exception, SystemExit) as e:
            print("Earth Engine is not initialised (%s): signing in" % type(e).__name__, flush=True)
            import ee
            ee.Authenticate()
            B.init_earth_engine()
        wards, wf, aoi = B.load_ward_geometry()
        B.build_rain_geometries(aoi)
        clients = {"event_peak": B.RecordingClient(), "date_mosaic": B.DateMosaicClient()}
        t0 = time.time()
        for job in jobs:
            if time.time() - t0 > budget_s:
                print("time budget reached: run again to resume", flush=True)
                break
            t1, d, reason = time.time(), job["anchor"].isoformat(), None
            try:
                rows, dropped, stats = B.process_set(job, clients, wards, wf, aoi)
                if rows is None:
                    reason = "no GFM coverage, or coverage below the minimum valid pixels"
                elif not rows:
                    reason = "no usable rows after the missing-value drops"
            except B.CoordinateMismatchError:
                raise
            except Exception as exc:
                reason = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            if reason:
                pd.DataFrame([{"date": d, "reason": reason}]).to_csv(fail_p, mode="a", header=not fail_p.exists(), index=False)
                print("  FAILED %s: %s (not replaced)" % (d, reason), flush=True)
                continue
            pd.DataFrame(rows).to_csv(rows_p, mode="a", header=not rows_p.exists(), index=False)
            if dropped:
                pd.DataFrame(dropped).to_csv(drop_p, mode="a", header=not drop_p.exists(), index=False)
            pd.DataFrame([stats]).to_csv(sets_p, mode="a", header=not sets_p.exists(), index=False)
            print("  wrote %d rows for %s (%d s)" % (len(rows), d, time.time() - t1), flush=True)
    done, failed = _read(sets_p, "sample_set"), _read(fail_p, "date")
    if len(done) + len(failed) < len(sel["block_a"]):
        return False
    if not fail_p.exists():
        pd.DataFrame(columns=["date", "reason"]).to_csv(fail_p, index=False)
    return True


LAYERS = ["ensemble_flood_extent", "exclusion_mask", "reference_water_mask"]
BASE = "https://stac.eodc.eu/api/v1/collections/GFM/items/"
NODATA = 255.0


def layer_flags(df):
    """The same reading as scripts/audit_gfm_layers.py, for a frame with gfm_item_id, lon, lat and flooded columns. It stops rather than write if a point has no valid layer value or a flood-extent value disagrees with its label."""
    import numpy as np
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from rasterio.env import Env
    gid = df["gfm_item_id"].astype(str).to_numpy()
    lon, lat = df["lon"].to_numpy(), df["lat"].to_numpy()
    res = {L: np.full(len(df), np.nan) for L in LAYERS}

    def read_layers(item, rows):
        out = {}
        for L in LAYERS:
            with rasterio.open(item["assets"][L]["href"]) as ds:
                xs, ys = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(lon[rows], lat[rows])
                xs, ys = np.asarray(xs), np.asarray(ys)
                vals = np.array([v[0] for v in ds.sample(list(zip(xs, ys)))], dtype=float)
                b = ds.bounds
                vals[~((xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top))] = np.nan
                out[L] = vals
        return out

    t0 = time.time()
    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
        for g in dict.fromkeys(gid.tolist()):
            rows = np.flatnonzero(gid == g)
            per_scene = []
            for iid in g.split(";"):
                req = urllib.request.Request(BASE + iid, headers={"User-Agent": "layer-audit"})
                per_scene.append(read_layers(json.load(urllib.request.urlopen(req, timeout=60)), rows))
            flood = np.vstack([p["ensemble_flood_extent"] for p in per_scene])
            valid = ~np.isnan(flood) & (flood != NODATA)
            for L in LAYERS:
                arr = np.vstack([p[L] for p in per_scene]).astype(float)
                arr[np.isnan(arr) | (arr == NODATA) | ~valid] = np.nan
                m = np.where(np.isnan(arr), -np.inf, arr).max(axis=0)
                m[np.isinf(m)] = np.nan
                res[L][rows] = m
            print("flags: %d rows, %d scene(s), %d s" % (len(rows), len(per_scene), time.time() - t0), flush=True)
    if any(np.isnan(res[L]).any() for L in LAYERS):
        raise SystemExit("some points have no valid layer value; nothing written")
    agree = float(((df["flooded"].to_numpy() == 1) == (res["ensemble_flood_extent"] == 1)).mean())
    if agree < 1.0:
        raise SystemExit("flood-extent value disagrees with the label on some rows (agreement %.4f); nothing written" % agree)
    return pd.DataFrame({"row": np.arange(len(df)), "lon": df["lon"], "lat": df["lat"], "flooded": df["flooded"].astype(int),
                         "exclusion_mask": res["exclusion_mask"].astype(int), "reference_water_mask": res["reference_water_mask"].astype(int)})


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/content/blockA_out"
    print("Block A build complete" if build(out) else "Block A build incomplete: run again")
