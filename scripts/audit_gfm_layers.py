"""Read the GFM exclusion mask and reference water mask at every training point (protocol r4).

Needs network access and the packages rasterio and pyproj (not in requirements.txt). From the repository root:
    python scripts/audit_gfm_layers.py
Each scene named in gfm_item_id is looked up in the EODC STAC catalogue (ids joined by ';' are separate scenes; their
values are combined by the maximum on pixels with a valid flood-extent value). Three layers are read at the sample
points. The flood-extent value must agree with the label on every row, and data/derived/gfm_layer_flags.csv is written."""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cv import FLAGS_PATH, LABEL, load_training_frame

BASE = "https://stac.eodc.eu/api/v1/collections/GFM/items/"
LAYERS = ["ensemble_flood_extent", "exclusion_mask", "reference_water_mask"]
NODATA = 255.0


def main():
    import rasterio
    from pyproj import Transformer
    from rasterio.env import Env

    df, _ = load_training_frame()
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
    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
             GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
        for g in dict.fromkeys(gid.tolist()):
            rows = np.flatnonzero(gid == g)
            per_scene = []
            for iid in g.split(";"):
                req = urllib.request.Request(BASE + iid, headers={"User-Agent": "layer-audit"})
                item = json.load(urllib.request.urlopen(req, timeout=60))
                per_scene.append(read_layers(item, rows))
            flood = np.vstack([p["ensemble_flood_extent"] for p in per_scene])
            valid = ~np.isnan(flood) & (flood != NODATA)
            for L in LAYERS:
                arr = np.vstack([p[L] for p in per_scene]).astype(float)
                arr[np.isnan(arr) | (arr == NODATA) | ~valid] = np.nan
                m = np.where(np.isnan(arr), -np.inf, arr).max(axis=0)
                m[np.isinf(m)] = np.nan
                res[L][rows] = m
            print(len(rows), "rows,", len(per_scene), "scene(s),", round(time.time() - t0), "s", flush=True)

    if any(np.isnan(res[L]).any() for L in LAYERS):
        raise SystemExit("some points have no valid layer value; nothing written")
    agree = float(((df[LABEL].to_numpy() == 1) == (res["ensemble_flood_extent"] == 1)).mean())
    if agree < 1.0:
        raise SystemExit("flood-extent value disagrees with the label on some rows (agreement %.4f); nothing written" % agree)
    out = pd.DataFrame({"row": np.arange(len(df)), "lon": df["lon"], "lat": df["lat"], "flooded": df[LABEL].astype(int),
                        "exclusion_mask": res["exclusion_mask"].astype(int),
                        "reference_water_mask": res["reference_water_mask"].astype(int)})
    FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FLAGS_PATH, index=False)
    print("wrote", FLAGS_PATH, "| controls in the exclusion mask:",
          int(out.loc[out["flooded"] == 0, "exclusion_mask"].sum()))


if __name__ == "__main__":
    main()
