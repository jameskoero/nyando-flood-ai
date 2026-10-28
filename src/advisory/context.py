"""Phase E advisory context (docs/PHASE_E_ADVISORY.md): numeric attributions become categorical drivers, so the prompt, the cache key and the template never see a number."""
import re

from src.advisory.rules import LEVELS

LABELS = {
    "elevation": "elevation",
    "slope": "ground slope",
    "rainfall_3day": "rainfall over the previous three days",
    "distance_river": "distance to mapped surface water",
    "clay_percent": "soil clay content",
    "land_cover": "land cover",
}
LAND_COVER = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up land", 60: "bare or sparsely vegetated land", 80: "permanent water", 90: "herbaceous wetland"}
WARD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z /'-]{1,40}$")


def to_context(ward, level, attributions):
    if ward is not None and not WARD_PATTERN.match(ward):
        raise ValueError("the ward name has unexpected characters")
    if level is not None and level not in LEVELS:
        raise ValueError("unknown level")
    drivers = []
    for a in attributions[:3]:
        d = {"feature": a["feature"], "effect": "raises" if a["delta"] > 0 else "lowers"}
        v, ref = a["value"], a["reference"]
        if a["feature"] == "land_cover":
            d["land_cover"] = None if v is None else LAND_COVER.get(int(v), "an unlisted land-cover class")
        else:
            d["position"] = "missing" if v is None else ("above" if v > ref else "below" if v < ref else "same")
        drivers.append(d)
    return {"ward": ward, "level": level, "drivers": drivers}
