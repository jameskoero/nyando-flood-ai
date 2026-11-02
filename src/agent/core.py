"""Phase H core: replay recorded Block A data through the live V2 service.

Standard library only, so it runs in Termux and in CI. The MCP server and the
LangGraph chain are thin wrappers over these functions. Every input row comes
from data/confirmatory/nyando_block_a.csv and every number in the report is
returned by the service; nothing is simulated.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://nyando-flood-v2.onrender.com"
USER_AGENT = "nyando-flood-ai-agent/2.0"
DATA = Path(__file__).resolve().parents[2] / "data"
BLOCK_A = DATA / "confirmatory" / "nyando_block_a.csv"
BLOCK_FLAGS = DATA / "confirmatory" / "nyando_block_a_layer_flags.csv"
TRAIN_CSV = DATA / "training" / "nyando_training_v2_multidate.csv"
TRAIN_FLAGS = DATA / "derived" / "gfm_layer_flags.csv"
REPLAY_NOTE = (
    "Replay of recorded data through the live service. It is an illustration, "
    "not a validation and not a forecast."
)


class ApiError(RuntimeError):
    """The service failed, or two of its answers disagreed."""


def load_rows(path=BLOCK_A):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def location_key(lon, lat):
    """Mirror of src/models/cv.py location_keys: lon and lat rounded to 6 decimals, as text."""
    return "%r,%r" % (round(float(lon) * 1e6) / 1e6, round(float(lat) * 1e6) / 1e6)


def mappable(rows, flags):
    """Flood rows plus controls outside the GFM exclusion mask (cv.mappable_mask).

    The flag file must describe exactly these rows, as cv.load_layer_flags checks."""
    if len(rows) != len(flags):
        raise ValueError("layer-flag file and data file differ in length")
    kept = []
    for r, f in zip(rows, flags):
        same = (
            abs(float(f["lon"]) - float(r["lon"])) <= 1e-9
            and abs(float(f["lat"]) - float(r["lat"])) <= 1e-9
            and float(f["flooded"]) == float(r["flooded"])
        )
        if not same:
            raise ValueError("layer-flag file is not aligned with the data file")
        if float(r["flooded"]) == 1 or float(f["exclusion_mask"]) == 0:
            kept.append(r)
    return kept


def primary_frame(block, block_flags, train, train_flags):
    """The rows Phase E validated the levels on: mappable Block A rows at locations
    that are not in the mappable training frame. Returns (rows, number_mappable)."""
    b = mappable(block, block_flags)
    seen = {location_key(r["lon"], r["lat"]) for r in mappable(train, train_flags)}
    return [r for r in b if location_key(r["lon"], r["lat"]) not in seen], len(b)


def load_primary():
    return primary_frame(
        load_rows(BLOCK_A), load_rows(BLOCK_FLAGS), load_rows(TRAIN_CSV), load_rows(TRAIN_FLAGS)
    )


def pick_event(rows):
    """Event with the highest mean rainfall_3day over its rows (ties: event id)."""
    acc = {}
    for r in rows:
        s = acc.setdefault(r["event_id"], [0.0, 0])
        s[0] += float(r["rainfall_3day"])
        s[1] += 1
    return min(acc, key=lambda e: (-acc[e][0] / acc[e][1], e))


def pick_ward_rows(rows, event_id):
    """Per ward, the row at the lower median of rainfall_3day (ties: file order)."""
    by_ward = {}
    for i, r in enumerate(rows):
        if r["event_id"] == event_id:
            by_ward.setdefault(r["ward"], []).append((float(r["rainfall_3day"]), i, r))
    picked = {}
    for ward, items in by_ward.items():
        items.sort(key=lambda t: (t[0], t[1]))
        picked[ward] = items[(len(items) - 1) // 2][2]
    return dict(sorted(picked.items()))


def to_payload(row):
    p = {
        "elevation": float(row["elevation"]),
        "slope": float(row["slope"]),
        "rainfall_3day": float(row["rainfall_3day"]),
        "distance_river": float(row["distance_river"]),
        "land_cover": int(float(row["land_cover"])),
        "ward": row["ward"],
    }
    if (row.get("clay_percent") or "").strip() != "":
        p["clay_percent"] = float(row["clay_percent"])
    return p


def plan(rows, n_mappable=None):
    event_id = pick_event(rows)
    ev = [r for r in rows if r["event_id"] == event_id]
    picked = pick_ward_rows(rows, event_id)
    return {
        "event_id": event_id,
        "event_date": ev[0]["event_date"],
        "n_rows": len(ev),
        "mean_rainfall": sum(float(r["rainfall_3day"]) for r in ev) / len(ev),
        "picked": picked,
        "wards_without_point": sorted({r["ward"] for r in rows} - set(picked)),
        "n_primary": len(rows),
        "n_mappable": n_mappable,
    }


def http_post(url, payload, timeout=90):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(text)
        except ValueError:
            return exc.code, {"raw": text[:200]}


def call(post, path, payload, base=BASE_URL, tries=4, wait=20.0, sleep=time.sleep):
    """POST with retries on 429, 5xx and network errors (free-tier wake-up)."""
    last = (0, {})
    for attempt in range(tries):
        try:
            status, body = post(base + path, payload)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status, body = 0, {"error": type(exc).__name__}
        if status == 200:
            return body
        last = (status, body)
        if status in (0, 429, 502, 503, 504) and attempt < tries - 1:
            sleep(wait)
            continue
        break
    raise ApiError("POST %s failed: HTTP %s %s" % (path, last[0], json.dumps(last[1])[:200]))


def predict_step(post, picked, **kw):
    return {w: call(post, "/v2/score", to_payload(r), **kw) for w, r in picked.items()}


def advise_step(post, picked, **kw):
    return {w: call(post, "/v2/advisory", to_payload(r), **kw) for w, r in picked.items()}


def check_consistency(scores, advice):
    problems = []
    for ward, s in scores.items():
        a = advice.get(ward)
        if a is None:
            problems.append("%s: no advisory" % ward)
        elif s.get("score") != a.get("score"):
            problems.append("%s: /v2/score %s vs /v2/advisory %s" % (ward, s.get("score"), a.get("score")))
        elif s.get("model") != a.get("model"):
            problems.append("%s: the two answers name different models" % ward)
    return problems


def _cell(value):
    return str(value).replace("|", "/").replace("\n", " ")


def _render(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, str):
                lines.append("- %s: %s" % (k, v))
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                lines.append("- %s:" % k)
                lines += ["  - %s" % x for x in v]
            else:
                lines.append("- %s: %s" % (k, json.dumps(v, ensure_ascii=False, sort_keys=True)))
        return "\n".join(lines)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_report(pl, scores, advice, generated):
    first = advice[next(iter(advice))]
    out = [
        "# Situation report: replay of a recorded event", "",
        "> " + REPLAY_NOTE, "",
        "- Event: %s (scene date %s)" % (pl["event_id"], pl["event_date"]),
        "- Mean rainfall_3day over the event's %d sampled rows: %.1f" % (pl["n_rows"], pl["mean_rainfall"]),
        "- Wards shown: one recorded sample point per ward, at the median rainfall_3day of that ward's rows in the event",
    ]
    if pl.get("n_mappable"):
        out.append("- Rows used: the primary frame of the level validation (%d of %d mappable Block A rows, at locations not in the mappable training frame)" % (pl["n_primary"], pl["n_mappable"]))
    if pl.get("wards_without_point"):
        out.append("- No primary-frame point in this event for: %s" % ", ".join(pl["wards_without_point"]))
    out += [
        "- Providers configured on the service: %s" % (", ".join(first.get("providers_configured") or []) or "none"),
        "- Generated (UTC): %s" % generated, "",
        "| Ward | rainfall_3day | Score | Level | Recorded label at the point |",
        "|---|---|---|---|---|",
    ]
    for ward, row in pl["picked"].items():
        a = advice[ward]
        label = "GFM flooded" if row["flooded"] in ("1", "1.0") else "GFM not flooded"
        out.append("| %s | %s | %s | %s | %s |" % (
            _cell(ward), _cell(row["rainfall_3day"]), _cell(scores[ward]["score"]),
            _cell(a.get("level") or "not available"), label))
    out += ["", "## Advisory by ward", ""]
    for ward in pl["picked"]:
        a = advice[ward]
        src, mdl = a.get("advisory_source"), a.get("advisory_model")
        out += ["### " + ward, "", _render(a.get("advisory")), "",
                "Source: %s" % (("%s (%s)" % (src, mdl)) if mdl else src)]
        if a.get("attempts"):
            out.append("Providers tried: %s" % json.dumps(a["attempts"], ensure_ascii=False, sort_keys=True)[:400])
        out.append("")
    out += [
        "## Notices (verbatim from the service)", "",
        "- Score meaning: %s" % first.get("score_meaning"),
        "- Level basis: %s" % first.get("level_basis"),
        "- Notice: %s" % first.get("notice"),
        "- Claim limit: %s" % first.get("claim_limit"),
        "- Model as reported: %s" % json.dumps(first.get("model"), sort_keys=True), "",
    ]
    return "\n".join(out)


def run_replay(pl, post=http_post, generated=None, **kw):
    scores = predict_step(post, pl["picked"], **kw)
    advice = advise_step(post, pl["picked"], **kw)
    problems = check_consistency(scores, advice)
    if problems:
        raise ApiError("; ".join(problems))
    stamp = generated or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return build_report(pl, scores, advice, stamp)
