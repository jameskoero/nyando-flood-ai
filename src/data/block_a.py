"""D20 Block A (docs/D20_PROTOCOL.md Sections 6 and 11): the committed dates as build jobs, the accounting of built and failed dates, and the scorable-event count. Offline: no network and no Earth Engine."""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "data" / "derived" / "d20_selection.json"
MIN_CLASS_N = 30


def load_selection(path=SELECTION):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def block_a_jobs(sel):
    """Section 6: one date-mosaic job per committed Block A date, in the committed order. A Block B date is refused."""
    a, b = list(sel["block_a"]), set(sel["block_b"])
    if set(a) & b:
        raise ValueError("Block A and Block B overlap: %s" % sorted(set(a) & b))
    return [{"set_id": "date-" + d, "anchor": date.fromisoformat(d), "group": "date-" + d, "mode": "date_mosaic"} for d in a]


def account_for(sel, built, failed):
    """Section 11: every Block A date is built or failed, exactly once; nothing else, least of all a Block B date, is accepted. Returns (built, failed) in the committed order."""
    a, b = list(sel["block_a"]), set(sel["block_b"])
    built, failed = list(built), list(failed)
    extra = (set(built) | set(failed)) - set(a)
    if extra:
        raise ValueError("dates that are not in Block A: %s%s" % (sorted(extra), " (Block B!)" if extra & b else ""))
    if set(built) & set(failed) or len(set(built)) != len(built) or len(set(failed)) != len(failed):
        raise ValueError("a date is recorded twice")
    missing = [d for d in a if d not in set(built) | set(failed)]
    if missing:
        raise ValueError("Block A dates neither built nor failed: %s" % missing)
    return [d for d in a if d in set(built)], [d for d in a if d in set(failed)]


def scorable_events(df, keep=None, min_class_n=MIN_CLASS_N):
    """Events with at least min_class_n rows of each class on a frame. `keep` is a boolean array aligned with df (the mappable frame); None means the full frame."""
    import numpy as np
    keep = np.ones(len(df), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    c = df.loc[keep, ["event_id", "flooded"]].groupby(["event_id", "flooded"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    return sorted(c.index[(c[0] >= min_class_n) & (c[1] >= min_class_n)])
