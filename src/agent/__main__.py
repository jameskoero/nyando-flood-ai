"""python -m src.agent [--dry-run] [--out FILE] [--base URL]"""
import argparse
import json
import sys

from . import core


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.agent")
    ap.add_argument("--dry-run", action="store_true", help="show the chosen event and payloads; no network")
    ap.add_argument("--out", help="write the report to this file")
    ap.add_argument("--base", default=core.BASE_URL)
    a = ap.parse_args(argv)
    pl = core.plan(core.load_rows())
    if a.dry_run:
        print("event", pl["event_id"], pl["event_date"], "rows", pl["n_rows"],
              "mean rainfall_3day", round(pl["mean_rainfall"], 3))
        for ward, row in pl["picked"].items():
            print(ward, json.dumps(core.to_payload(row), sort_keys=True))
        return 0
    print("calling", a.base, "(the free-tier server can take a minute to wake)", file=sys.stderr)
    report = core.run_replay(pl, base=a.base)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print("wrote", a.out, file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
