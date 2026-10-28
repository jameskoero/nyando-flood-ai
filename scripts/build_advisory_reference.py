"""Writes docs/PHASE_E_REFERENCE.json from the real training file (docs/PHASE_E_ADVISORY.md Section 7). Stops, writing nothing, if any verified hash, count, column or alignment differs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.advisory.reference import REFERENCE_PATH, build  # noqa: E402


def main():
    try:
        ref = build(ROOT)
    except ValueError as e:
        sys.exit("STOPPED: %s" % e)
    REFERENCE_PATH.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", REFERENCE_PATH.relative_to(ROOT))
    print(json.dumps(ref, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
