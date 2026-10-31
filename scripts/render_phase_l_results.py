#!/usr/bin/env python3
"""Writes the generated Phase L sections into MODEL_CARD.md and data/DATA_SOURCES.md from docs/PHASE_L_RESULTS.json (docs/PHASE_L_PROTOCOL.md Section 7). Re-running is safe. Standard library only."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models import phase_l_report as RP  # noqa: E402

TARGETS = (("MODEL_CARD.md", RP.render_card, r"^## Previous registered model"), ("data/DATA_SOURCES.md", RP.render_datasheet, r"^## 7\. Maintenance"))


def splice(text, body, anchor):
    block = "%s\n%s\n%s" % (RP.BEGIN, body, RP.END)
    if RP.BEGIN in text:
        head, rest = text.split(RP.BEGIN, 1)
        return head + block + rest.split(RP.END, 1)[1]
    m = re.search(anchor, text, re.M)
    if not m:
        sys.exit("STOPPED: anchor %r not found" % anchor)
    return text[:m.start()] + block + "\n\n" + text[m.start():]


def main():
    res = json.loads((ROOT / "docs" / "PHASE_L_RESULTS.json").read_text(encoding="utf-8"))
    for rel, render, anchor in TARGETS:
        p = ROOT / rel
        p.write_text(splice(p.read_text(encoding="utf-8"), render(res), anchor), encoding="utf-8")
        print("wrote", rel)


if __name__ == "__main__":
    main()
