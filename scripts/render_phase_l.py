#!/usr/bin/env python3
"""Writes the generated counts into docs/PHASE_L_PROTOCOL.md between its two markers, from the training file (docs/PHASE_L_PROTOCOL.md Section 8). Re-running is safe. Standard library only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models import phase_l as L  # noqa: E402


def main():
    doc = ROOT / "docs" / "PHASE_L_PROTOCOL.md"
    text = doc.read_text(encoding="utf-8")
    if text.count(L.BEGIN) != 1 or text.count(L.END) != 1:
        sys.exit("STOPPED: the protocol must hold the counts markers exactly once")
    head, rest = text.split(L.BEGIN, 1)
    tail = rest.split(L.END, 1)[1]
    body = L.render_counts(L.count_tables(ROOT / "data" / "training" / "nyando_training_v2_multidate.csv"))
    doc.write_text(head + L.BEGIN + "\n" + body + "\n" + L.END + tail, encoding="utf-8")
    print("wrote the counts into", doc.relative_to(ROOT))


if __name__ == "__main__":
    main()
