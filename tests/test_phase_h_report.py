"""Phase H situation report: the committed report is a labelled replay that matches the repo."""
import hashlib
import re
from pathlib import Path

from src.agent import core

ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "docs" / "PHASE_H_SITUATION_REPORT.md").read_text(encoding="utf-8")


def test_the_report_says_it_is_a_replay_and_carries_the_service_limits():
    for text in ("Replay of recorded data", "not a validation", "not a forecast",
                 "- Notice:", "- Claim limit:", "- Score meaning:"):
        assert text in REPORT


def test_the_report_names_the_primary_frame_the_code_computes():
    rows, n_mappable = core.load_primary()
    assert "(%d of %d mappable Block A rows" % (len(rows), n_mappable) in REPORT


def test_the_report_names_the_model_file_in_the_repo():
    m = re.search(r'"file": "([^"]+\.onnx)".*?"sha256": "([0-9a-f]{64})"', REPORT, re.S)
    assert m
    assert hashlib.sha256((ROOT / "models" / m.group(1)).read_bytes()).hexdigest() == m.group(2)
