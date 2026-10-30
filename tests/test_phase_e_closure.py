"""Phase E closure (register rows D49 and D50; docs/PHASE_E_ADVISORY.md Section 12): D49 closed on its existing test, D50 recording the second service, the README tick and layout line telling the truth, and the live evidence written down."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines()
README = (ROOT / "README.md").read_text(encoding="utf-8")
DOC = (ROOT / "docs" / "PHASE_E_ADVISORY.md").read_text(encoding="utf-8")


def _row(row_id):
    rows = [line for line in REG if line.startswith("| %s |" % row_id)]
    assert len(rows) == 1, row_id
    return rows[0]


def test_d49_is_closed_on_its_existing_test_and_d50_records_the_second_service():
    d49, d50 = _row("D49"), _row("D50")
    assert d49.rstrip().endswith("| closed |") and "tests/test_phase_e.py::test_the_protocol_states_the_frozen_rules_and_the_four_deviations" in d49
    assert "nyando-flood-v2" in d50 and "render.yaml" in d50 and d50.rstrip().endswith("| closed |")


def test_the_readme_ticks_phase_e_and_describes_the_advisory_package_truthfully():
    assert "- [x] **E. LLM advisory layer**" in README and "- [ ] **E. LLM advisory layer**" not in README
    assert "(no model, network or data files)" not in README
    line = [x for x in README.splitlines() if x.startswith("src/advisory/")]
    assert len(line) == 1 and "providers" in line[0]


def test_the_advisory_document_records_the_live_evidence():
    for must in ("## 12.", "nyando-flood-v2.onrender.com", "e59d18d", "gemini-3.1-flash-lite", "6 of 6", "18 of 18", "fallback chain was exercised live", "is closed"):
        assert must in DOC, must
