"""Phase F closure (register rows D35, D50 and D51; docs/ROADMAP_DEVIATIONS.md, README): the dashboard rows are closed on tests that exist, the legacy endpoint is recorded as an open row, and the README names the V2 host."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines()
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _cells(row_id):
    rows = [l for l in REG if l.startswith("| %s |" % row_id)]
    assert len(rows) == 1, row_id
    cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    assert len(cells) == 7, row_id
    return cells


def _functions(path):
    return {n.name for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}


def test_d35_and_d50_are_closed_on_tests_that_exist():
    for row_id in ("D35", "D50"):
        cells = _cells(row_id)
        assert cells[6] == "closed" and cells[5].startswith("tests/"), row_id
        rel, name = cells[5].split("::")
        assert name in _functions(ROOT / rel), (row_id, name)


def test_d51_records_the_legacy_endpoint_as_an_open_row():
    cells = _cells("D51")
    assert cells[1] == "Open" and cells[6] == "open" and cells[5].startswith("open:") and "/predict" in cells[3]


def test_the_readme_names_the_v2_host_and_not_the_old_one():
    assert "nyando-flood-v2.onrender.com" in README and "nyando-flood-api" not in README
