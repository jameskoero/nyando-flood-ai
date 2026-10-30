"""Register row D51, option C: the legacy POST /predict answers 410 Gone unless LEGACY_PREDICT=on. These checks read files only, so they run on the phone too; tests/test_legacy_predict_gate.py exercises the route."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _gate_source():
    m = re.search(r'def _legacy_gate\(.*?(?=^@app\.post\("/predict")', MAIN, re.S | re.M)
    assert m, "the gate is not defined before the /predict route"
    return m.group(0)


def test_the_legacy_route_is_behind_the_gate_and_the_default_is_off():
    assert '@app.post("/predict", dependencies=[Depends(_legacy_gate)])' in MAIN
    gate = _gate_source()
    assert 'os.environ.get("LEGACY_PREDICT", "off")' in gate and "status_code=410" in gate
    assert '"/v2/score"' in gate and '"/v2/advisory"' in gate


def test_the_gate_logs_the_attempt_but_never_the_caller():
    gate = _gate_source()
    assert "legacy_predict_attempt" in gate
    for banned in ("client", "remote_address", "cf-connecting-ip", "x-forwarded"):
        assert banned not in gate.lower(), banned


def test_the_two_audit_files_switch_the_legacy_route_on_for_their_own_tests():
    for name in ("test_v2_audit.py", "test_backend_registered.py"):
        text = (ROOT / "tests" / name).read_text(encoding="utf-8")
        assert '_os.environ.setdefault("LEGACY_PREDICT", "on")' in text, name


def test_the_readme_states_the_default_and_the_switch():
    assert "410 Gone" in README and "LEGACY_PREDICT=on" in README


def test_d51_stays_open_and_records_the_gate():
    rows = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D51 |")]
    assert len(rows) == 1
    cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    assert cells[1] == "Open" and cells[6] == "open" and cells[5].startswith("open:")
    assert "410" in cells[3] and "LEGACY_PREDICT" in cells[3] and "/predict" in cells[3]
