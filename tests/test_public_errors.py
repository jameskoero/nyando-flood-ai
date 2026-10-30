"""Code scanning alerts 1 to 4 (register row D54): no internal exception text reaches an API response. The registered model's load failure is reported with a fixed message; the exception text goes to the server log only."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_a_load_failure_reports_a_fixed_reason_and_never_the_exception_text():
    from backend import registered as REG
    for exc, hidden in ((ValueError("the SHA-256 differs for /secret/path/model.onnx"), "/secret"), (KeyError("hidden-key"), "hidden-key"), (RuntimeError("internal detail 12345"), "12345")):
        reason = REG._public_reason(exc)
        assert hidden not in reason and "internal detail" not in reason, reason
    assert "SHA-256" in REG._public_reason(ValueError("x SHA-256 y"))
    assert REG._public_reason(ValueError("something else")) == "the model manifest or input contract check failed"
    assert REG._public_reason(RuntimeError("boom")) == "the model could not be loaded (RuntimeError)"


def test_the_load_code_never_formats_exception_text_into_the_reason():
    src = (ROOT / "backend" / "registered.py").read_text(encoding="utf-8")
    assert "str(e)[:200]" not in src and "r.reason = _public_reason(e)" in src
    assert not re.search(r'reason\s*=\s*"%s: %s"', src)


def test_no_backend_module_returns_a_traceback_or_raw_exception_text():
    for path in sorted((ROOT / "backend").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert "format_exc" not in src and "traceback." not in src, path.name
        assert not re.search(r"(detail|content|error|reason)\s*[=:]\s*(str|repr)\(\s*e", src), path.name
