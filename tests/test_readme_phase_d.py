"""The README states Phase D and the hotfix exactly: the Stage 0 tick follows the protocol file, the MLP tick follows a registered MLP, and the sealed block and the open rate-limit gap are stated."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_stage_0_tick_follows_the_protocol_file_and_the_mlp_tick_follows_a_registered_mlp():
    assert ("- [x] **D. Stage 0:**" in README) == (ROOT / "docs" / "PHASE_D_PROTOCOL.md").exists()
    manifest = json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert ("- [x] **D. (remainder):**" in README) == any("mlp" in k.lower() for k in manifest)
    assert "register row D46" in README and "Block B stays unbuilt and unscored" in README


def test_the_hotfix_lines_state_what_was_checked_and_what_is_open():
    assert "`/docs` and `/redoc` return 404 on the live service" in README
    assert "- [ ] **0. Hotfix (remainder):**" in README and "per-client rate limit" in README
