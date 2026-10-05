"""Model file integrity (docs/PHASE_C_PROTOCOL.md Sections 6 and 17; register row D34): nothing is unpickled or loaded before its SHA-256 matches models/MANIFEST.json."""
import hashlib
import json
from pathlib import Path

import pytest

from backend.integrity import IntegrityError, load_verified_pickle
from backend.registered import load_registered

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _root(base, files, entries):
    (base / "models").mkdir()
    for rel, data in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (base / "models" / "MANIFEST.json").write_text(json.dumps(entries), encoding="utf-8")
    return base


def _never(_):
    raise AssertionError("unpickled before the hash was checked")


def test_a_file_whose_hash_differs_is_never_unpickled(tmp_path):
    root = _root(tmp_path, {"backend/models/x.pkl": b"tampered"}, {"backend/models/x.pkl": {"sha256": hashlib.sha256(b"payload").hexdigest(), "status": "legacy"}})
    with pytest.raises(IntegrityError, match="SHA-256"):
        load_verified_pickle(str(root / "backend/models/x.pkl"), str(root), _never)


def test_a_file_without_a_manifest_entry_is_refused(tmp_path):
    root = _root(tmp_path, {"backend/models/x.pkl": b"payload"}, {})
    with pytest.raises(IntegrityError, match="no entry"):
        load_verified_pickle(str(root / "backend/models/x.pkl"), str(root), _never)


def test_a_verified_file_is_loaded_from_the_bytes_that_were_hashed(tmp_path):
    root = _root(tmp_path, {"backend/models/x.pkl": b"payload"}, {"backend/models/x.pkl": {"sha256": hashlib.sha256(b"payload").hexdigest(), "status": "legacy"}})
    obj, digest = load_verified_pickle(str(root / "backend/models/x.pkl"), str(root), lambda f: f.read())
    assert obj == b"payload" and digest == hashlib.sha256(b"payload").hexdigest()


def test_the_real_legacy_model_loads_through_the_verified_path():
    joblib = pytest.importorskip("joblib")
    model, digest = load_verified_pickle(str(ROOT / "backend" / "models" / "nyando_xgb_v1.pkl"), str(ROOT), joblib.load)
    assert hasattr(model, "predict_proba") and digest == MANIFEST["backend/models/nyando_xgb_v1.pkl"]["sha256"]


def test_a_tampered_registered_model_is_not_loaded(tmp_path):
    pytest.importorskip("onnxruntime")
    rel, entry = [(k, v) for k, v in MANIFEST.items() if v.get("status") == "active"][0]
    root = _root(tmp_path, {rel: (ROOT / rel).read_bytes() + b"\x00"}, {rel: entry})
    r = load_registered(str(root))
    assert r.session is None and "SHA-256" in r.reason


def test_without_onnxruntime_the_registered_model_is_unloaded_with_a_reason():
    def boom():
        raise ImportError("no onnxruntime")
    r = load_registered(str(ROOT), ort_loader=boom)
    assert r.session is None and "onnxruntime" in r.reason and r.describe() == {"loaded": False, "reason": r.reason}


def test_the_real_registered_model_loads_and_reports_its_hash():
    pytest.importorskip("onnxruntime")
    rel, entry = [(k, v) for k, v in MANIFEST.items() if v.get("status") == "active"][0]
    r = load_registered(str(ROOT))
    assert r.session is not None, r.reason
    assert r.sha256 == entry["sha256"] and r.describe()["trained_on"] == entry["trained_on"]
