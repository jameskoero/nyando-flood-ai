"""Integrity checks for model files (docs/PHASE_C_PROTOCOL.md Sections 6 and 17): a file's SHA-256 must equal its models/MANIFEST.json entry before anything is loaded from it."""
import hashlib
import io
import json
import os


class IntegrityError(Exception):
    """A model file is missing from the manifest or its bytes differ from the recorded SHA-256."""


def manifest(root):
    with open(os.path.join(root, "models", "MANIFEST.json"), encoding="utf-8") as f:
        return json.load(f)


def read_verified(path, root):
    """The file's bytes, returned only if their SHA-256 equals the manifest entry for the file's path relative to root."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace(os.sep, "/")
    entry = manifest(root).get(rel)
    if entry is None:
        raise IntegrityError("%s has no entry in models/MANIFEST.json" % rel)
    with open(path, "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise IntegrityError("%s: SHA-256 differs from models/MANIFEST.json" % rel)
    return data


def load_verified_pickle(path, root, loader):
    """Verify first, then unpickle the very bytes that were hashed. Returns (object, sha256)."""
    data = read_verified(path, root)
    return loader(io.BytesIO(data)), hashlib.sha256(data).hexdigest()
