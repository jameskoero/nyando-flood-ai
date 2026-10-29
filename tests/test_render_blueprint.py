"""Render Blueprint (docs/PHASE_E_ADVISORY.md Section 10): a parallel serving instance that touches no existing service, deploys only after CI passes and stores no secret; plus a scan that no key-shaped text is tracked."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BP = (ROOT / "render.yaml").read_text(encoding="utf-8")
EXISTING_SERVICES = {"nyando-flood-api", "nyando-flood-ai"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".sh", ".toml", ".cfg", ".ini", ".js", ".jsx", ".html", ".css", ".env"}


def test_the_blueprint_defines_one_new_docker_service_that_deploys_only_after_checks_pass():
    for must in ("type: web", "runtime: docker", "plan: free", "healthCheckPath: /health", "autoDeployTrigger: checksPass"):
        assert must in BP, must
    names = re.findall(r"^\s+name:\s*(\S+)\s*$", BP, re.M)
    assert names == ["nyando-flood-v2"] and not (set(names) & EXISTING_SERVICES), names


def test_the_four_provider_settings_are_prompted_for_and_never_stored():
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL"):
        assert re.search(r"- key: %s\n\s+sync: false\n" % key, BP), key


def test_the_port_matches_the_one_the_dockerfile_binds():
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    bound = re.findall(r'"--port",\s*"(\d+)"', docker)
    declared = re.findall(r'key: PORT\s+value: "(\d+)"', BP)
    assert bound and bound == declared, (bound, declared)


def test_no_key_shaped_text_is_tracked_in_the_repository():
    files = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True, check=True).stdout.split("\n")
    pattern = re.compile(rb"AIza[0-9A-Za-z_\-]{30,}|gsk_[0-9A-Za-z]{30,}")
    bad = [f for f in files if f and Path(f).suffix in TEXT_SUFFIXES and (ROOT / f).is_file() and pattern.search((ROOT / f).read_bytes())]
    assert bad == [], "key-shaped text in: %s" % bad


def test_the_advisory_document_records_the_parallel_instance():
    doc = (ROOT / "docs" / "PHASE_E_ADVISORY.md").read_text(encoding="utf-8")
    for must in ("render.yaml", "nyando-flood-v2", "sync: false", "autoDeployTrigger"):
        assert must in doc, must
