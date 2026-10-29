#!/usr/bin/env bash
# Pre-push replay of the repository's structural rules with the standard library only: the README layout, the register rows, syntax of the staged files, workflows without write permission and key-shaped text. tools/ship.sh runs it after staging; skip it with SKIP_CHECK=1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec python3 - <<'PY'
import ast, json, re, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
TEXT = {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".sh", ".toml", ".cfg", ".ini", ".js", ".jsx", ".html", ".css", ".env"}
KEY = re.compile(rb"AIza[0-9A-Za-z_\-]{30,}|gsk_[0-9A-Za-z]{30,}")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


tracked = git("ls-files").splitlines()
staged = [f for f in git("diff", "--cached", "--name-only", "--diff-filter=AM").splitlines() if f]
problems, ran = [], []


def check(name, fn):
    ran.append(name)
    try:
        problems.extend("%s: %s" % (name, p) for p in (fn() or []))
    except Exception as e:
        problems.append("%s: %s %s" % (name, type(e).__name__, e))


def readme_layout():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## Repository layout\s*$", text, re.M)
    if not m:
        return ["the README has no Repository layout section"]
    nxt = re.search(r"^## ", text[m.end():], re.M)
    layout = text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]
    need = set()
    for f in tracked:
        parts = f.split("/")
        if len(parts) == 1:
            if not f.startswith("."):
                need.add(f)
            continue
        need.add(parts[0] + "/")
        if parts[0] in ("src", "data", "docs") and len(parts) > 2:
            need.add("%s/%s/" % (parts[0], parts[1]))
    missing = sorted(x for x in need if x not in layout)
    return ["the README Repository layout does not mention: %s" % missing] if missing else []


def functions(path):
    return {n.name for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}


def register():
    out = []
    for line in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| D\d+ \|", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 7:
            out.append("%s has %d cells, not 7" % (cells[0], len(cells)))
            continue
        rid, kind, ver, status = cells[0], cells[1], cells[5], cells[6]
        if kind not in ("Deviation", "Improvement", "Open"):
            out.append("%s: kind %r" % (rid, kind))
        if not (status.startswith("open") or status == "closed"):
            out.append("%s: status %r" % (rid, status))
        if ver.startswith("tests/"):
            rel, _, name = ver.partition("::")
            if not (ROOT / rel).exists():
                out.append("%s: missing file %s" % (rid, rel))
            elif name not in functions(ROOT / rel):
                out.append("%s: missing test %s" % (rid, name))
        elif ver.startswith("run:"):
            if not re.fullmatch(r"run:[0-9a-f]{32}", ver):
                out.append("%s: a run id has 32 hex characters" % rid)
        elif not (ver.startswith("open:") and status.startswith("open")):
            out.append("%s: a closed row needs a test or a run" % rid)
    return out


def syntax():
    out = []
    for f in staged:
        p = ROOT / f
        if f.endswith(".py"):
            try:
                ast.parse(p.read_text(encoding="utf-8"), f)
            except SyntaxError as e:
                out.append("%s line %s: %s" % (f, e.lineno, e.msg))
        elif f.endswith(".json"):
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except ValueError as e:
                out.append("%s: %s" % (f, e))
        elif f.endswith(".sh"):
            r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
            if r.returncode:
                out.append("%s: %s" % (f, r.stderr.strip()[:120]))
    return out


def workflows():
    return ["%s contains ': write'" % f for f in tracked if f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml")) and ": write" in (ROOT / f).read_text(encoding="utf-8")]


def secrets():
    return ["%s holds key-shaped text" % f for f in staged if Path(f).suffix in TEXT and (ROOT / f).is_file() and KEY.search((ROOT / f).read_bytes())]


check("readme layout", readme_layout)
check("register rows", register)
check("syntax of staged files", syntax)
check("workflow permissions", workflows)
check("key-shaped text", secrets)
if problems:
    print("PRE-PUSH CHECK FAILED (skip with SKIP_CHECK=1 only if you know why):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("pre-push check: %d rules replayed, none broken" % len(ran))
PY
