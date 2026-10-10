# Security baseline (V2)

Status: 2026-10-10. Scope: the public scoring API on Render, the dashboard on Vercel, CI on GitHub Actions and the repository. A baseline with evidence, not a penetration test.

## Threat model

| Asset | Threat | Control |
|---|---|---|
| Registered model | A file is swapped or corrupted | SHA-256 checked against the manifest before any load |
| Production dependencies | A known vulnerability ships | The image installs the CI-tested pins; weekly audit; Dependabot alerts and security updates |
| Secrets | A credential is committed | Secret scanning with push protection; tokens only through hidden prompts |
| Build pipeline | A workflow or action is hijacked | Read-only token permissions, SHA-pinned actions, a protected `main` that binds admins |
| API availability | Abuse or malformed input | Input validation, per-client and global rate limits, no OpenAPI schema |
| API responses | Browser-side attacks | Security headers on every response |
| Host | A compromised process gains root | The container runs as a non-root user |

Trust boundaries: the internet to the API (unauthenticated by design: it returns ranking scores and states its claim limit); a browser on the dashboard origin to the API; GitHub to Render (auto-deploy from `main`).

## Controls

| Control | Where | Checked by |
|---|---|---|
| Hash-verified model loading | `backend/integrity.py`, `backend/registered.py` | `tests/test_backend_integrity.py` |
| CI pins in the image, no multipart parser, non-root user | `Dockerfile`, `constraints-ci.txt` | `tests/test_security_baseline.py` |
| Weekly audit of the served set and the CI pins | `.github/workflows/security.yml` | `tests/test_security_baseline.py` |
| Action pinning, read-only permissions | `.github/workflows/ci.yml` | `tests/test_ci_hardening.py` |
| Security headers, rate limits, input validation | `backend/main.py` | `tests/test_v2_audit.py` |
| Disclosure policy | `SECURITY.md` | `tests/test_security_baseline.py` |
| Deviation register | `docs/ROADMAP_DEVIATIONS.md` | `tests/test_roadmap_conformance.py` |

## Verification log (2026-10-08 and 2026-10-10, from the owner's runs)

- pip-audit on the CI pins (`constraints-ci.txt`): no known vulnerabilities.
- pip-audit on the versions the Dockerfile used to pin, resolved on Python 3.11: 7 advisories in python-multipart 0.0.9 (fixed from 0.0.31) and 7 in starlette 0.36.3 (fixed from 1.3.1). CI could not see them because the image installed its own set (D32).
- The same serving set rebuilt under the CI constraints without python-multipart: no known vulnerabilities; the API tests passed (30 passed, 1 skipped).
- GitHub: secret scanning and push protection were already on; Dependabot alerts, security updates and CodeQL default setup were enabled and read back, and the first CodeQL run succeeded. Branch protection binds admins and requires `test`, `data-gate` and `manifest-check`.
- Before this change the live API sent no security headers beyond `access-control-allow-origin: *`.

## Open items

- Per-client rate limit: the cause of the earlier leak is not established. The limiter keys on the connection address, which behind Render's proxy may be the proxy's, and a client-set forwarding header may be trusted somewhere; neither is verified. The global cap depends on neither and protects availability. A fix needs the cause first, from a two-address measurement against the live API, because trusting a forwarding header blindly can make the limit easier to evade.
- CORS stays `*` by an earlier decision: the API is public, takes no cookies or credentials and returns ranking scores; CORS is a browser rule, not access control.
- The dashboard's own security headers are not set: a Vercel preview build failed on a `vercel.json` change earlier and the project settings were never seen. Vercel already sends HSTS.
- D32 stays open for its other half: installs are pinned but not hash-verified.
- No authentication on the public API, by design on the free tier (D45).
