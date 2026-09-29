# Provenance note: Copilot as a listed contributor

GitHub lists "Copilot" as a contributor to this repository. This is why.

- Two commits authored by `copilot-swe-agent[bot]`, both dated 2026-05-15:
  - `8f0a76b` - restore preprocessing/features contracts and enforce the canonical real-GEE schema
  - `2ef58a4` - tighten canonical dataset selection and loader/error handling
- They entered `main` through pull request #1 (`copilot/identify-ci-failure-causes`), merged 2026-05-15.
- Files touched: the legacy pipeline (`src/data`, `src/features`, `src/models`), `tests/test_pipeline.py`, `.github/workflows/ci.yml`, and the legacy model binary `models/nyando_xgb_v1.pkl`.
- That binary is recorded as legacy in `models/MANIFEST.json`. The V2 rebuild (Phase A onward) does not use it.
- A search of history for Copilot co-author trailers found none. The two commits above are the only source of the entry.

This note is a factual record. History is not rewritten, so existing commit hashes and DOI-linked references stay valid.
