# CI Fix: Restoring Feature Contracts and Enforcing Canonical GEE Schema

**Author:** James Koero  
**Date:** May 2026  
**Commit type:** Understanding review — I reviewed, documented, and own these changes.

---

## Why the CI was red (my diagnosis)

After going through the GitHub Actions logs carefully, I found four real problems.
None of them were cosmetic. They were structural.

### 1. The test suite expected functions that did not exist in the source

`tests/test_pipeline.py` was importing:
- `add_rainfall_categories` from `src.features.build_features`
- `clean_features`, `apply_smote`, `FEATURES`, `TARGET` from `src.data.preprocess`

But the actual source files only had a monolithic `build()` and a numpy-based
`clean()` / `smote()` that did not export those names.

This is called a **broken API contract**. pytest exits with code 2 (collection failure)
before a single test even runs. That's why fixing test failures one by one was
never going to work — the runner couldn't collect the tests at all.

**My fix:** Refactored both modules into named, single-responsibility functions
that match what the tests expect. Kept backward-compatible aliases (`build`,
`clean`, `smote`) so nothing else breaks.

### 2. The raw GEE CSV was poisoning the pipeline

Google Earth Engine exports CSVs with a BOM (Byte Order Mark: `\ufeff`) prepended
to the first column header. So `elevation` becomes `\ufeffelevation`.

This means:
- Schema checks silently fail
- The column is unreadable by name
- `test_pipeline.py` was also using `glob("*.csv")` which could pick the raw GEE
  file (`nyando_training_v1_raw_gee.csv`) before the curated training file

I understand this from working with GEE exports during the Nyando data collection
phase. The fix is: strip BOM headers at read time and pin the canonical CSV
explicitly in tests.

### 3. Extreme class imbalance in the curated training CSV

`nyando_training_v1.csv` has 2308 rows but only **2 flooded=1 records**.
That is a 0.09% positive rate. For a flood prediction model, this is
a labeling error, not a real-world distribution.

The raw GEE file (`nyando_training_v1_raw_gee.csv`) has 1150 positives out of 5000,
which is much more realistic for the Nyando basin (which floods seasonally).

The filtering/labeling step that produced the curated CSV introduced a bug.
I've added `audit_real_gee.py` to surface this clearly so it can be corrected
in the next data pipeline run.

### 4. GitHub Actions using deprecated Node.js 16 action versions

`actions/checkout@v4` and `actions/setup-python@v5` both use Node.js 16
which GitHub deprecated. The CI was showing Node.js 20 deprecation warnings
that were obscuring the real import errors. Updated to `@v5` and `@v6`.

---

## What I changed and why

### `src/features/build_features.py`
Split the monolithic `build()` into four named functions. Each function has
a single hydrological responsibility:
- `add_rainfall_categories`: bins 3-day rainfall into intensity levels
  (0=dry, 1=light, 2=moderate, 3=heavy, 4=extreme). Thresholds based on
  Kenya Meteorological Department rainfall classification.
- `add_flood_plain_index`: computes elevation/slope ratio as a proxy for
  how much water accumulates in a given cell. Flat, low areas flood first.
- `add_soil_permeability`: bins clay percentage into permeability classes.
  High clay = low permeability = longer waterlogging after rain.
- `build_all_features`: orchestrates all three in sequence.

### `src/data/preprocess.py`
Rewrote the SMOTE implementation. The original used raw numpy array indexing
which silently broke when the DataFrame had non-contiguous indices (which
happens after `.dropna()` and `.query()` filtering). The new version uses
pandas `.loc[]` throughout and resets the index at entry. Also added
`FEATURES` and `TARGET` as module-level constants so any downstream
module can import them without hardcoding column names.

### `src/data/load_data.py`
Added `_validate()` that strips BOM characters from column names before
checking schema. Also added `REQUIRED_COLUMNS` constant. This catches
bad GEE exports at load time rather than letting them silently corrupt
the feature matrix.

### `src/models/train_model.py`
Changed from `joblib.dump` to `pickle.dump` with `open(path, 'wb')`.
`joblib` is optimized for numpy arrays but adds a version-sensitive
serialization layer. Standard `pickle` is more portable across environments
(Colab → Render → CI runner) and easier to audit. Added `joblib` as a
fallback loader for backward compatibility with the existing `.pkl` file.

### `tests/test_pipeline.py`
- Added `CANONICAL_CSV` constant pointing to `nyando_training_v1.csv`
- Added `_assert_schema()` helper that checks required columns and strips BOM
- Replaced fragile `glob("*.csv")[0]` with explicit canonical CSV selection
- Added `test_canonical_csv_selected` and `test_required_schema_columns`

### `.github/workflows/ci.yml`
- Upgraded `actions/checkout@v4` → `@v5`
- Upgraded `actions/setup-python@v5` → `@v6`
- Added schema validation step before pytest:
  `python src/data/validate_training_data.py`
  This fails fast if the training CSV is missing or malformed, giving a
  clear error message instead of a confusing pytest collection crash.

### `src/data/validate_training_data.py` (new)
A standalone script that checks the canonical training CSV exists and
has all required columns. Called by CI before tests run. Also callable
locally: `python src/data/validate_training_data.py`

### `src/data/audit_real_gee.py` (new)
A diagnostic script for comparing the curated vs raw GEE CSV. Reports:
- Row count, flood rate, missing rate
- BOM headers present/absent
- Lon/lat coordinate ranges
Run manually: `python src/data/audit_real_gee.py`
This helped me discover the 2/2308 label imbalance bug documented above.

---

## What I learned from this

1. **pytest exit code 2 is a collection failure, not a test failure.** Always
   check if tests are even being collected before debugging individual assertions.

2. **GEE CSV exports always have BOM headers.** Strip `\ufeff` from column names
   at read time. This is a one-liner but it must be there.

3. **API contracts between modules must be explicit.** If a test imports a name,
   that name must be exported from the source. Monolithic functions that work
   internally but don't expose their components break test-driven development.

4. **Class imbalance in flood labels needs investigation.** 2/2308 is not a
   real flood rate for the Nyando basin. The GEE label generation step needs
   to be re-examined before the next model retraining.
