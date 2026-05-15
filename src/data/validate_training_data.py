"""
Pre-test schema gate for the canonical training CSV.

Called by ci.yml BEFORE pytest runs. If this script fails, the CI
gives a clear error: "Canonical training CSV not found" or
"Missing required columns: [x]" — instead of a confusing
pytest collection crash (exit code 2).

Can also be run manually to check your local data:
  python src/data/validate_training_data.py

I added this after discovering that exit code 2 from pytest means
"collection failed" (usually an import error or missing file),
not "tests failed". This script converts that silent crash into
a loud, descriptive error.
"""

from pathlib import Path
import pandas as pd

FEATURES = [
    "elevation", "slope", "rainfall_3day",
    "distance_river", "clay_percent", "land_cover",
]
TARGET = "flooded"
REQUIRED_COLUMNS = {"lon", "lat", TARGET, *FEATURES}


def validate_schema(csv_path: Path):
    """
    Read CSV, strip BOM headers, check all required columns are present.
    Raises ValueError with a clear message if anything is missing.
    """
    df = pd.read_csv(csv_path)
    # Strip BOM (\ufeff) from column names — GEE always adds this
    normalized = {c.lstrip("\ufeff") for c in df.columns}
    missing = sorted(REQUIRED_COLUMNS - normalized)
    if missing:
        raise ValueError(
            f"[validate_training_data] Missing required columns in {csv_path}: "
            f"{missing}\n"
            f"  Found columns: {sorted(df.columns.tolist())}"
        )
    return df


def main():
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "data" / "training" / "nyando_training_v1.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"[validate_training_data] Canonical training CSV not found: "
            f"{csv_path}\n"
            "  Make sure data/training/nyando_training_v1.csv is committed."
        )

    validate_schema(csv_path)
    print(f"[validate_training_data] OK: {csv_path}")


if __name__ == "__main__":
    main()
