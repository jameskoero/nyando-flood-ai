from pathlib import Path

import pandas as pd

FEATURES = ["elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover"]
TARGET = "flooded"
REQUIRED_COLUMNS = {"lon", "lat", TARGET, *FEATURES}


def validate_schema(csv_path: Path):
    df = pd.read_csv(csv_path)
    normalized = {c.lstrip("\ufeff") for c in df.columns}
    missing = sorted(REQUIRED_COLUMNS - normalized)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def main():
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "data" / "training" / "nyando_training_v1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Canonical training CSV not found: {csv_path}")
    validate_schema(csv_path)
    print(f"[validate_training_data] OK: {csv_path}")


if __name__ == "__main__":
    main()
