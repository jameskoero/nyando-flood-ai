import json
from pathlib import Path

import pandas as pd


def audit_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    flooded_counts = df["flooded"].value_counts(dropna=False).to_dict() if "flooded" in df.columns else {}
    flood_rate = float(df["flooded"].mean()) if "flooded" in df.columns else None
    missing_rate = (df.isna().mean().sort_values(ascending=False)).to_dict()
    has_bom_headers = any(c.startswith("\ufeff") for c in df.columns)
    lon_stats = df["lon"].agg(["min", "max"]).to_dict() if "lon" in df.columns else {}
    lat_stats = df["lat"].agg(["min", "max"]).to_dict() if "lat" in df.columns else {}
    return {
        "csv_path": str(csv_path),
        "rows": total_rows,
        "columns": list(df.columns),
        "flooded_counts": flooded_counts,
        "flood_rate": flood_rate,
        "missing_rate": missing_rate,
        "has_bom_headers": has_bom_headers,
        "lon_range": lon_stats,
        "lat_range": lat_stats,
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]
    curated = repo_root / "data" / "training" / "nyando_training_v1.csv"
    raw = repo_root / "data" / "training" / "nyando_training_v1_raw_gee.csv"
    report = {
        "curated": audit_dataset(curated) if curated.exists() else {"missing": str(curated)},
        "raw_gee": audit_dataset(raw) if raw.exists() else {"missing": str(raw)},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
