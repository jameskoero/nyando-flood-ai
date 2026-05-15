"""
Diagnostic tool: compare the curated vs raw GEE training CSV.

Run manually to investigate data quality issues:
  python src/data/audit_real_gee.py

I wrote this after noticing that nyando_training_v1.csv has only
~2 flooded=1 records out of 2308, while the raw GEE export
(nyando_training_v1_raw_gee.csv) has 1150/5000.

This script surfaces:
  - Row counts
  - Flood rate (% of rows where flooded=1)
  - Missing value rates per column
  - Whether BOM headers are present
  - Lon/lat coordinate ranges (sanity check: Nyando is lon 34.7-35.4, lat 0.4S-0.1N)

The goal is to find where the bad filtering happened in the
GEE → curated pipeline so we can fix the label generation
and retrain on clean data.
"""

import json
from pathlib import Path
import pandas as pd


def audit_dataset(csv_path: Path) -> dict:
    """
    Return a dict of quality metrics for a training CSV.
    Handles missing files gracefully (returns {"missing": path}).
    """
    if not csv_path.exists():
        return {"missing": str(csv_path)}

    df = pd.read_csv(csv_path)

    # Check for BOM headers (GEE export artifact)
    has_bom_headers = any(c.startswith("\ufeff") for c in df.columns)

    # Flood rate
    flooded_counts = (
        df["flooded"].value_counts(dropna=False).to_dict()
        if "flooded" in df.columns else {}
    )
    flood_rate = (
        float(df["flooded"].mean()) if "flooded" in df.columns else None
    )

    # Missing rates
    missing_rate = (
        df.isna().mean().sort_values(ascending=False).to_dict()
    )

    # Coordinate ranges (lon/lat sanity check)
    lon_stats = (
        df["lon"].agg(["min", "max"]).to_dict() if "lon" in df.columns else {}
    )
    lat_stats = (
        df["lat"].agg(["min", "max"]).to_dict() if "lat" in df.columns else {}
    )

    return {
        "csv_path":       str(csv_path),
        "rows":           len(df),
        "columns":        list(df.columns),
        "flooded_counts": flooded_counts,
        "flood_rate":     flood_rate,
        "missing_rate":   missing_rate,
        "has_bom_headers": has_bom_headers,
        "lon_range":      lon_stats,
        "lat_range":      lat_stats,
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]
    curated = repo_root / "data" / "training" / "nyando_training_v1.csv"
    raw     = repo_root / "data" / "training" / "nyando_training_v1_raw_gee.csv"

    report = {
        "curated": audit_dataset(curated) if curated.exists()
                   else {"missing": str(curated)},
        "raw_gee": audit_dataset(raw) if raw.exists()
                   else {"missing": str(raw)},
    }
    print(json.dumps(report, indent=2))

    # Highlight the imbalance if both files present
    if "flood_rate" in report["curated"] and "flood_rate" in report["raw_gee"]:
        c_rate = report["curated"]["flood_rate"] or 0
        r_rate = report["raw_gee"]["flood_rate"] or 0
        if c_rate < 0.01 and r_rate > 0.1:
            print(
                "\n⚠️  LABEL IMBALANCE WARNING:\n"
                f"   Curated CSV flood rate: {c_rate:.2%}\n"
                f"   Raw GEE CSV flood rate: {r_rate:.2%}\n"
                "   The filtering/labeling step that produced the curated CSV\n"
                "   appears to have dropped most flood records. Investigate\n"
                "   the GEE label generation notebook before retraining."
            )


if __name__ == "__main__":
    main()
