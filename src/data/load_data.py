"""
load_data.py — Data loading functions for Nyando Flood AI.
"""
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

FEATURES = ["elevation","slope","rainfall_3day",
            "distance_river","clay_percent","land_cover"]
TARGET   = "flooded"


def load_training_csv(version: str = "v1") -> pd.DataFrame:
    path = DATA_DIR / "training" / f"nyando_training_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training file not found: {path}\n"
            "Run notebooks/01_data_prep.ipynb to generate it first.")
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {len(df):,} rows from {path.name}")
    return df


def load_raw_chirps(year: int) -> pd.DataFrame:
    path = DATA_DIR / "raw" / f"chirps_nyando_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CHIRPS file not found: {path}")
    return pd.read_csv(path)


def load_external_source(name: str) -> pd.DataFrame:
    path = DATA_DIR / "external" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"External source not found: {path}")
    return pd.read_csv(path)
