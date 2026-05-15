import pandas as pd; from pathlib import Path
FEATURES=['elevation','slope','rainfall_3day','distance_river','clay_percent','land_cover']
TARGET='flooded'
# Real GEE data — 2308 observations, coordinates lon 34.7-35.4°E, lat 0.4°S-0.1°N
GITHUB_RAW='https://raw.githubusercontent.com/jameskoero/nyando-flood-ai/main/data/training/nyando_training_v1.csv'
def load_training_csv():
    local=Path(__file__).parent.parent.parent/'data/training/nyando_training_v1.csv'
    if local.exists(): return pd.read_csv(local)
    print('[load_data] Fetching from GitHub...'); return pd.read_csv(GITHUB_RAW)
def load_raw_gee():
    local=Path(__file__).parent.parent.parent/'data/training/nyando_training_v1_raw_gee.csv'
    if local.exists(): return pd.read_csv(local)
    raise FileNotFoundError('Raw GEE CSV not found. Run notebooks/01_gee_data_extraction.ipynb')
