import pandas as pd; from pathlib import Path
FEATURES=['elevation','slope','rainfall_3day','distance_river','clay_percent','land_cover']; TARGET='flooded'
GITHUB_RAW='https://raw.githubusercontent.com/jameskoero/nyando-flood-ai/main/data/training/nyando_training_v1.csv'
def load_training_csv(version='v1'):
    local=Path(__file__).parent.parent.parent/f'data/training/nyando_training_{version}.csv'
    if local.exists(): return pd.read_csv(local)
    print('[load_data] Fetching from GitHub...'); return pd.read_csv(GITHUB_RAW)
