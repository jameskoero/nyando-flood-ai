import pickle
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
MODEL_PATH=Path(__file__).parent.parent.parent/'models/nyando_xgb_v1.pkl'
def train(X,y): return GradientBoostingClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,random_state=42).fit(X,y)
def save(m,path=None):
    p=Path(path) if path else MODEL_PATH
    p.parent.mkdir(exist_ok=True)
    with open(p,'wb') as f: pickle.dump(m,f)
def load(path=None):
    p=Path(path) if path else MODEL_PATH
    try:
        with open(p,'rb') as f: return pickle.load(f)
    except Exception:
        return joblib.load(p)
