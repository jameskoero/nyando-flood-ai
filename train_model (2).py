import joblib; from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
MODEL_PATH=Path(__file__).parent.parent.parent/'models/nyando_xgb_v1.pkl'
def train_xgboost(X,y,tune=False):
    m=GradientBoostingClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.80,random_state=42)
    m.fit(X,y); return m
def save_model(m,path=None): p=Path(path) if path else MODEL_PATH; p.parent.mkdir(exist_ok=True); joblib.dump(m,p)
def load_model(path=None): return joblib.load(Path(path) if path else MODEL_PATH)
