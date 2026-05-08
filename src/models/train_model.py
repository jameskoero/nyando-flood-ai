"""
train_model.py — XGBoost training pipeline for Nyando Flood AI.
"""
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

BEST_PARAMS = {
    "n_estimators": 300, "max_depth": 6,
    "learning_rate": 0.05, "subsample": 0.80,
    "colsample_bytree": 0.80,
    "eval_metric": "auc", "random_state": 42,
}


def train_xgboost(X_train, y_train, tune: bool = False):
    base = xgb.XGBClassifier(**BEST_PARAMS)
    if tune:
        print("[train] Running GridSearchCV...")
        param_grid = {
            "max_depth": [4,6,8],
            "learning_rate": [0.01,0.05,0.1],
        }
        grid = GridSearchCV(base, param_grid, cv=5,
                            scoring="roc_auc", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"[train] Best params: {grid.best_params_}")
        return grid.best_estimator_
    print("[train] Training XGBoost...")
    base.fit(X_train, y_train)
    print("[train] Done.")
    return base


def save_model(model, name: str = "nyando_xgb_v1") -> Path:
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"[train] Model saved -> {path}")
    return path


def load_model(name: str = "nyando_xgb_v1"):
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run python main.py first.")
    model = joblib.load(path)
    print(f"[train] Model loaded <- {path}")
    return model
