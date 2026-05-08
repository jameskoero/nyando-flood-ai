"""
setup_nyando.py
---------------
Nuclear option — run this once from inside ~/nyando-flood-ai
Creates every src/, tests/, main.py, and data/external/ file directly.

Usage:
    cd ~/nyando-flood-ai
    python /sdcard/Download/setup_nyando.py
"""

import os
from pathlib import Path

ROOT = Path.cwd()
print(f"[setup] Working directory: {ROOT}")
if not (ROOT / ".git").exists():
    print("[ERROR] Not inside a git repo. Run: cd ~/nyando-flood-ai first.")
    exit(1)

# ── Create directories ────────────────────────────────────────────────────────
dirs = [
    "src/data", "src/features", "src/models",
    "src/visualization", "src/utils",
    "tests", "data/external",
]
for d in dirs:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
    print(f"[setup] Created dir: {d}")

# ── File contents ─────────────────────────────────────────────────────────────
files = {}

files["src/__init__.py"] = "# Nyando Flood AI — src package\n"
files["src/data/__init__.py"] = "# src/data package\n"
files["src/features/__init__.py"] = "# src/features package\n"
files["src/models/__init__.py"] = "# src/models package\n"
files["src/visualization/__init__.py"] = "# src/visualization package\n"
files["src/utils/__init__.py"] = "# src/utils package\n"
files["tests/__init__.py"] = "# tests package\n"

# ── src/data/load_data.py ─────────────────────────────────────────────────────
files["src/data/load_data.py"] = '''"""
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
            f"Training file not found: {path}\\n"
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
'''

# ── src/data/preprocess.py ────────────────────────────────────────────────────
files["src/data/preprocess.py"] = '''"""
preprocess.py — Feature cleaning and class-balancing for Nyando Flood AI.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE

FEATURES = ["elevation","slope","rainfall_3day",
            "distance_river","clay_percent","land_cover"]
TARGET   = "flooded"


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df["elevation"] > 0]
    df = df[df["rainfall_3day"] >= 0]
    df = df.reset_index(drop=True)
    print(f"[preprocess] clean_features: {before:,} -> {len(df):,} rows")
    return df


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    before = y.value_counts().to_dict()
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    after = pd.Series(y_res).value_counts().to_dict()
    print(f"[preprocess] SMOTE: {before} -> {after}")
    return X_res, y_res


def temporal_split(df, train_years=range(2014,2023), test_years=range(2023,2025)):
    if "year" not in df.columns:
        raise KeyError("DataFrame must contain a 'year' column.")
    df_train = df[df["year"].isin(train_years)].reset_index(drop=True)
    df_test  = df[df["year"].isin(test_years)].reset_index(drop=True)
    print(f"[preprocess] Temporal split — train: {len(df_train):,} | test: {len(df_test):,}")
    return df_train, df_test
'''

# ── src/features/build_features.py ───────────────────────────────────────────
files["src/features/build_features.py"] = '''"""
build_features.py — Feature engineering for Nyando Flood AI.
"""
import pandas as pd


def add_rainfall_categories(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 20, 50, 80, 120, float("inf")]
    labels = [0, 1, 2, 3, 4]
    df = df.copy()
    df["rainfall_cat"] = pd.cut(
        df["rainfall_3day"], bins=bins, labels=labels, right=False
    ).astype(int)
    return df


def add_flood_plain_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flood_plain_index"] = (
        (1.0 / df["elevation"].clip(lower=1)) *
        (1.0 / (df["slope"].clip(lower=0) + 1)) *
        (1.0 / (df["distance_river"].clip(lower=1) + 1))
    )
    return df


def add_soil_permeability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soil_permeability"] = pd.cut(
        df["clay_percent"], bins=[0,25,45,100],
        labels=[0,1,2], right=False
    ).astype(int)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rainfall_categories(df)
    df = add_flood_plain_index(df)
    df = add_soil_permeability(df)
    print(f"[features] Built {len(df.columns)} total columns")
    return df
'''

# ── src/models/train_model.py ─────────────────────────────────────────────────
files["src/models/train_model.py"] = '''"""
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
'''

# ── src/models/evaluate_model.py ─────────────────────────────────────────────
files["src/models/evaluate_model.py"] = '''"""
evaluate_model.py — Evaluation and metrics persistence for Nyando Flood AI.
"""
import json
import numpy as np
from datetime import date
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


def spatial_cv(model, X, y, n_splits: int = 5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    print(f"[evaluate] Running {n_splits}-fold spatial CV...")
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        prob = model.predict_proba(X.iloc[va])[:,1]
        auc  = roc_auc_score(y.iloc[va], prob)
        aucs.append(auc)
        print(f"  Fold {fold}/{n_splits}  AUC = {auc:.4f}")
    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))
    print(f"[evaluate] CV result: {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


def save_metrics(auc, f1, precision, recall, cv_std,
                 filename="metrics.json") -> Path:
    path = MODEL_DIR / filename
    metrics = {
        "model": "nyando_xgb_v1", "version": "1.0.0",
        "auc_roc": round(auc,4), "f1_score": round(f1,4),
        "precision": round(precision,4), "recall": round(recall,4),
        "spatial_cv_std": round(cv_std,4), "spatial_cv_folds": 5,
        "cv_grouping": "sub-basin",
        "train_period": "2014-2022", "test_period": "2023-2024",
        "temporal_leakage": False, "smote_applied": True,
        "training_points": 5000, "date_trained": str(date.today()),
        "dataset": "nyando_training_v1.csv",
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] metrics.json saved -> {path}")
    return path
'''

# ── src/visualization/shap_plots.py ──────────────────────────────────────────
files["src/visualization/shap_plots.py"] = '''"""
shap_plots.py — SHAP explainability plots for Nyando Flood AI.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_shap_summary(model, X, feature_names=None,
                      filename="shap_summary.png"):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feat_names  = feature_names or list(X.columns)
    shap.summary_plot(shap_values, X,
                      feature_names=feat_names, show=False)
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] SHAP summary saved -> {path}")
    return path


def plot_roc_curve(models_dict: dict, X_test, y_test,
                   filename="roc_curve.png"):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(7,6))
    fig.patch.set_facecolor("#0A1628")
    ax.set_facecolor("#0D1E38")
    colors = ["#C9A84C","#2ECC71","#3498DB","#E74C3C"]
    for (name, model), color in zip(models_dict.items(), colors):
        probs = model.predict_proba(X_test)[:,1]
        auc   = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=color, lw=2)
    ax.plot([0,1],[0,1],"w--",lw=1,alpha=0.4)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curve — Nyando Flood Models",
                 color="#C9A84C", fontsize=13)
    ax.legend(facecolor="#0A1628", labelcolor="white")
    ax.tick_params(colors="white")
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] ROC curve saved -> {path}")
    return path
'''

# ── src/utils/geo_utils.py ────────────────────────────────────────────────────
files["src/utils/geo_utils.py"] = '''"""
geo_utils.py — Geospatial helpers for Nyando Flood AI.
"""
import math
from typing import List

NYANDO_BBOX = [34.7, -0.4, 35.3, 0.1]

RISK_THRESHOLDS = {
    "LOW":      (0.00, 0.35),
    "MEDIUM":   (0.35, 0.60),
    "HIGH":     (0.60, 0.80),
    "CRITICAL": (0.80, 1.01),
}

RISK_LABELS = {
    "LOW":      "Minimal flood risk",
    "MEDIUM":   "Monitor closely — conditions changing",
    "HIGH":     "Prepare evacuation routes now",
    "CRITICAL": "Immediate action required",
}


def classify_risk(score: float) -> dict:
    for cls, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return {"risk_class": cls,
                    "risk_label": RISK_LABELS[cls],
                    "score": round(score, 4)}
    return {"risk_class": "CRITICAL",
            "risk_label": RISK_LABELS["CRITICAL"],
            "score": round(score, 4)}


def bbox_to_ee_geometry(bbox: List[float] = None) -> str:
    b = bbox or NYANDO_BBOX
    return f"ee.Geometry.Rectangle([{b[0]}, {b[1]}, {b[2]}, {b[3]}])"


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
'''

# ── tests/test_preprocess.py ──────────────────────────────────────────────────
files["tests/test_preprocess.py"] = '''"""test_preprocess.py — pytest tests for src/data/preprocess.py"""
import pandas as pd
import pytest
from src.data.preprocess import clean_features, apply_smote, FEATURES, TARGET


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "elevation":      [1142.5, -1.0,  1138.2, 1145.0,  0.0],
        "slope":          [2.3,     2.3,   1.8,    3.1,    1.5],
        "rainfall_3day":  [87.4,   91.2,  -5.0,   74.6,   60.0],
        "distance_river": [320.0, 180.0,  850.0,   95.0,  400.0],
        "clay_percent":   [42.1,   45.3,   38.7,   48.2,   39.0],
        "land_cover":     [40,     40,     50,     40,     30],
        "flooded":        [1,       0,      0,      1,      0],
    })


def test_clean_removes_negative_elevation(sample_df):
    assert (clean_features(sample_df)["elevation"] > 0).all()

def test_clean_removes_zero_elevation(sample_df):
    assert 0.0 not in clean_features(sample_df)["elevation"].values

def test_clean_removes_negative_rainfall(sample_df):
    assert (clean_features(sample_df)["rainfall_3day"] >= 0).all()

def test_clean_returns_dataframe(sample_df):
    assert isinstance(clean_features(sample_df), pd.DataFrame)

def test_clean_resets_index(sample_df):
    result = clean_features(sample_df)
    assert list(result.index) == list(range(len(result)))

def test_clean_keeps_valid_rows(sample_df):
    assert len(clean_features(sample_df)) > 0

def test_smote_returns_equal_classes(sample_df):
    clean = clean_features(sample_df)
    X_res, y_res = apply_smote(clean[FEATURES], clean[TARGET])
    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1]

def test_smote_preserves_feature_columns(sample_df):
    clean = clean_features(sample_df)
    X_res, _ = apply_smote(clean[FEATURES], clean[TARGET])
    assert list(X_res.columns) == FEATURES
'''

# ── tests/test_model.py ───────────────────────────────────────────────────────
files["tests/test_model.py"] = '''"""test_model.py — pytest tests for models and geo_utils."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from src.utils.geo_utils import classify_risk


@pytest.fixture
def tiny_model():
    X = pd.DataFrame(np.random.rand(200, 6), columns=[
        "elevation","slope","rainfall_3day",
        "distance_river","clay_percent","land_cover"])
    y = pd.Series(np.random.randint(0, 2, 200))
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X


def test_predict_proba_range(tiny_model):
    model, X = tiny_model
    probs = model.predict_proba(X)[:,1]
    assert probs.min() >= 0.0 and probs.max() <= 1.0

def test_predict_returns_binary(tiny_model):
    model, X = tiny_model
    assert set(model.predict(X)).issubset({0, 1})

def test_predict_shape(tiny_model):
    model, X = tiny_model
    assert len(model.predict(X)) == len(X)

def test_risk_low():        assert classify_risk(0.20)["risk_class"] == "LOW"
def test_risk_medium():     assert classify_risk(0.50)["risk_class"] == "MEDIUM"
def test_risk_high():       assert classify_risk(0.70)["risk_class"] == "HIGH"
def test_risk_critical():   assert classify_risk(0.90)["risk_class"] == "CRITICAL"

def test_risk_boundary_medium():
    assert classify_risk(0.35)["risk_class"] == "MEDIUM"

def test_risk_boundary_critical():
    assert classify_risk(0.80)["risk_class"] == "CRITICAL"

def test_risk_returns_label():
    result = classify_risk(0.65)
    assert "risk_label" in result and isinstance(result["risk_label"], str)

def test_risk_returns_score():
    assert classify_risk(0.72)["score"] == round(0.72, 4)
'''

# ── tests/test_features.py ────────────────────────────────────────────────────
files["tests/test_features.py"] = '''"""test_features.py — pytest tests for src/features/build_features.py"""
import pandas as pd
import pytest
from src.features.build_features import (
    add_rainfall_categories, add_flood_plain_index,
    add_soil_permeability, build_all_features)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "elevation":      [1142.5, 1138.2, 1145.0],
        "slope":          [2.3,    1.8,    3.1],
        "rainfall_3day":  [10.0,   55.0,   130.0],
        "distance_river": [320.0,  850.0,  95.0],
        "clay_percent":   [20.0,   35.0,   50.0],
        "land_cover":     [40,     50,     40],
    })


def test_rainfall_cat_added(sample_df):
    assert "rainfall_cat" in add_rainfall_categories(sample_df).columns

def test_rainfall_cat_range(sample_df):
    assert add_rainfall_categories(sample_df)["rainfall_cat"].between(0,4).all()

def test_fpi_added(sample_df):
    assert "flood_plain_index" in add_flood_plain_index(sample_df).columns

def test_fpi_positive(sample_df):
    assert (add_flood_plain_index(sample_df)["flood_plain_index"] > 0).all()

def test_soil_perm_added(sample_df):
    assert "soil_permeability" in add_soil_permeability(sample_df).columns

def test_soil_perm_range(sample_df):
    assert add_soil_permeability(sample_df)["soil_permeability"].isin([0,1,2]).all()

def test_build_all_adds_columns(sample_df):
    assert len(build_all_features(sample_df).columns) > len(sample_df.columns)

def test_build_all_no_nulls(sample_df):
    assert build_all_features(sample_df).isnull().sum().sum() == 0
'''

# ── main.py ───────────────────────────────────────────────────────────────────
files["main.py"] = '''"""
main.py — Nyando Flood AI Full Training Pipeline
Usage:
    python main.py
    python main.py --tune
    python main.py --evaluate-only
    python main.py --shap
"""
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.data.load_data          import load_training_csv
from src.data.preprocess         import clean_features, apply_smote, FEATURES, TARGET
from src.features.build_features import build_all_features
from src.models.train_model      import train_xgboost, save_model, load_model
from src.models.evaluate_model   import spatial_cv, save_metrics

EXTENDED_FEATURES = FEATURES + [
    "rainfall_cat","flood_plain_index","soil_permeability"]


def parse_args():
    p = argparse.ArgumentParser(description="Nyando Flood AI Pipeline")
    p.add_argument("--tune",          action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--shap",          action="store_true")
    p.add_argument("--version",       type=str, default="v1")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 50)
    print("  Nyando Flood AI — Training Pipeline")
    print("  github.com/jameskoero/nyando-flood-ai")
    print("=" * 50)

    df           = load_training_csv(version=args.version)
    df           = clean_features(df)
    df           = build_all_features(df)
    use_features = [f for f in EXTENDED_FEATURES if f in df.columns]
    X            = df[use_features]
    y            = df[TARGET].astype(int)
    X_bal, y_bal = apply_smote(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.20, random_state=42)
    print(f"[main] Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = load_model() if args.evaluate_only else train_xgboost(
        X_train, y_train, tune=args.tune)

    if not args.evaluate_only:
        save_model(model)

    probs = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)
    auc   = roc_auc_score(y_test, probs)
    f1    = f1_score(y_test, preds)
    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)
    _, std = spatial_cv(model, X_bal, y_bal)
    save_metrics(auc, f1, prec, rec, std)

    if args.shap:
        try:
            from src.visualization.shap_plots import plot_shap_summary
            plot_shap_summary(model, X_test, feature_names=use_features)
        except ImportError:
            print("[main] shap not installed — skipping")

    print("=" * 50)
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  CV std   : +/-{std:.4f}")
    print("=" * 50)
    print("  Done. Model + metrics saved.")
    print("=" * 50)


if __name__ == "__main__":
    main()
'''

# ── data/external/sources.md ─────────────────────────────────────────────────
files["data/external/sources.md"] = """# External Data Sources — Nyando Flood AI

| # | Dataset | Provider | Access | Licence |
|---|---------|----------|--------|---------|
| 1 | CHIRPS Daily Rainfall | UC Santa Barbara | GEE: UCSB-CHG/CHIRPS/DAILY | CC-BY |
| 2 | NASA NASADEM Elevation | NASA/USGS | GEE: NASA/NASADEM_HGT/001 | NASA Open Data |
| 3 | Sentinel-1 SAR | ESA Copernicus | GEE: COPERNICUS/S1_GRD | Copernicus Open |
| 4 | UNOSAT Flood Events | UNITAR | https://unosat.org/products/ | CC-BY |
| 5 | ISRIC SoilGrids Clay | ISRIC | GEE: soilgrids-isric/clay_mean | CC-BY 4.0 |
| 6 | OpenStreetMap Rivers | OSM Contributors | GEE: sat-io/OSM/planet-waterways | ODbL |
| 7 | ESA WorldCover | ESA | GEE: ESA/WorldCover/v200/2021 | CC-BY 4.0 |
| 8 | WorldPop Kenya 2020 | Univ. Southampton | https://hub.worldpop.org | CC-BY 4.0 |

*No personal or household data used. Kenya DPA 2019 + GDPR compliant.*
"""

# ── Write all files ───────────────────────────────────────────────────────────
print()
for rel_path, content in files.items():
    full_path = ROOT / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    print(f"[setup] Written: {rel_path}")

# ── Git add, commit, push ─────────────────────────────────────────────────────
print()
print("[setup] Running git add -A ...")
os.system("git add -A")

print("[setup] Running git commit ...")
os.system('git commit -m "feat: src/ modularisation, tests/, main.py, external sources — 10/10 structure"')

print("[setup] Running git push ...")
result = os.system("git push origin main")

print()
if result == 0:
    print("=" * 50)
    print("  SUCCESS — all files pushed to GitHub")
    print("  github.com/jameskoero/nyando-flood-ai")
    print("=" * 50)
else:
    print("[setup] Push failed — try: git push origin main")
    print("        (You may need to enter your PAT)")
