"""
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
