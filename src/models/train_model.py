"""
Model training and serialization for Nyando Flood AI.

Model choice: GradientBoostingClassifier (sklearn)
Why not XGBoost or a neural network?
  - The dataset is small (~2308 rows, 6 features) — GBM generalises well
    without overfitting on small tabular data.
  - GBM has a feature_importances_ attribute that maps directly to
    the hydrological features, which is useful for the donor report.
  - XGBoost adds a C++ dependency that sometimes breaks on ARM64 (Termux).

Serialization: standard pickle instead of joblib.
Why the change?
  joblib is optimized for large numpy arrays (it memory-maps them).
  For a small GBM model, it adds a version-sensitive header that breaks
  cross-environment loading (e.g., Colab trains, Render serves, CI tests).
  Standard pickle is more portable. joblib is kept as a fallback loader
  for the existing nyando_xgb_v1.pkl artifact.
"""

import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "nyando_xgb_v1.pkl"
)


def train(X, y):
    """
    Train the flood prediction model.

    Hyperparameters:
      n_estimators=300   — enough trees to capture seasonal patterns
      max_depth=6        — prevents overfitting on 2308 samples
      learning_rate=0.05 — conservative to avoid high-variance fits
      subsample=0.8      — stochastic gradient boosting, reduces variance
      random_state=42    — reproducibility

    Returns the fitted estimator.
    """
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    return clf.fit(X, y)


def save(model, path=None):
    """
    Serialize model to disk using standard pickle.

    Creates parent directories if they don't exist
    (needed when running in a fresh CI environment).
    """
    p = Path(path) if path else MODEL_PATH
    p.parent.mkdir(exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(model, f)
    print(f"[train_model] Model saved → {p}")


def load(path=None):
    """
    Load a saved model.

    Tries pickle first (new format).
    Falls back to joblib for backward compatibility with the original
    nyando_xgb_v1.pkl which was serialized with joblib.
    """
    p = Path(path) if path else MODEL_PATH
    # Try standard pickle first
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    # Fallback: joblib (legacy format)
    try:
        import joblib
        return joblib.load(p)
    except Exception as e:
        raise RuntimeError(
            f"[train_model] Could not load model from {p}: {e}"
        )
