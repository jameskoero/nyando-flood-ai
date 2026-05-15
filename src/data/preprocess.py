import numpy as np
import pandas as pd

FEATURES = ['elevation', 'slope', 'rainfall_3day', 'distance_river', 'clay_percent', 'land_cover']
TARGET = 'flooded'
MIN_ELEVATION = 0.1


def clean(df):
    out = df.dropna(subset=FEATURES + [TARGET]).copy()
    out['elevation'] = out['elevation'].clip(lower=MIN_ELEVATION)
    out['rainfall_3day'] = out['rainfall_3day'].clip(lower=0)
    return out.reset_index(drop=True)


def smote(X, y):
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    y_values = y.values if isinstance(y, pd.Series) else y
    y_values = np.asarray(y_values).astype(int)

    class_counts = {label: int((y_values == label).sum()) for label in np.unique(y_values)}
    if len(class_counts) < 2:
        return X, y

    majority_label = max(class_counts, key=class_counts.get)
    minority_label = min(class_counts, key=class_counts.get)
    n = class_counts[majority_label] - class_counts[minority_label]
    if n <= 0:
        return X, y

    mi = np.where(y_values == minority_label)[0]
    i = np.random.choice(mi, n, replace=True)
    j = np.random.choice(mi, n, replace=True)
    a = np.random.uniform(.2, .8, n)

    synthetic = X_values[i] * a[:, None] + X_values[j] * (1 - a[:, None])
    X_res = np.vstack([X_values, synthetic])
    y_res = np.concatenate([y_values, np.full(n, minority_label, int)])

    if isinstance(X, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
    return X_res, y_res


def clean_features(df):
    return clean(df)


def apply_smote(X, y):
    return smote(X, y)
