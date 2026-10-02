"""Sign-constrained logistic regression (docs/PHASE_C_PROTOCOL.md, Section 14). Offline: numpy, scipy and scikit-learn only.

The objective is scikit-learn's L2 logistic objective, 0.5 * ||w||^2 + C * (sum of log-loss), with the intercept
unpenalized, solved by L-BFGS-B so that chosen coefficients can be bounded. With no active bound the fit equals
scikit-learn's (tests/test_constrained.py)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.baseline import build_logistic
from src.models.cv import CATEGORICAL, CONSTRAINTS


def _objective(theta, X, y, C):
    w, b = theta[:-1], theta[-1]
    z = X @ w + b
    loss = 0.5 * float(w @ w) + C * float(np.logaddexp(0.0, -(2.0 * y - 1.0) * z).sum())
    r = expit(z) - y
    return loss, np.concatenate([w + C * (X.T @ r), [C * float(r.sum())]])


def padded_bounds(lower, upper, n):
    """Per-coefficient bounds as arrays of length n; entries beyond the given tuples (or None) mean unbounded."""
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    if lower is not None:
        lo[:len(lower)] = lower
    if upper is not None:
        hi[:len(upper)] = upper
    return lo, hi


class BoundedLogistic(ClassifierMixin, BaseEstimator):
    """Binary logistic regression (labels 0 and 1) with bounds on the first len(lower) or len(upper) coefficients."""

    def __init__(self, C=1.0, lower=None, upper=None, max_iter=5000, tol=1e-9):
        self.C = C
        self.lower = lower
        self.upper = upper
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        lo, hi = padded_bounds(self.lower, self.upper, X.shape[1])
        bounds = [(None if np.isinf(a) else a, None if np.isinf(b) else b) for a, b in zip(lo, hi)] + [(None, None)]
        theta0 = np.append(np.clip(0.0, lo, hi), 0.0)
        res = minimize(_objective, theta0, args=(X, y, self.C), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": self.tol})
        self.coef_, self.intercept_ = res.x[:-1], float(res.x[-1])
        self.classes_ = np.array([0, 1])
        self.n_iter_, self.converged_ = int(res.nit), bool(res.success)
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def constraint_bounds(features):
    """Bounds for the numeric block of build_logistic(features): rainfall >= 0 and elevation, distance, slope <= 0.
    Standardizing divides by a positive number, so the signs carry over from raw to standardized features."""
    num = [f for f in features if f not in CATEGORICAL]
    lower = tuple(0.0 if CONSTRAINTS.get(f) == 1 else -np.inf for f in num)
    upper = tuple(0.0 if CONSTRAINTS.get(f) == -1 else np.inf for f in num)
    return lower, upper


def build_constrained_logistic(features):
    """build_logistic(features) with its classifier replaced by the bounded one; same preprocessing."""
    pipe = build_logistic(features)
    lower, upper = constraint_bounds(features)
    pipe.steps[-1] = ("clf", BoundedLogistic(C=1.0, lower=lower, upper=upper))
    return pipe
