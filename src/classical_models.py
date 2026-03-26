"""
classical_models.py
-------------------
Baseline ML models for comparison with the proposed Hybrid (Signature + LSTM) method.

Models:
  - Random Forest (scikit-learn)
  - XGBoost (if installed)
  - Logistic Regression (lightweight fallback / sanity check)

Each model has a matching train_* / eval_* function pair.
eval_* always returns a dict of metrics compatible with metrics.py.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    print(f"[classical_models] Training RandomForest (n_estimators={n_estimators})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)
    return clf


def eval_rf(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Evaluate a fitted Random Forest and return metrics dict."""
    y_pred = model.predict(X)
    return _compute_metrics(y, y_pred, model_name="RandomForest")


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Any:
    """Train an XGBoost classifier. Raises ImportError if xgboost not installed."""
    if not _XGB_AVAILABLE:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        )
    print(f"[classical_models] Training XGBoost (n_estimators={n_estimators})...")
    clf = XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf


def eval_xgb(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Evaluate a fitted XGBoost model and return metrics dict."""
    y_pred = model.predict(X)
    return _compute_metrics(y, y_pred, model_name="XGBoost")


# ---------------------------------------------------------------------------
# Logistic Regression (fallback / sanity check)
# ---------------------------------------------------------------------------

def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a Logistic Regression classifier (fast baseline)."""
    print("[classical_models] Training LogisticRegression...")
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def eval_logreg(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Evaluate a fitted Logistic Regression and return metrics dict."""
    y_pred = model.predict(X)
    return _compute_metrics(y, y_pred, model_name="LogisticRegression")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Compute standard classification metrics and return as dict."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    # False Alarm Rate = FP / (FP + TN)
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "model":     model_name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "far":       round(far,  4),
    }
