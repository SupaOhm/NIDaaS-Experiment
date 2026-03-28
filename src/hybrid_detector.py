"""
hybrid_detector.py
------------------
Reusable fusion logic for RF-only and hybrid prediction.
"""

from __future__ import annotations

import numpy as np


def predict_rf_only(
    scores_rf_all: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores_rf = scores_rf_all.copy()
    y_pred_rf = (scores_rf >= threshold).astype(int)
    return y_pred_rf, scores_rf


def predict_hybrid(
    scores_rf_all: np.ndarray,
    threshold: float,
    sig_hits: np.ndarray,
    delta: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    # RF is the primary decision maker
    rf_pred = (scores_rf_all >= threshold).astype(int)
    hybrid_pred = rf_pred.copy()
    hybrid_scores = scores_rf_all.copy()

    # Signature assists only in the gray zone
    gray_zone = (scores_rf_all >= (threshold - delta)) & (scores_rf_all < threshold)
    assist_idx = gray_zone & (sig_hits == 1)

    hybrid_pred[assist_idx] = 1
    hybrid_scores[assist_idx] = 1.0

    return hybrid_pred, hybrid_scores