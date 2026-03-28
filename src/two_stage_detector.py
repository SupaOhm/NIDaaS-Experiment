"""
two_stage_detector.py
---------------------
Two-stage detection model:

Stage 1: Logistic Regression screening
Stage 2: Random Forest confirmation

Design intent:
- Stage 1 provides broad, high-recall candidate screening
- Stage 2 confirms candidates with higher precision
- Final alert is raised only if both stages agree
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class TwoStageConfig:
    lr_c: float = 1.0
    lr_max_iter: int = 1000

    rf_n_estimators: int = 200
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 5

    screen_threshold: float = 0.2
    confirm_threshold: float = 0.8

    random_state: int = 42


class TwoStageLRRFDetector:
    def __init__(self, config: TwoStageConfig | None = None):
        self.config = config or TwoStageConfig()

        self.scaler = StandardScaler()

        self.stage1 = LogisticRegression(
            C=self.config.lr_c,
            max_iter=self.config.lr_max_iter,
            random_state=self.config.random_state,
        )

        self.stage2 = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_train_lr = self.scaler.fit_transform(X_train)

        self.stage1.fit(X_train_lr, y_train)
        self.stage2.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def stage1_scores(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_lr = self.scaler.transform(X)
        return self.stage1.predict_proba(X_lr)[:, 1]

    def stage2_scores(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.stage2.predict_proba(X)[:, 1]

    def predict_candidates(self, X: np.ndarray) -> np.ndarray:
        scores = self.stage1_scores(X)
        return (scores >= self.config.screen_threshold).astype(int)

    def predict_confirmed(self, X: np.ndarray) -> np.ndarray:
        scores = self.stage2_scores(X)
        return (scores >= self.config.confirm_threshold).astype(int)

    def predict(
        self,
        X: np.ndarray,
        screen_threshold: float | None = None,
        confirm_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        final_pred : np.ndarray
            Final two-stage binary predictions
        stage1_scores : np.ndarray
            Logistic Regression screening scores
        stage2_scores : np.ndarray
            Random Forest confirmation scores
        """
        self._check_fitted()

        s_thr = self.config.screen_threshold if screen_threshold is None else screen_threshold
        c_thr = self.config.confirm_threshold if confirm_threshold is None else confirm_threshold

        s1_scores = self.stage1_scores(X)
        s2_scores = self.stage2_scores(X)

        stage1_pred = (s1_scores >= s_thr).astype(int)
        final_pred = np.zeros(len(X), dtype=int)

        candidate_idx = np.where(stage1_pred == 1)[0]
        if len(candidate_idx) > 0:
            confirmed = (s2_scores[candidate_idx] >= c_thr).astype(int)
            final_pred[candidate_idx] = confirmed

        return final_pred, s1_scores, s2_scores

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("TwoStageLRRFDetector must be fitted before prediction.")