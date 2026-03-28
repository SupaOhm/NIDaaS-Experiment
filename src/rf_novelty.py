"""
rf_novelty.py
-------------
Implements Random-Forest-based novelty scoring for intrusion detection.

Main idea:
- Train only on normal data
- Generate synthetic pseudo-anomalies by permuting feature columns
- Train a Random Forest to distinguish:
    0 = real normal samples
    1 = synthetic reference samples

At inference time:
- The model computes a novelty score for each sample
- Higher score means the sample looks less like normal behavior
- A threshold is fitted from validation-normal data using a quantile

Score components:
1. Probability that a sample resembles synthetic behavior
2. Rarity of the leaf regions reached by the sample compared with normal training data

This module is designed for:
- novelty detection experiments
- low-latency hybrid IDS pipelines
- replacing heavier anomaly models such as LSTM in efficiency-focused settings
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class RFNoveltyConfig:
    n_estimators: int = 200
    max_depth: int = 12
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    random_state: int = 42
    alpha: float = 0.7
    threshold_quantile: float = 0.99


class RFNoveltyScorer:
    """
    Random-Forest-based novelty scoring.

    Train on:
      - real normal samples
      - synthetic pseudo-anomalies built by per-column permutation

    Score combines:
      1) probability sample looks synthetic
      2) rarity of the reached leaves compared with normal training data
    """

    def __init__(self, config: Optional[RFNoveltyConfig] = None):
        self.config = config or RFNoveltyConfig()

        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.threshold_: Optional[float] = None
        self.leaf_mass_: list[dict[int, float]] = []

    def _make_synthetic(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_state)
        X_syn = np.empty_like(X, dtype=float)

        for j in range(X.shape[1]):
            X_syn[:, j] = rng.permutation(X[:, j])

        return X_syn

    def fit(self, X_normal: np.ndarray) -> "RFNoveltyScorer":
        X_normal = np.asarray(X_normal, dtype=float)
        if X_normal.ndim != 2:
            raise ValueError("X_normal must be a 2D array.")
        if len(X_normal) == 0:
            raise ValueError("X_normal is empty.")

        X_syn = self._make_synthetic(X_normal)

        X_train = np.vstack([X_normal, X_syn])
        y_train = np.concatenate([
            np.zeros(len(X_normal), dtype=int),
            np.ones(len(X_syn), dtype=int),
        ])

        self.model.fit(X_train, y_train)

        normal_leaves = self.model.apply(X_normal)
        self.leaf_mass_ = []

        for t in range(normal_leaves.shape[1]):
            leaves = normal_leaves[:, t]
            vals, counts = np.unique(leaves, return_counts=True)
            mass = {int(v): float(c) / float(len(X_normal)) for v, c in zip(vals, counts)}
            self.leaf_mass_.append(mass)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        p_syn = self.model.predict_proba(X)[:, 1]

        leaves = self.model.apply(X)
        leaf_mass = np.zeros(len(X), dtype=float)

        for i in range(len(X)):
            masses = []
            for t, leaf_id in enumerate(leaves[i]):
                masses.append(self.leaf_mass_[t].get(int(leaf_id), 0.0))
            leaf_mass[i] = float(np.mean(masses))

        rarity = 1.0 - leaf_mass
        score = self.config.alpha * p_syn + (1.0 - self.config.alpha) * rarity
        return score

    def fit_threshold(self, X_val_normal: np.ndarray, quantile: Optional[float] = None) -> float:
        q = self.config.threshold_quantile if quantile is None else float(quantile)
        scores = self.score_samples(X_val_normal)
        self.threshold_ = float(np.quantile(scores, q))
        return self.threshold_

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.threshold_ is None:
            raise RuntimeError("Threshold is not fitted yet. Call fit_threshold first.")

        scores = self.score_samples(X)
        y_pred = (scores >= self.threshold_).astype(int)
        return y_pred, scores