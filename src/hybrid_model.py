"""
hybrid_model.py
---------------
Hybrid detector combining Signature-first matching with LSTM anomaly detection.

Paper concept:
  1. Try signature rules first — fast, zero-overhead for known attacks.
  2. If the signature module is inconclusive ("UNKNOWN"), route to LSTM.
  3. LSTM decides based on temporal flow patterns.

This preserves the key design principle: signatures handle the easy/known cases;
the LSTM handles the harder, behavioural, or novel cases.

Usage
-----
    hybrid = HybridDetector(lstm_model, input_shape=(window_size, n_features))
    hybrid.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    preds = hybrid.predict(df_test_raw, X_test_seq)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from signature_detector import SignatureDetector
from lstm_model import LSTMModel


class HybridDetector:
    """
    Signature-first hybrid IDS.

    Parameters
    ----------
    input_shape  : (window_size, n_features) for the LSTM
    lstm_units   : LSTM hidden units
    dropout      : LSTM dropout rate
    lstm_threshold : binary threshold on LSTM sigmoid output
    """

    def __init__(
        self,
        input_shape: tuple,
        lstm_units: int = 64,
        dropout: float = 0.3,
        lstm_threshold: float = 0.5,
    ):
        self.sig_detector = SignatureDetector()
        self.lstm = LSTMModel(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dropout=dropout,
        )
        self.lstm_threshold = lstm_threshold

    def fit(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_seq: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
    ):
        """Train only the LSTM component (signature rules need no training)."""
        print("[hybrid_model] Training LSTM component...")
        self.lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                        epochs=epochs, batch_size=batch_size)
        return self

    def predict(
        self,
        df_raw: pd.DataFrame,
        X_seq: np.ndarray,
    ) -> np.ndarray:
        """
        Run hybrid prediction.

        For each sample:
          - If the signature detector fires → predict ATTACK (1)
          - If signature is UNKNOWN       → use LSTM output

        Parameters
        ----------
        df_raw : raw DataFrame aligned to X_seq (same row order, post-windowing).
                 Used for signature rule evaluation.
                 Must be trimmed to (N - window_size + 1) rows to match X_seq.
        X_seq  : (N_seq, window_size, features) array ready for LSTM

        Returns
        -------
        (N_seq,) binary prediction array
        """
        assert len(df_raw) == len(X_seq), (
            f"df_raw ({len(df_raw)}) and X_seq ({len(X_seq)}) must have the same length."
        )

        # Step 1: signature pass
        sig_labels, sig_conf = self.sig_detector.predict(df_raw)

        # Step 2: LSTM pass for inconclusive rows
        lstm_preds = self.lstm.predict(X_seq, threshold=self.lstm_threshold)

        # Step 3: merge — signature wins when confident
        final = np.where(sig_labels == "ATTACK", 1, lstm_preds)
        return final.astype(int)

    def save_lstm(self, path: str):
        """Persist the LSTM weights to disk."""
        self.lstm.save(path)

    def load_lstm(self, path: str):
        """Load previously saved LSTM weights from disk."""
        self.lstm.load(path)
        return self
