"""
hybrid_fusion.py
----------------
Implements the Hybrid Intrusion Detection Pipeline:
1. deterministic Snort signature match -> 2. stateful LSTM anomaly forecasting
"""

import numpy as np
import pandas as pd
from ids.snort_runner import SnortSignatureDetector
from ids.lstm_model import LSTMEngine

class HybridDetectionSystem:
    def __init__(self, lstm_model: LSTMEngine):
        self.snort = SnortSignatureDetector()
        self.lstm = lstm_model

    def evaluate(self, df_raw: pd.DataFrame, X_seq: np.ndarray, y_next: np.ndarray):
        """
        Coordinates the dual-stage evaluation.
        df_raw: Original dataframe (aligned to sequence length) for Snort heuristics.
        X_seq, y_next: For LSTM forecasting.
        """
        print("[ids/fusion] Executing Snort Signature pass on raw traffic...")
        # Snort is evaluated on the raw features (simulating packet payload inspection)
        snort_preds = self.snort.predict(df_raw)
        
        final_preds = np.zeros(len(snort_preds), dtype=int)
        
        print("[ids/fusion] Routing unknowns through LSTM Anomaly Engine...")
        # LSTM processes the temporally aligned scaled sequence windows
        lstm_preds = self.lstm.predict(X_seq, y_next)
        
        for i in range(len(final_preds)):
            if snort_preds[i] == 1:
                # Deterministic signature overrides anomaly scores
                final_preds[i] = 1
            else:
                # If signature is clean, defer to Behavioral Anomaly
                final_preds[i] = lstm_preds[i]
                
        return final_preds, snort_preds, lstm_preds
