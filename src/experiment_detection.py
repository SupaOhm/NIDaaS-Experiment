"""
experiment_detection.py (NIDSaaS-DDA Aligned)
---------------------------------------------
Experiment Group 1: Detection Performance Comparison.
Aligned with Section III-B Methodology of the paper.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_dataset
from preprocess import clean, split_dataset, scale_data
from feature_engineering import extract_behavioral_features, make_forecasting_data
from signature_detector import SignatureDetector
from lstm_model import LSTMModel
from hybrid_model import HybridDetector
from classical_models import (
    train_rf, eval_rf, train_xgb, eval_xgb, train_logreg, eval_logreg
)
from metrics import classification_report, save_table, save_confusion_matrix, plot_f1_comparison

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")

def _make_synthetic_df(n=1000):
    """Generates a small synthetic CIC-IDS2017-like DataFrame for smoke tests."""
    data = {
        "Total Fwd Packets": np.random.randint(1, 100, n),
        "Total Backward Packets": np.random.randint(1, 100, n),
        "Total Length of Fwd Packets": np.random.randint(100, 10000, n),
        "Total Length of Bwd Packets": np.random.randint(100, 10000, n),
        "Flow Duration": np.random.randint(1000, 1000000, n),
        "Flow Packets/s": np.random.random(n) * 100,
        "Flow Bytes/s": np.random.random(n) * 1000,
        "RST Flag Count": np.random.randint(0, 2, n),
        "Packet Length Variance": np.random.random(n) * 500,
        "Fwd Packet Length Mean": np.random.random(n) * 100,
        "Bwd Packet Length Mean": np.random.random(n) * 100,
        "Label": [0] * (n // 2) + [1] * (n - n // 2)
    }
    return pd.DataFrame(data)

def run_detection_experiment(data_path: str, smoke: bool = False):
    print("\n" + "=" * 65)
    print("EXPERIMENT 1 — Performance (NIDSaaS-DDA Methodology)")
    print("=" * 65)

    # 1. Load Data
    if smoke:
        df = _make_synthetic_df(n=1000)
    else:
        df = load_dataset(data_path)

    # 2. Behavioral Feature Engineering (Section III-B-3-2)
    X, y = extract_behavioral_features(df)
    
    # 3. Clean & Split
    train_idx, val_idx, test_idx = split_dataset(y, train_size=0.7, val_size=0.1)
    
    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_val_raw, y_val     = X[val_idx], y[val_idx]
    X_test_raw, y_test   = X[test_idx], y[test_idx]

    # 4. Scaling
    X_train, X_val, X_test = scale_data(X_train_raw, X_val_raw, X_test_raw)

    results_list = []

    # ── 1. Signature-Only (Snort) ─────────────────────────
    print("[experiment_detection] Running Signature-Based (Snort) Baseline...")
    sig = SignatureDetector()
    y_pred_sig_str, _ = sig.predict(df.iloc[test_idx])
    y_pred_sig = (y_pred_sig_str == "ATTACK").astype(int)
    results_list.append(classification_report("Signature-Only", y_test, y_pred_sig))

    # ── 2. Classical Models ───────────────────────────────
    print("[experiment_detection] Training Classical Baselines...")
    results_list.append(eval_rf(train_rf(X_train, y_train), X_test, y_test))
    # results_list.append(eval_xgb(train_xgb(X_train, y_train), X_test, y_test))
    results_list.append(eval_logreg(train_logreg(X_train, y_train), X_test, y_test))

    # ── 3. Hybrid (NIDSaaS-DDA Proposed) ─────────────────
    print("[experiment_detection] Training Hybrid (Forecasting LSTM)...")
    
    # Paper Section III-B-3-3: Train LSTM ONLY on normal records
    X_train_norm = X_train[y_train == 0]
    
    WINDOW_SIZE = 10
    X_train_seq, y_train_next = make_forecasting_data(X_train_norm, WINDOW_SIZE)
    X_val_seq, y_val_next     = make_forecasting_data(X_val[y_val==0], WINDOW_SIZE)
    
    # Forecasting model input size is the number of features
    input_shape = (WINDOW_SIZE, X_train.shape[1])
    lstm = LSTMModel(input_shape=input_shape)
    lstm.train(X_train_seq, y_train_next, X_val_seq, y_val_next, epochs=5 if smoke else 20)

    # Evaluation: Entire Test Set
    # Note: LSTM can only predict from record [WINDOW_SIZE:]
    X_test_seq, y_test_next = make_forecasting_data(X_test, WINDOW_SIZE)
    y_test_aligned = y_test[WINDOW_SIZE:]
    
    # 1. Signature Layer
    y_pred_sig_test_str, _ = sig.predict(df.iloc[test_idx].iloc[WINDOW_SIZE:])
    y_pred_hybrid = np.zeros(len(y_test_aligned))
    
    # 2. Routing logic (Section III-B: Hybrid Detection Engine)
    for i in range(len(y_pred_hybrid)):
        if y_pred_sig_test_str[i] == "ATTACK":
            y_pred_hybrid[i] = 1
        else:
            # Anomaly Layer: Euclidean distance prediction error
            dist = lstm.predict_anomaly_score(
                X_test_seq[i:i+1], y_test_next[i:i+1]
            )
            # Thresholding (Standard research tau)
            if dist[0] > 0.3: # Threshold τ
                y_pred_hybrid[i] = 1

    results_list.append(classification_report("Hybrid (NIDSaaS-DDA)", y_test_aligned, y_pred_hybrid))

    # ── Final Processing ──────────────────────────────────
    save_table(results_list, os.path.join(TABLES_DIR, "detection_results.csv"))
    
    for res in results_list:
        if "confusion_matrix" in res:
             save_confusion_matrix(res["confusion_matrix"], 
                                   os.path.join(FIGURES_DIR, f"cm_{res['model'].replace('/','_')}.png"),
                                   res["model"])
             
    plot_f1_comparison(results_list, os.path.join(FIGURES_DIR, "f1_comparison.png"))
    print("\n[experiment_detection] ✓ Experiment 1 complete.")
    return results_list
