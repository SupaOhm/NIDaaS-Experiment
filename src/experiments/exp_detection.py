"""
exp_detection.py
----------------
Evaluates the detection performance of Snort, LSTM, and Hybrid models 
using the strict, leak-proof sequential academic pipeline.
"""
import sys
import numpy as np
import pandas as pd
from data.loader import load_dataset
from data.preprocess import clean_data, split_dataframe, scale_dataframes
from data.sequence_builder import build_temporal_sequences
from ids.snort_runner import SnortSignatureDetector
from ids.lstm_model import LSTMEngine
from ids.hybrid_fusion import HybridDetectionSystem
from metrics.evaluator import get_classification_report, print_metrics_table

def run_detection_experiment(data_path: str, n_records: int = None):
    print("\n" + "="*70)
    print(" EXPERIMENT 1: HYBRID IDS DETECTION EVALUATION ")
    print("="*70)
    
    # 1. Pipeline Ingestion
    df = load_dataset(data_path)
    df = clean_data(df)
    
    # Sort chronologically as the foundation of valid temporal forecasting
    time_col = "Timestamp" if "Timestamp" in df.columns else " Timestamp"
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], format="mixed", dayfirst=True)
        df.sort_values(by=time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)

    if n_records is not None:
        df = df.head(n_records)
        
    print(f"\n[exp_detection] Total Valid Flow Records: {len(df):,}")
    
    # 2. Strict Sequential Split & Scale
    df_train, df_val, df_test = split_dataframe(df, train_size=0.7, val_size=0.1)
    df_train, df_val, df_test = scale_dataframes(df_train, df_val, df_test)
    
    # 3. Time-Series Sequence Construction
    window_size = 10
    X_train_seq, y_train_next, y_train_labels = build_temporal_sequences(df_train, window_size)
    X_val_seq, y_val_next, y_val_labels       = build_temporal_sequences(df_val, window_size)
    X_test_seq, y_test_next, y_test_labels    = build_temporal_sequences(df_test, window_size)
    
    # The raw evaluation dataframe strictly paired with y_test_labels for Snort
    df_test_aligned = df_test.iloc[window_size:].reset_index(drop=True)

    results_list = []
    
    # ──────── 1. SIGNATURE ONLY (SNORT) ────────
    print("\n[exp_detection] Evaluating Baseline: Signature-Only (Snort)...")
    snort = SnortSignatureDetector()
    y_pred_snort = snort.predict(df_test_aligned)
    results_list.append(get_classification_report("Snort-Only (Signatures)", y_test_labels, y_pred_snort))
    
    # ──────── 2. ANOMALY ONLY (LSTM) ────────
    print("\n[exp_detection] Training Baseline: Anomaly-Only (LSTM)...")
    lstm = LSTMEngine(input_shape=(window_size, X_train_seq.shape[2]))
    
    # The paper mandates LSTM is trained EXCLUSIVELY on BENIGN (Normal) traffic
    benign_idx = np.where(y_train_labels == 0)[0]
    X_train_norm = X_train_seq[benign_idx]
    y_train_next_norm = y_train_next[benign_idx]
    
    val_benign_idx = np.where(y_val_labels == 0)[0]
    X_val_norm = X_val_seq[val_benign_idx]
    y_val_next_norm = y_val_next[val_benign_idx]
    
    epochs = 3 if n_records and n_records < 20000 else 20
    lstm.train(X_train_norm, y_train_next_norm, X_val_norm, y_val_next_norm, epochs=epochs)
    
    y_pred_lstm = lstm.predict(X_test_seq, y_test_next)
    results_list.append(get_classification_report("LSTM-Only (Anomaly)", y_test_labels, y_pred_lstm))
    
    # ──────── 3. HYBRID FUSION (SNORT + LSTM) ────────
    print("\n[exp_detection] Evaluating System: Hybrid Fusion (Signatures+Anomaly)...")
    fusion = HybridDetectionSystem(lstm)
    y_pred_hybrid, _, _ = fusion.evaluate(df_test_aligned, X_test_seq, y_test_next)
    results_list.append(get_classification_report("Hybrid IDS (NIDSaaS)", y_test_labels, y_pred_hybrid))
    
    # Final Output
    print_metrics_table(results_list)
    return results_list
