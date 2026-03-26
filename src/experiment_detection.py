"""
experiment_detection.py
-----------------------
Experiment Group 1: Detection Performance Comparison

Leakage-safe sequence evaluation
--------------------------------
This version avoids sequence leakage by:
1) Processing each CSV file independently
2) Splitting each file into train / val / test FIRST
3) Creating LSTM sequences separately inside each split
4) Evaluating all models on the same endpoint-aligned test labels

Why this matters
----------------
If you create overlapping windows first and split later, train/test windows
can share many timesteps. That can inflate LSTM performance unrealistically.
This version prevents that issue.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_dataset
from preprocess import clean, split_dataset, scale_data
from feature_engineering import (
    engineer_features,
    extract_behavioral_features,
    make_sequence_classification_data,
)
from signature_detector import SignatureDetector
from lstm_model import LSTMModel
from classical_models import (
    train_rf,
    eval_rf,
    train_logreg,
    eval_logreg,
)
from metrics import (
    classification_report,
    save_table,
    save_confusion_matrix,
    plot_f1_comparison,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def _make_synthetic_df(n=1000, seed=42):
    """
    Small synthetic dataframe for smoke testing.

    The goal is only to verify pipeline execution.
    It is NOT intended to represent meaningful IDS performance.
    """
    rng = np.random.default_rng(seed)

    # Create labels first
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(labels)

    # Make class 1 slightly different so smoke mode is not completely random
    flow_pkts = rng.random(n) * 100
    flow_bytes = rng.random(n) * 1000
    rst_flags = rng.integers(0, 2, n)

    attack_idx = labels == 1
    flow_pkts[attack_idx] *= 2.0
    flow_bytes[attack_idx] *= 2.5
    rst_flags[attack_idx] = rng.integers(0, 3, attack_idx.sum())

    data = {
        "Total Fwd Packets": rng.integers(1, 100, n),
        "Total Backward Packets": rng.integers(1, 100, n),
        "Total Length of Fwd Packets": rng.integers(100, 10000, n),
        "Total Length of Bwd Packets": rng.integers(100, 10000, n),
        "Flow Duration": rng.integers(1000, 1000000, n),
        "Flow Packets/s": flow_pkts,
        "Flow Bytes/s": flow_bytes,
        "RST Flag Count": rst_flags,
        "Packet Length Variance": rng.random(n) * 500,
        "Fwd Packet Length Mean": rng.random(n) * 100,
        "Bwd Packet Length Mean": rng.random(n) * 100,
        "label": labels,
    }
    return pd.DataFrame(data)


def _ensure_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the target column name to 'label'.
    """
    df = df.copy()

    if "Label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"Label": "label"})

    if "label" not in df.columns:
        raise KeyError(f"'label' column not found. Columns: {list(df.columns)}")

    return df


def _discover_csv_files(path: str):
    """
    Return a sorted list of CSV files from:
    - a single CSV path
    - a directory
    - nested directories
    """
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        csv_files = sorted(set(csv_files))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return csv_files

    raise FileNotFoundError(f"Path does not exist: {path}")


def _prepare_real_files(data_path: str):
    """
    Load, clean, and feature-engineer each CSV independently.

    This preserves file boundaries for leakage-safe sequence creation later.
    """
    csv_files = _discover_csv_files(data_path)
    print(f"[experiment_detection] Found {len(csv_files)} CSV file(s).")

    prepared = []
    for fp in csv_files:
        print(f"[experiment_detection] Preparing file: {os.path.basename(fp)}")
        df_i = load_dataset(fp)
        df_i = clean(df_i)
        df_i = _ensure_label(df_i)
        df_i = engineer_features(df_i)
        prepared.append(df_i)

    return prepared


def _prepare_smoke_files():
    """
    Build multiple synthetic files so smoke mode follows the same logic
    as real-data mode.
    """
    dfs = []
    for n, seed in [(800, 1), (900, 2), (1000, 3)]:
        df = _make_synthetic_df(n=n, seed=seed)
        df = _ensure_label(df)
        df = engineer_features(df)
        dfs.append(df)
    return dfs


def _split_one_file_then_make_sequences(df_i: pd.DataFrame, window_size: int, random_state: int = 42):
    """
    For ONE file:
    1) extract tabular features
    2) split rows into train / val / test FIRST
    3) create sequences separately inside each split

    This prevents sequence overlap across splits.
    """
    X_i, y_i = extract_behavioral_features(df_i)

    # Need enough rows to support splitting plus at least one window
    min_rows = max(window_size * 3, 30)
    if len(X_i) < min_rows:
        return None

    train_idx, val_idx, test_idx = split_dataset(
        y_i,
        train_size=0.7,
        val_size=0.1,
        random_state=random_state,
    )

    # Sort indices so sequence order inside each split follows original file order
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    X_train_raw, y_train = X_i[train_idx], y_i[train_idx]
    X_val_raw, y_val = X_i[val_idx], y_i[val_idx]
    X_test_raw, y_test = X_i[test_idx], y_i[test_idx]

    # Scale row-level features for classical models
    X_train_row_s, X_val_row_s, X_test_row_s = scale_data(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )

    # Create supervised sequences separately inside each split
    X_train_seq, y_train_seq = make_sequence_classification_data(X_train_row_s, y_train, window_size)
    X_val_seq, y_val_seq = make_sequence_classification_data(X_val_row_s, y_val, window_size)
    X_test_seq, y_test_seq = make_sequence_classification_data(X_test_row_s, y_test, window_size)

    # Skip files that become too short after splitting
    if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
        return None

    # Endpoint-aligned row features and rows for fair evaluation
    X_train_end = X_train_row_s[window_size - 1:]
    X_val_end = X_val_row_s[window_size - 1:]
    X_test_end = X_test_row_s[window_size - 1:]

    df_train_end = df_i.iloc[train_idx].reset_index(drop=True).iloc[window_size - 1:].reset_index(drop=True)
    df_val_end = df_i.iloc[val_idx].reset_index(drop=True).iloc[window_size - 1:].reset_index(drop=True)
    df_test_end = df_i.iloc[test_idx].reset_index(drop=True).iloc[window_size - 1:].reset_index(drop=True)

    return {
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
        "X_train_end": X_train_end,
        "X_val_end": X_val_end,
        "X_test_end": X_test_end,
        "df_train_end": df_train_end,
        "df_val_end": df_val_end,
        "df_test_end": df_test_end,
    }


def _build_leakage_safe_dataset(file_dfs, window_size: int):
    """
    Combine leakage-safe per-file splits into one final dataset.

    Output datasets are aligned so all models evaluate on the same y_test.
    """
    train_seq_list, val_seq_list, test_seq_list = [], [], []
    train_y_list, val_y_list, test_y_list = [], [], []

    train_end_list, val_end_list, test_end_list = [], [], []
    train_df_end_list, val_df_end_list, test_df_end_list = [], [], []

    skipped_files = 0

    for i, df_i in enumerate(file_dfs, start=1):
        result = _split_one_file_then_make_sequences(df_i, window_size=window_size, random_state=42)

        if result is None:
            skipped_files += 1
            print(f"[experiment_detection] Skip file #{i}: not enough usable rows after split/windowing.")
            continue

        train_seq_list.append(result["X_train_seq"])
        train_y_list.append(result["y_train_seq"])
        val_seq_list.append(result["X_val_seq"])
        val_y_list.append(result["y_val_seq"])
        test_seq_list.append(result["X_test_seq"])
        test_y_list.append(result["y_test_seq"])

        train_end_list.append(result["X_train_end"])
        val_end_list.append(result["X_val_end"])
        test_end_list.append(result["X_test_end"])

        train_df_end_list.append(result["df_train_end"])
        val_df_end_list.append(result["df_val_end"])
        test_df_end_list.append(result["df_test_end"])

    if not train_seq_list:
        raise ValueError("No valid per-file splits could produce sequences.")

    X_train_seq = np.concatenate(train_seq_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    X_val_seq = np.concatenate(val_seq_list, axis=0)
    y_val = np.concatenate(val_y_list, axis=0)

    X_test_seq = np.concatenate(test_seq_list, axis=0)
    y_test = np.concatenate(test_y_list, axis=0)

    X_train_end = np.concatenate(train_end_list, axis=0)
    X_val_end = np.concatenate(val_end_list, axis=0)
    X_test_end = np.concatenate(test_end_list, axis=0)

    df_train_end = pd.concat(train_df_end_list, ignore_index=True)
    df_val_end = pd.concat(val_df_end_list, ignore_index=True)
    df_test_end = pd.concat(test_df_end_list, ignore_index=True)

    print(f"[experiment_detection] Built leakage-safe dataset from {len(file_dfs) - skipped_files} file(s).")
    print(f"[experiment_detection] Skipped {skipped_files} file(s).")
    print(f"[experiment_detection] Train seq: {len(X_train_seq):,}")
    print(f"[experiment_detection] Val seq:   {len(X_val_seq):,}")
    print(f"[experiment_detection] Test seq:  {len(X_test_seq):,}")

    return {
        "X_train_seq": X_train_seq,
        "y_train": y_train,
        "X_val_seq": X_val_seq,
        "y_val": y_val,
        "X_test_seq": X_test_seq,
        "y_test": y_test,
        "X_train_end": X_train_end,
        "X_val_end": X_val_end,
        "X_test_end": X_test_end,
        "df_train_end": df_train_end,
        "df_val_end": df_val_end,
        "df_test_end": df_test_end,
    }


def run_detection_experiment(data_path: str, smoke: bool = False, profile: str = "fast"):
    print("\n" + "=" * 65)
    print("EXPERIMENT 1 — Performance (Detection Performance) Comparison")
    print("=" * 65)
    print(f"[experiment_detection] Profile: {profile}")

    # 1) Prepare input files
    if smoke:
        file_dfs = _prepare_smoke_files()
    else:
        file_dfs = _prepare_real_files(data_path)

    # 2) Profile config
    if smoke:
        cfg = {
            "window_size": 10,
            "lstm_units": 32,
            "epochs": 3,
            "batch_size": 512,
        }
    elif profile == "fast":
        cfg = {
            "window_size": 10,
            "lstm_units": 64,
            "epochs": 6,
            "batch_size": 1024,
        }
    else:  # full
        cfg = {
            "window_size": 20,
            "lstm_units": 128,
            "epochs": 12,
            "batch_size": 512,
        }

    window_size = cfg["window_size"]

    # 3) Build leakage-safe train/val/test datasets
    ds = _build_leakage_safe_dataset(file_dfs, window_size=window_size)

    X_train_seq = ds["X_train_seq"]
    y_train = ds["y_train"]
    X_val_seq = ds["X_val_seq"]
    y_val = ds["y_val"]
    X_test_seq = ds["X_test_seq"]
    y_test = ds["y_test"]

    X_train_end = ds["X_train_end"]
    X_val_end = ds["X_val_end"]
    X_test_end = ds["X_test_end"]

    df_test_end = ds["df_test_end"]

    print("[split] train label counts:", np.bincount(y_train))
    print("[split] val   label counts:", np.bincount(y_val))
    print("[split] test  label counts:", np.bincount(y_test))

    results_list = []

    # 4) Signature baseline on the same endpoint-aligned test rows
    print("[experiment_detection] Running Signature-Based (Snort) Baseline...")
    sig = SignatureDetector()
    y_pred_sig_str, _ = sig.predict(df_test_end)
    y_pred_sig = (y_pred_sig_str == "ATTACK").astype(int)
    results_list.append(classification_report("Signature-Only", y_test, y_pred_sig))

    # 5) Classical baselines on the same endpoint-aligned test labels
    print("[experiment_detection] Training Classical Baselines...")
    results_list.append(
        eval_rf(train_rf(X_train_end, y_train), X_test_end, y_test)
    )
    results_list.append(
        eval_logreg(train_logreg(X_train_end, y_train), X_test_end, y_test)
    )

    # 6) Supervised LSTM sequence classifier
    print("[experiment_detection] Training supervised LSTM sequence models...")

    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    pos_weight = neg / pos

    print(f"[experiment_detection] LSTM pos_weight: {pos_weight:.4f}")

    input_shape = (window_size, X_train_seq.shape[2])
    lstm = LSTMModel(
        input_shape=input_shape,
        lstm_units=cfg["lstm_units"],
        pos_weight=pos_weight,
    )

    lstm.train(
        X_train_seq,
        y_train,
        X_val_seq,
        y_val,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
    )

    # 7) Tune probability threshold on validation F1
    val_probs = lstm.predict_proba(X_val_seq)
    candidate_thresholds = np.linspace(0.10, 0.90, 17)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in candidate_thresholds:
        y_pred_val = (val_probs >= thr).astype(int)
        rep = classification_report("tmp", y_val, y_pred_val)
        if rep["f1_score"] > best_f1:
            best_f1 = rep["f1_score"]
            best_thr = float(thr)

    print(f"[experiment_detection] Best LSTM threshold: {best_thr:.2f} (val_f1={best_f1:.4f})")

    # 8) LSTM-only
    test_probs = lstm.predict_proba(X_test_seq)
    y_pred_lstm = (test_probs >= best_thr).astype(int)
    results_list.append(classification_report("LSTM-Only", y_test, y_pred_lstm))

    # 9) Hybrid: signature first, otherwise LSTM
    sig_attack = (y_pred_sig_str == "ATTACK").astype(int)
    y_pred_hybrid = np.where(sig_attack == 1, 1, y_pred_lstm)
    results_list.append(classification_report("Hybrid (Signature+LSTM)", y_test, y_pred_hybrid))

    # 10) Save results
    save_table(results_list, os.path.join(TABLES_DIR, "detection_results.csv"))

    for res in results_list:
        if "confusion_matrix" in res:
            save_confusion_matrix(
                res["confusion_matrix"],
                os.path.join(FIGURES_DIR, f"cm_{res['model'].replace('/', '_')}.png"),
                res["model"],
            )

    plot_f1_comparison(
        results_list,
        os.path.join(FIGURES_DIR, "f1_comparison.png"),
    )

    print("\n[experiment_detection] ✓ Experiment 1 complete.")
    return results_list