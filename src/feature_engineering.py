"""
feature_engineering.py
-----------------------
Feature engineering and sequence builders for NIDaaS experiments.
"""

import numpy as np
import pandas as pd


# CIC-IDS2017 candidate column names
_COL_FWD_PKTS = "Total Fwd Packets"
_COL_BWD_PKTS = "Total Backward Packets"
_COL_FWD_BYTES = "Total Length of Fwd Packets"
_COL_BWD_BYTES = "Total Length of Bwd Packets"
_COL_DURATION = "Flow Duration"
_COL_FLOW_PKTS = "Flow Packets/s"
_COL_FLOW_BYTES = "Flow Bytes/s"
_COL_FWD_PKT_MEAN = "Fwd Packet Length Mean"
_COL_BWD_PKT_MEAN = "Bwd Packet Length Mean"
_COL_PKT_VAR = "Packet Length Variance"
_COL_RST = "RST Flag Count"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the DataFrame.
    """
    df = df.copy()

    if "Label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"Label": "label"})

    fwd_pkts = df.get(_COL_FWD_PKTS, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    bwd_pkts = df.get(_COL_BWD_PKTS, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    fwd_bytes = df.get(_COL_FWD_BYTES, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    bwd_bytes = df.get(_COL_BWD_BYTES, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    duration = df.get(_COL_DURATION, pd.Series(np.ones(len(df)), index=df.index)).astype(float)
    flow_pkts = df.get(_COL_FLOW_PKTS, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    flow_bytes = df.get(_COL_FLOW_BYTES, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    fwd_pkt_mean = df.get(_COL_FWD_PKT_MEAN, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    bwd_pkt_mean = df.get(_COL_BWD_PKT_MEAN, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    pkt_var = df.get(_COL_PKT_VAR, pd.Series(np.zeros(len(df)), index=df.index)).astype(float)

    total_bytes = fwd_bytes + bwd_bytes
    total_pkts = fwd_pkts + bwd_pkts

    safe_total_bytes = total_bytes + 1.0
    safe_total_pkts = total_pkts + 1.0
    safe_duration_sec = (duration / 1e6).clip(lower=1e-6)

    # Basic ratios
    df["bytes_ratio"] = fwd_bytes / safe_total_bytes
    df["packet_ratio"] = fwd_pkts / safe_total_pkts
    df["bytes_per_packet"] = safe_total_bytes / safe_total_pkts

    # Volume features
    df["total_bytes"] = total_bytes
    df["total_pkts"] = total_pkts
    df["fwd_bwd_byte_diff"] = fwd_bytes - bwd_bytes
    df["fwd_bwd_pkt_diff"] = fwd_pkts - bwd_pkts

    # Rate-like features
    df["pkt_rate_calc"] = total_pkts / safe_duration_sec
    df["byte_rate_calc"] = total_bytes / safe_duration_sec

    # Packet statistics
    df["mean_pkt_len_bidir"] = (fwd_pkt_mean + bwd_pkt_mean) / 2.0
    df["pkt_len_var_log"] = np.log1p(pkt_var.clip(lower=0))

    # Logs / buckets
    df["duration_sec"] = safe_duration_sec
    df["log_duration"] = np.log1p(safe_duration_sec)
    df["log_flow_pkts"] = np.log1p(flow_pkts.clip(lower=0))
    df["log_flow_bytes"] = np.log1p(flow_bytes.clip(lower=0))

    df["duration_bucket"] = pd.cut(
        safe_duration_sec,
        bins=[-np.inf, 0.1, 1.0, 10.0, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(float).fillna(0.0)

    # Interaction-like features
    df["burst_indicator"] = (df["log_flow_pkts"] * df["log_flow_bytes"]).astype(float)
    df["directional_intensity"] = (
        np.abs(df["fwd_bwd_byte_diff"]) / (safe_total_bytes)
    ).astype(float)

    return df


def extract_behavioral_features(df: pd.DataFrame):
    """
    Build final tabular feature matrix for classical models and sequence models.
    """
    df = df.copy()

    if "Label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"Label": "label"})

    if "label" not in df.columns:
        raise KeyError(f"'label' column not found. Available columns: {list(df.columns)}")

    raw_features = [
        _COL_FWD_PKTS,
        _COL_BWD_PKTS,
        _COL_FWD_BYTES,
        _COL_BWD_BYTES,
        _COL_DURATION,
        _COL_FLOW_PKTS,
        _COL_FLOW_BYTES,
        _COL_RST,
        _COL_PKT_VAR,
        _COL_FWD_PKT_MEAN,
        _COL_BWD_PKT_MEAN,
    ]

    engineered_features = [
        "bytes_ratio",
        "packet_ratio",
        "bytes_per_packet",
        "total_bytes",
        "total_pkts",
        "fwd_bwd_byte_diff",
        "fwd_bwd_pkt_diff",
        "pkt_rate_calc",
        "byte_rate_calc",
        "mean_pkt_len_bidir",
        "pkt_len_var_log",
        "duration_sec",
        "log_duration",
        "duration_bucket",
        "log_flow_pkts",
        "log_flow_bytes",
        "burst_indicator",
        "directional_intensity",
    ]

    selected = [c for c in raw_features + engineered_features if c in df.columns]
    if not selected:
        raise ValueError("No usable behavioral feature columns found.")

    X = df[selected].astype(float).values
    y = df["label"].astype(int).values
    return X, y


def make_forecasting_data(X_data: np.ndarray, window_size: int = 10):
    """
    Kept for backward compatibility.
    """
    X, y_next = [], []
    for i in range(len(X_data) - window_size):
        X.append(X_data[i:i + window_size])
        y_next.append(X_data[i + window_size])

    if len(X) == 0:
        n_features = X_data.shape[1] if X_data.ndim == 2 and X_data.size > 0 else 0
        return (
            np.empty((0, window_size, n_features), dtype=float),
            np.empty((0, n_features), dtype=float),
        )

    return np.array(X), np.array(y_next)


def make_sequence_classification_data(X_data: np.ndarray, y_data: np.ndarray, window_size: int = 10):
    """
    Build supervised sequences.

    Sequence ending at index t predicts the label at index t.
    X_seq[k] shape = [window_size, n_features]
    y_seq[k] = y[t]
    """
    if len(X_data) != len(y_data):
        raise ValueError("X_data and y_data must have the same length.")

    X_seq, y_seq = [], []

    for end_idx in range(window_size - 1, len(X_data)):
        start_idx = end_idx - window_size + 1
        X_seq.append(X_data[start_idx:end_idx + 1])
        y_seq.append(y_data[end_idx])

    if len(X_seq) == 0:
        n_features = X_data.shape[1] if X_data.ndim == 2 and X_data.size > 0 else 0
        return (
            np.empty((0, window_size, n_features), dtype=float),
            np.empty((0,), dtype=int),
        )

    return np.array(X_seq), np.array(y_seq)