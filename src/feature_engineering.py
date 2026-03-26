"""
feature_engineering.py
-----------------------
Applies paper-inspired feature engineering before feeding data to models.

The paper's LSTM is NOT just raw CIC-IDS2017 columns — it uses engineered
features that capture behavioural patterns (ratios, rates, aggregations).

This module:
  1. Derives ratio and rate features from raw flow columns.
  2. Bins duration into categorical buckets encoded as integers.
  3. (Optionally) reshapes feature arrays into time-window sequences for the LSTM.

CIC-IDS2017 already contains many flow-level features, so we add a focused set
of derived features on top to represent the "feature engineering" contribution
described in the paper.
"""

import numpy as np
import pandas as pd
from typing import Optional


# CIC-IDS2017 candidate column names (stripped of whitespace)
_COL_FWD_PKTS    = "Total Fwd Packets"
_COL_BWD_PKTS    = "Total Backward Packets"
_COL_FWD_BYTES   = "Total Length of Fwd Packets"
_COL_BWD_BYTES   = "Total Length of Bwd Packets"
_COL_DURATION    = "Flow Duration"
_COL_FLOW_PKTS   = "Flow Packets/s"
_COL_FLOW_BYTES  = "Flow Bytes/s"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the DataFrame.

    New columns added (when source columns are present):
      - bytes_ratio      : fwd_bytes / (fwd_bytes + bwd_bytes + 1)
      - packet_ratio     : fwd_pkts  / (fwd_pkts  + bwd_pkts  + 1)
      - bytes_per_packet : total bytes / total packets
      - duration_bucket  : fast / medium / long (0 / 1 / 2)
      - log_flow_pkts    : log1p of flow packets/s
      - log_flow_bytes   : log1p of flow bytes/s

    Parameters
    ----------
    df : pd.DataFrame — cleaned DataFrame (output of preprocess.clean)

    Returns
    -------
    pd.DataFrame with additional engineered columns
    """
    df = df.copy()

    fwd_pkts   = df.get(_COL_FWD_PKTS,   pd.Series(np.zeros(len(df))))
    bwd_pkts   = df.get(_COL_BWD_PKTS,   pd.Series(np.zeros(len(df))))
    fwd_bytes  = df.get(_COL_FWD_BYTES,  pd.Series(np.zeros(len(df))))
    bwd_bytes  = df.get(_COL_BWD_BYTES,  pd.Series(np.zeros(len(df))))
    duration   = df.get(_COL_DURATION,   pd.Series(np.ones(len(df))))
    flow_pkts  = df.get(_COL_FLOW_PKTS,  pd.Series(np.zeros(len(df))))
    flow_bytes = df.get(_COL_FLOW_BYTES, pd.Series(np.zeros(len(df))))

    # --- Ratios ---
    total_bytes = fwd_bytes + bwd_bytes + 1
    total_pkts  = fwd_pkts  + bwd_pkts  + 1

    df["bytes_ratio"]      = fwd_bytes / total_bytes
    df["packet_ratio"]     = fwd_pkts  / total_pkts
    df["bytes_per_packet"] = total_bytes / total_pkts

    # --- Duration bucket: fast (<1s=0), medium (1-10s=1), long (>10s=2) ---
    # CIC-IDS2017 duration is in microseconds
    dur_sec = duration / 1e6
    df["duration_bucket"] = pd.cut(
        dur_sec,
        bins=[-np.inf, 1.0, 10.0, np.inf],
        labels=[0, 1, 2],
    ).astype(float).fillna(0)

    # --- Log-scale rate features (stabilise large value ranges) ---
    df["log_flow_pkts"]  = np.log1p(flow_pkts.clip(lower=0))
    df["log_flow_bytes"] = np.log1p(flow_bytes.clip(lower=0))

    return df


def make_sequences(
    X: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """
    Reshape a 2-D feature array into 3-D time-window sequences for the LSTM.

    Parameters
    ----------
    X           : (N, F) array of scaled features
    window_size : number of consecutive time steps per sequence

    Returns
    -------
    (N - window_size + 1, window_size, F) array
    """
    n_samples, n_features = X.shape
    sequences = np.stack(
        [X[i: i + window_size] for i in range(n_samples - window_size + 1)],
        axis=0,
    )
    return sequences


def make_forecasting_data(X_data, window_size=10):
    """
    Transforms [N, features] into (X, y_next) forecasting pairs.
    X: [N - window_size, window_size, features] - Behavioral sequence (St)
    y_next: [N - window_size, features] - Actual future vector (vt+1)
    """
    X, y_next = [], []
    for i in range(len(X_data) - window_size):
        X.append(X_data[i : i + window_size])
        y_next.append(X_data[i + window_size])
    return np.array(X), np.array(y_next)

def extract_behavioral_features(df):
    """
    Maps CIC-IDS2017 fields to NIDSaaS Section III-B-3-2 features:
      - Volumetric (counts, bytes)
      - Fdeny (Deny frequency proxy: RST flag count)
      - Diversity (Unique Dst IP proxy: Packet Length Variance)
    """
    # Paper Section III-B-3-2 Features
    features = [
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
        "RST Flag Count",   # Proxy for Fdeny (Deny actions)
        "Packet Length Variance",  # Proxy for Diversity (Unique hosts/ports)
        "Fwd Packet Length Mean", "Bwd Packet Length Mean"
    ]
    # Ensure columns exist
    cols = [c for c in features if c in df.columns]
    return df[cols].values, df["Label"].values


def align_labels_to_sequences(y: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Align labels to sequence windows by taking the label of the last step.

    Parameters
    ----------
    y           : (N,) label array
    window_size : must match the value used in make_sequences

    Returns
    -------
    (N - window_size + 1,) label array
    """
    return y[window_size - 1:]
