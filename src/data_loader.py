"""
data_loader.py
--------------
Loads CIC-IDS2017 CSV files from a directory or a single file path.
Returns a clean DataFrame with a unified 'label' column.

The CIC-IDS2017 dataset has a column named ' Label' (with a leading space).
This module strips whitespace from column names, maps the label to binary
(BENIGN=0, ATTACK=1), and optionally keeps the raw multiclass label too.
"""

import os
import glob
import pandas as pd


def load_dataset(path: str, multiclass: bool = False) -> pd.DataFrame:
    """
    Load one or more CIC-IDS2017 CSV files.

    Parameters
    ----------
    path : str
        Path to a single CSV file OR a directory containing CSV files.
    multiclass : bool
        If True, keep original multiclass label in column 'attack_type'.
        If False (default), collapse to binary BENIGN=0 / ATTACK=1.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns stripped of whitespace and a 'label' column.
    """
    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        csv_files += glob.glob(os.path.join(path, "*.csv"))
        csv_files = list(set(csv_files))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        print(f"[data_loader] Found {len(csv_files)} CSV file(s). Loading...")
        dfs = [_load_single_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    elif os.path.isfile(path):
        df = _load_single_csv(path)
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    df = _unify_labels(df, multiclass=multiclass)
    print(f"[data_loader] Loaded {len(df):,} records. "
          f"Label distribution:\n{df['label'].value_counts()}")
    return df


def _load_single_csv(filepath: str) -> pd.DataFrame:
    """Read one CSV file, stripping whitespace from column names and string values."""
    # CIC-IDS2017 files often contain non-UTF8 characters. 
    # 'latin1' (ISO-8859-1) is the standard fix for loading these.
    print(f"  [data_loader] Loading: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, low_memory=False, encoding="latin1")
    # Strip leading/trailing whitespace from column names (CIC-IDS2017 quirk)
    df.columns = [c.strip() for c in df.columns]
    return df


def _unify_labels(df: pd.DataFrame, multiclass: bool) -> pd.DataFrame:
    """
    Normalise the label column.

    CIC-IDS2017 uses 'Label' column with values like 'BENIGN', 'DDoS', etc.
    """
    if "Label" not in df.columns:
        raise KeyError("Expected column 'Label' not found after stripping whitespace. "
                       "Check that your CSV files are CIC-IDS2017 format.")

    df["Label"] = df["Label"].str.strip()

    if multiclass:
        df["attack_type"] = df["Label"]

    # Binary label: 0 = benign, 1 = attack
    df["label"] = (df["Label"] != "BENIGN").astype(int)
    df = df.drop(columns=["Label"])
    return df
