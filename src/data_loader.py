"""
data_loader.py
--------------
Loads CIC-IDS2017 CSV files from a directory or a single file path.

Main responsibilities:
- Read one or more CSV files from the given path
- Strip whitespace from column names
- Normalize the original 'Label' column into a unified binary 'label' column
- Optionally keep the original attack type for multiclass experiments
- Add '__source_file__' to track which CSV each record came from

Label convention:
- BENIGN -> 0
- Any non-BENIGN label -> 1

Typical usage:
    from data_loader import load_dataset
    df = load_dataset("data/", multiclass=False)

Expected output:
- A merged pandas DataFrame
- Clean column names
- A unified 'label' column ready for downstream experiments
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
        If False, collapse to binary BENIGN=0 / ATTACK=1.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with stripped column names and a unified 'label' column.
    """
    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        csv_files = sorted(set(csv_files))
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
    print(
        f"[data_loader] Loaded {len(df):,} records.\n"
        f"[data_loader] Label distribution:\n{df['label'].value_counts()}"
    )
    return df


def _load_single_csv(filepath: str) -> pd.DataFrame:
    """Read one CSV file and strip whitespace from column names."""
    print(f"  [data_loader] Loading: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, low_memory=False, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    df["__source_file__"] = os.path.basename(filepath)
    return df


def _unify_labels(df: pd.DataFrame, multiclass: bool) -> pd.DataFrame:
    """Normalize CIC-IDS2017 labels into binary or keep multiclass labels."""
    if "Label" not in df.columns:
        raise KeyError(
            "Expected column 'Label' not found after stripping whitespace. "
            "Check that your CSV files are CIC-IDS2017 format."
        )

    df["Label"] = df["Label"].astype(str).str.strip()

    if multiclass:
        df["attack_type"] = df["Label"]

    df["label"] = (df["Label"] != "BENIGN").astype(int)
    df = df.drop(columns=["Label"])
    return df