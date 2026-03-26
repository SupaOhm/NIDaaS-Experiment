"""
preprocess.py
-------------
Cleans the raw CIC-IDS2017 DataFrame and prepares train/val/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with NaN or infinite values.
    """
    # Replace inf with NaN, then drop
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    before = len(df_clean)
    df_clean = df_clean.dropna()
    after = len(df_clean)

    print(f"[preprocess] Cleaned {before - after:,} rows with NaN/Inf. Remaining: {after:,}")
    return df_clean.reset_index(drop=True)

def split_dataset(y, train_size=0.7, val_size=0.1, random_state=42):
    """
    Returns indices for stratified train, validation, and test sets.
    """
    n = len(y)
    indices = np.arange(n)
    
    # First split: Train vs (Val + Test)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_size, stratify=y, random_state=random_state
    )
    
    # Second split: Val vs Test
    relative_val_size = val_size / (1.0 - train_size)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=relative_val_size, stratify=y[temp_idx], random_state=random_state
    )
    
    print(f"[preprocess] Split → train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    return train_idx, val_idx, test_idx

def scale_data(X_train, X_val, X_test):
    """
    StandardScaler fitted on training data only.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s
