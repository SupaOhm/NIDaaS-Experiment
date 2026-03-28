"""
preprocess.py
-------------
Utility functions for experiment preprocessing.

Main responsibilities:
- Clean loaded DataFrames
- Apply file/row limits for quick experiments
- Select usable numeric feature columns
- Fill missing values
- Prepare novelty-detection train/validation/test splits

This module keeps preprocessing logic separate from experiment orchestration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.replace(["Infinity", "inf", "-inf", "NaN", "nan"], np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)

    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype(str).str.strip()

    return out


def apply_dataset_limits(
    df: pd.DataFrame,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if "__source_file__" in out.columns and max_files is not None:
        selected_files = sorted(out["__source_file__"].dropna().unique().tolist())[:max_files]
        out = out[out["__source_file__"].isin(selected_files)].copy()

    if "__source_file__" in out.columns and max_rows_per_file is not None:
        out = (
            out.groupby("__source_file__", group_keys=False)
            .head(max_rows_per_file)
            .reset_index(drop=True)
        )
    elif max_rows_per_file is not None:
        out = out.head(max_rows_per_file).reset_index(drop=True)

    return out.reset_index(drop=True)


def pick_numeric_feature_columns(df: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    good_cols = []
    for c in feature_cols:
        series = df[c]
        if series.notna().sum() < max(10, int(0.1 * len(df))):
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        good_cols.append(c)

    if not good_cols:
        raise ValueError("No usable numeric feature columns found.")

    return good_cols


def fit_fill_values(X_train_normal_df: pd.DataFrame) -> pd.Series:
    return X_train_normal_df.median(numeric_only=True)


def apply_fill_values(X_df: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    X = X_df.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(fill_values)
    X = X.replace([np.inf, -np.inf], 0.0)
    return X


def prepare_splits(
    X_df: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Novelty setting:
    - train normal only
    - val normal only
    - test mixed benign + attack
    """
    normal_mask = (y == 0)
    attack_mask = (y == 1)

    X_normal = X_df.loc[normal_mask].reset_index(drop=True)
    X_attack = X_df.loc[attack_mask].reset_index(drop=True)

    if len(X_normal) < 100:
        raise ValueError("Too few benign samples for novelty experiment.")

    X_train_normal, X_temp_normal = train_test_split(
        X_normal,
        test_size=0.4,
        random_state=random_state,
        shuffle=True,
    )

    X_val_normal, X_test_normal = train_test_split(
        X_temp_normal,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )

    X_test = pd.concat([X_test_normal, X_attack], ignore_index=True)
    y_test = np.concatenate([
        np.zeros(len(X_test_normal), dtype=int),
        np.ones(len(X_attack), dtype=int),
    ])

    return X_train_normal, X_val_normal, X_test, y_test