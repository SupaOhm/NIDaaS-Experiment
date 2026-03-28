"""
exp_supervised_rf.py
--------------------
Runs a supervised Random Forest classifier experiment on CIC-IDS2017 data.

Pipeline overview:
1. Load dataset using data_loader.py
2. Optionally limit the number of source files or rows per file
3. Select usable numeric features
4. Split data into stratified train / validation / test sets
5. Train a supervised RandomForestClassifier on BENIGN vs ATTACK labels
6. Sweep probability thresholds to compare precision / recall / F1 / FAR
7. Save metrics, metadata, feature importances, and prediction outputs

This experiment is intended for:
- precision-oriented baselines
- comparing supervised RF against RF novelty and hybrid baselines
- threshold tuning for precision-first deployment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import load_dataset
from evaluation import compute_metrics, print_metrics
from preprocess import (
    apply_dataset_limits,
    apply_fill_values,
    clean_dataframe,
    fit_fill_values,
    pick_numeric_feature_columns,
)
from result_utils import save_metadata, save_predictions, save_summary


def create_result_dir(base_dir: str = "result_supervised_rf") -> str:
    project_root = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    run_id = datetime.now().strftime("supervised_rf_%Y%m%d_%H%M%S")
    result_dir = os.path.join(project_root, base_dir, run_id)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def threshold_tag(thr: float) -> str:
    return str(thr).replace(".", "p")


def prepare_supervised_splits(
    X_df: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split:
    - train: 60%
    - val:   20%
    - test:  20%
    """
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
        shuffle=True,
    )

    return (
        X_train_df.reset_index(drop=True),
        X_val_df.reset_index(drop=True),
        X_test_df.reset_index(drop=True),
        y_train.to_numpy(dtype=int),
        y_val.to_numpy(dtype=int),
        y_test.to_numpy(dtype=int),
    )


def save_feature_importances(
    result_dir: str,
    feature_cols: list[str],
    model: RandomForestClassifier,
):
    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(result_dir, "feature_importances.csv"), index=False)
    return imp_df


def run_supervised_rf_experiment(args):
    result_dir = create_result_dir(args.result_dir)
    print(f"\nResults will be saved to: {result_dir}")

    df = load_dataset(args.input_dir, multiclass=False)
    df = apply_dataset_limits(
        df,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )
    df = clean_dataframe(df)

    if "label" not in df.columns:
        raise KeyError("Expected unified 'label' column from data_loader.load_dataset().")

    y = df["label"].astype(int)

    exclude_cols = ["label"]
    if "__source_file__" in df.columns:
        exclude_cols.append("__source_file__")
    if "attack_type" in df.columns:
        exclude_cols.append("attack_type")

    feature_cols = pick_numeric_feature_columns(df, exclude_cols=exclude_cols)
    X_df = df[feature_cols].copy()

    benign_rows = int((y == 0).sum())
    attack_rows = int((y == 1).sum())
    source_files = sorted(df["__source_file__"].dropna().unique().tolist()) if "__source_file__" in df.columns else []

    print(f"\nTotal rows: {len(df)}")
    print(f"Benign rows: {benign_rows}")
    print(f"Attack rows: {attack_rows}")
    print(f"Feature count: {len(feature_cols)}")
    if source_files:
        print(f"Source files used: {len(source_files)}")

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = prepare_supervised_splits(
        X_df, y, random_state=args.random_state
    )

    fill_values = fit_fill_values(X_train_df)
    X_train = apply_fill_values(X_train_df, fill_values).to_numpy(dtype=float)
    X_val = apply_fill_values(X_val_df, fill_values).to_numpy(dtype=float)
    X_test = apply_fill_values(X_test_df, fill_values).to_numpy(dtype=float)

    metadata = {
        "input_dir": args.input_dir,
        "source_files": source_files,
        "max_rows_per_file": args.max_rows_per_file,
        "max_files": args.max_files,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "thresholds": args.thresholds,
        "total_rows": int(len(df)),
        "benign_rows": benign_rows,
        "attack_rows": attack_rows,
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "train_size": int(len(X_train_df)),
        "val_size": int(len(X_val_df)),
        "test_size": int(len(X_test_df)),
        "train_benign_size": int((y_train == 0).sum()),
        "train_attack_size": int((y_train == 1).sum()),
        "val_benign_size": int((y_val == 0).sum()),
        "val_attack_size": int((y_val == 1).sum()),
        "test_benign_size": int((y_test == 0).sum()),
        "test_attack_size": int((y_test == 1).sum()),
    }
    save_metadata(result_dir, metadata)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - t0
    print(f"\nSupervised RF fit time: {train_elapsed:.6f}s")

    t_val0 = time.perf_counter()
    val_scores = clf.predict_proba(X_val)[:, 1]
    val_score_elapsed = time.perf_counter() - t_val0
    print(f"Validation score precompute time: {val_score_elapsed:.6f}s")

    t_test0 = time.perf_counter()
    test_scores = clf.predict_proba(X_test)[:, 1]
    test_score_elapsed = time.perf_counter() - t_test0
    print(f"Test score precompute time: {test_score_elapsed:.6f}s")

    imp_df = save_feature_importances(result_dir, feature_cols, clf)

    summary_rows = []

    for thr in args.thresholds:
        tag = threshold_tag(thr)

        t1 = time.perf_counter()
        y_pred = (test_scores >= thr).astype(int)
        infer_elapsed = time.perf_counter() - t1

        metrics = compute_metrics(y_test, y_pred, infer_elapsed)
        print_metrics(f"Supervised RF (threshold={thr})", metrics)

        summary_rows.append(
            {
                "model": "supervised_rf",
                "threshold": thr,
                "train_time_s": float(train_elapsed),
                **metrics,
            }
        )

        save_predictions(
            result_dir=result_dir,
            filename=f"supervised_rf_thr_{tag}_predictions.csv",
            y_true=y_test,
            y_pred=y_pred,
            scores=test_scores,
        )

    save_summary(result_dir, summary_rows)
    print(f"\nSaved summary to: {os.path.join(result_dir, 'summary.csv')}")

    print("\nTop 20 feature importances:")
    for _, row in imp_df.head(20).iterrows():
        print(f" - {row['feature']}: {row['importance']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Supervised Random Forest experiment for IDS.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory or CSV path for CIC-IDS2017.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per CSV after loading.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of source CSV files.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--result-dir", type=str, default="result_supervised_rf", help="Directory to store experiment outputs.")
    args = parser.parse_args()
    run_supervised_rf_experiment(args)


if __name__ == "__main__":
    main()