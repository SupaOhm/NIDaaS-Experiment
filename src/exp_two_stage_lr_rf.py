"""
exp_two_stage_lr_rf.py
----------------------
Experiment runner for the proposed two-stage detection architecture:

Stage 1: Logistic Regression screening
Stage 2: Random Forest confirmation
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

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
from two_stage_detector import TwoStageConfig, TwoStageLRRFDetector


def create_result_dir(base_dir: str = "result_two_stage_lr_rf") -> str:
    project_root = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    run_id = datetime.now().strftime("two_stage_lr_rf_%Y%m%d_%H%M%S")
    result_dir = os.path.join(project_root, base_dir, run_id)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def threshold_tag(thr: float) -> str:
    return str(thr).replace(".", "p")


def prepare_supervised_splits(
    X_df: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
    from sklearn.model_selection import train_test_split

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


def run_two_stage_lr_rf_experiment(args):
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
        "lr_c": args.lr_c,
        "lr_max_iter": args.lr_max_iter,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "screen_thresholds": args.screen_thresholds,
        "confirm_thresholds": args.confirm_thresholds,
        "random_state": args.random_state,
        "total_rows": int(len(df)),
        "benign_rows": benign_rows,
        "attack_rows": attack_rows,
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "train_size": int(len(X_train_df)),
        "val_size": int(len(X_val_df)),
        "test_size": int(len(X_test_df)),
    }
    save_metadata(result_dir, metadata)

    detector = TwoStageLRRFDetector(
        TwoStageConfig(
            lr_c=args.lr_c,
            lr_max_iter=args.lr_max_iter,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
            rf_min_samples_leaf=args.rf_min_samples_leaf,
            random_state=args.random_state,
        )
    )

    t0 = time.perf_counter()
    detector.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - t0
    print(f"\nTwo-stage model fit time: {train_elapsed:.6f}s")

    t_score0 = time.perf_counter()
    s1_val_scores = detector.stage1_scores(X_val)
    s1_test_scores = detector.stage1_scores(X_test)
    s2_test_scores = detector.stage2_scores(X_test)
    score_elapsed = time.perf_counter() - t_score0
    print(f"Stage score precompute time: {score_elapsed:.6f}s")

    summary_rows = []

    for s_thr in args.screen_thresholds:
        # Stage 1 only
        t1 = time.perf_counter()
        stage1_pred = (s1_test_scores >= s_thr).astype(int)
        elapsed1 = time.perf_counter() - t1

        metrics_stage1 = compute_metrics(y_test, stage1_pred, elapsed1)
        print_metrics(f"Stage 1 LR Screen Only (threshold={s_thr})", metrics_stage1)

        candidate_idx = np.where(stage1_pred == 1)[0]

        summary_rows.append(
            {
                "model": "stage1_lr_screen_only",
                "screen_threshold": s_thr,
                "confirm_threshold": np.nan,
                "train_time_s": float(train_elapsed),
                "candidate_count": int(len(candidate_idx)),
                **metrics_stage1,
            }
        )

        save_predictions(
            result_dir=result_dir,
            filename=f"stage1_lr_screen_only_thr_{threshold_tag(s_thr)}_predictions.csv",
            y_true=y_test,
            y_pred=stage1_pred,
            scores=s1_test_scores,
        )

        for c_thr in args.confirm_thresholds:
            t2 = time.perf_counter()
            final_pred, _, _ = detector.predict(
                X_test,
                screen_threshold=s_thr,
                confirm_threshold=c_thr,
            )
            elapsed2 = time.perf_counter() - t2

            metrics_two_stage = compute_metrics(y_test, final_pred, elapsed2)
            print_metrics(
                f"Two-Stage LR->RF (screen={s_thr}, confirm={c_thr})",
                metrics_two_stage,
            )

            summary_rows.append(
                {
                    "model": "two_stage_lr_supervised_rf",
                    "screen_threshold": s_thr,
                    "confirm_threshold": c_thr,
                    "train_time_s": float(train_elapsed),
                    "candidate_count": int(len(candidate_idx)),
                    "confirmed_count": int(final_pred.sum()),
                    **metrics_two_stage,
                }
            )

            two_stage_scores = np.zeros(len(y_test), dtype=float)
            if len(candidate_idx) > 0:
                two_stage_scores[candidate_idx] = s2_test_scores[candidate_idx]

            save_predictions(
                result_dir=result_dir,
                filename=(
                    f"two_stage_lr_rf_screen_{threshold_tag(s_thr)}"
                    f"_confirm_{threshold_tag(c_thr)}_predictions.csv"
                ),
                y_true=y_test,
                y_pred=final_pred,
                scores=two_stage_scores,
            )

    save_summary(result_dir, summary_rows)
    print(f"\nSaved summary to: {os.path.join(result_dir, 'summary.csv')}")


def main():
    parser = argparse.ArgumentParser(description="Two-stage LR screener + RF confirmer experiment.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory or CSV path for CIC-IDS2017.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per CSV after loading.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of source CSV files.")

    parser.add_argument("--lr-c", type=float, default=1.0)
    parser.add_argument("--lr-max-iter", type=int, default=1000)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=12)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=5)

    parser.add_argument("--screen-thresholds", nargs="*", type=float, default=[0.2, 0.3, 0.4])
    parser.add_argument("--confirm-thresholds", nargs="*", type=float, default=[0.8, 0.9, 0.95])

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--result-dir", type=str, default="result_two_stage_lr_rf", help="Directory to store experiment outputs.")

    args = parser.parse_args()
    run_two_stage_lr_rf_experiment(args)


if __name__ == "__main__":
    main()