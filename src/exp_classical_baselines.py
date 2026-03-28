"""
exp_classical_baselines.py
--------------------------
Runs classical baseline models on CIC-IDS2017 data.

Included baselines:
- Logistic Regression
- Decision Tree
- Gaussian Naive Bayes
- Random Forest
- Isolation Forest

Outputs:
- summary.csv
- metadata.json
- per-model prediction CSV files

Notes:
- Supervised models use stratified train/validation/test splits
- Isolation Forest is trained on benign-only training data
- Supervised models sweep probability thresholds
- Isolation Forest uses anomaly-score thresholds derived from benign validation data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def create_result_dir(base_dir: str = "result_classical_baselines") -> str:
    project_root = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    run_id = datetime.now().strftime("classical_baselines_%Y%m%d_%H%M%S")
    result_dir = os.path.join(project_root, base_dir, run_id)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def tag_value(x: float) -> str:
    return str(x).replace(".", "p")


def prepare_supervised_splits(
    X_df: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
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


def save_feature_importances_if_available(
    result_dir: str,
    feature_cols: list[str],
    model_name: str,
    model,
):
    if not hasattr(model, "feature_importances_"):
        return

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    imp_df.to_csv(
        os.path.join(result_dir, f"{model_name}_feature_importances.csv"),
        index=False,
    )


def save_coefficients_if_available(
    result_dir: str,
    feature_cols: list[str],
    model_name: str,
    model,
):
    if not hasattr(model, "coef_"):
        return

    coef = np.ravel(model.coef_)
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(
        os.path.join(result_dir, f"{model_name}_coefficients.csv"),
        index=False,
    )


def evaluate_supervised_model(
    result_dir: str,
    model_name: str,
    display_name: str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    thresholds: Iterable[float],
    summary_rows: list[dict],
):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - t0
    print(f"\n{display_name} fit time: {train_elapsed:.6f}s")

    t_score0 = time.perf_counter()
    test_scores = model.predict_proba(X_test)[:, 1]
    score_elapsed = time.perf_counter() - t_score0
    print(f"{display_name} score precompute time: {score_elapsed:.6f}s")

    for thr in thresholds:
        tag = tag_value(thr)

        t1 = time.perf_counter()
        y_pred = (test_scores >= thr).astype(int)
        infer_elapsed = time.perf_counter() - t1

        metrics = compute_metrics(y_test, y_pred, infer_elapsed)
        print_metrics(f"{display_name} (threshold={thr})", metrics)

        summary_rows.append(
            {
                "model": model_name,
                "threshold": thr,
                "train_time_s": float(train_elapsed),
                **metrics,
            }
        )

        save_predictions(
            result_dir=result_dir,
            filename=f"{model_name}_thr_{tag}_predictions.csv",
            y_true=y_test,
            y_pred=y_pred,
            scores=test_scores,
        )

    return model, test_scores, train_elapsed


def evaluate_isolation_forest(
    result_dir: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    quantiles: Iterable[float],
    random_state: int,
    summary_rows: list[dict],
):
    model_name = "isolation_forest"
    display_name = "Isolation Forest"

    X_train_normal = X_train_raw[y_train == 0]
    X_val_normal = X_val_raw[y_val == 0]

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    iso.fit(X_train_normal)
    train_elapsed = time.perf_counter() - t0
    print(f"\n{display_name} fit time: {train_elapsed:.6f}s")

    # Higher anomaly_score => more anomalous
    t_score0 = time.perf_counter()
    val_anomaly_scores = -iso.score_samples(X_val_normal)
    test_anomaly_scores = -iso.score_samples(X_test_raw)
    score_elapsed = time.perf_counter() - t_score0
    print(f"{display_name} score precompute time: {score_elapsed:.6f}s")

    for q in quantiles:
        tag = tag_value(q)
        threshold = float(np.quantile(val_anomaly_scores, q))

        t1 = time.perf_counter()
        y_pred = (test_anomaly_scores >= threshold).astype(int)
        infer_elapsed = time.perf_counter() - t1

        metrics = compute_metrics(y_test, y_pred, infer_elapsed)
        print_metrics(f"{display_name} (quantile={q})", metrics)

        summary_rows.append(
            {
                "model": model_name,
                "threshold": np.nan,
                "quantile": q,
                "train_time_s": float(train_elapsed),
                **metrics,
            }
        )

        save_predictions(
            result_dir=result_dir,
            filename=f"{model_name}_q_{tag}_predictions.csv",
            y_true=y_test,
            y_pred=y_pred,
            scores=test_anomaly_scores,
        )


def run_classical_baselines(args):
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

    X_train_raw = apply_fill_values(X_train_df, fill_values).to_numpy(dtype=float)
    X_val_raw = apply_fill_values(X_val_df, fill_values).to_numpy(dtype=float)
    X_test_raw = apply_fill_values(X_test_df, fill_values).to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    metadata = {
        "input_dir": args.input_dir,
        "source_files": source_files,
        "max_rows_per_file": args.max_rows_per_file,
        "max_files": args.max_files,
        "thresholds": args.thresholds,
        "iforest_quantiles": args.iforest_quantiles,
        "random_state": args.random_state,
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

    summary_rows: list[dict] = []

    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=args.random_state,
    )
    lr, _, _ = evaluate_supervised_model(
        result_dir=result_dir,
        model_name="logistic_regression",
        display_name="Logistic Regression",
        model=lr,
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        thresholds=args.thresholds,
        summary_rows=summary_rows,
    )
    save_coefficients_if_available(result_dir, feature_cols, "logistic_regression", lr)

    dt = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        random_state=args.random_state,
    )
    dt, _, _ = evaluate_supervised_model(
        result_dir=result_dir,
        model_name="decision_tree",
        display_name="Decision Tree",
        model=dt,
        X_train=X_train_raw,
        y_train=y_train,
        X_test=X_test_raw,
        y_test=y_test,
        thresholds=args.thresholds,
        summary_rows=summary_rows,
    )
    save_feature_importances_if_available(result_dir, feature_cols, "decision_tree", dt)

    gnb = GaussianNB()
    evaluate_supervised_model(
        result_dir=result_dir,
        model_name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        model=gnb,
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        thresholds=args.thresholds,
        summary_rows=summary_rows,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=args.random_state,
        n_jobs=-1,
    )
    rf, _, _ = evaluate_supervised_model(
        result_dir=result_dir,
        model_name="random_forest",
        display_name="Random Forest",
        model=rf,
        X_train=X_train_raw,
        y_train=y_train,
        X_test=X_test_raw,
        y_test=y_test,
        thresholds=args.thresholds,
        summary_rows=summary_rows,
    )
    save_feature_importances_if_available(result_dir, feature_cols, "random_forest", rf)

    evaluate_isolation_forest(
        result_dir=result_dir,
        X_train_raw=X_train_scaled,
        y_train=y_train,
        X_val_raw=X_val_scaled,
        y_val=y_val,
        X_test_raw=X_test_scaled,
        y_test=y_test,
        quantiles=args.iforest_quantiles,
        random_state=args.random_state,
        summary_rows=summary_rows,
    )

    save_summary(result_dir, summary_rows)
    print(f"\nSaved summary to: {os.path.join(result_dir, 'summary.csv')}")


def main():
    parser = argparse.ArgumentParser(description="Classical baseline experiments for IDS.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory or CSV path for CIC-IDS2017.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per CSV after loading.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of source CSV files.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--iforest-quantiles", nargs="*", type=float, default=[0.95, 0.975, 0.99])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--result-dir", type=str, default="result_classical_baselines", help="Directory to store experiment outputs.")
    args = parser.parse_args()
    run_classical_baselines(args)


if __name__ == "__main__":
    main()