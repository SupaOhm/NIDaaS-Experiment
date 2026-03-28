"""
exp_rf_novelty.py
-----------------
Runs Random-Forest novelty detection experiments on CIC-IDS2017 data.

Pipeline overview:
1. Load dataset using data_loader.py
2. Optionally limit the number of source files or rows per file
3. Select usable numeric features
4. Split data in novelty-detection style
5. Train RFNoveltyScorer
6. Evaluate RF-only performance at multiple thresholds
7. Optionally run signature-first and hybrid evaluation
8. Save metrics, metadata, and predictions into the result folder
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

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
    prepare_splits,
)
from result_utils import (
    create_result_dir,
    quantile_tag,
    save_metadata,
    save_predictions,
    save_summary,
)
from rf_novelty import RFNoveltyConfig, RFNoveltyScorer
from signature_detector import SignatureConfig, predict as signature_predict
from hybrid_detector import predict_hybrid, predict_rf_only


def run_rf_novelty_experiment(args):
    result_dir = create_result_dir(CURRENT_DIR, args.result_dir)
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

    X_train_normal_df, X_val_normal_df, X_test_df, y_test = prepare_splits(
        X_df, y, random_state=args.random_state
    )

    fill_values = fit_fill_values(X_train_normal_df)
    X_train_normal = apply_fill_values(X_train_normal_df, fill_values).to_numpy(dtype=float)
    X_val_normal = apply_fill_values(X_val_normal_df, fill_values).to_numpy(dtype=float)
    X_test = apply_fill_values(X_test_df, fill_values).to_numpy(dtype=float)

    metadata = {
        "input_dir": args.input_dir,
        "source_files": source_files,
        "max_rows_per_file": args.max_rows_per_file,
        "max_files": args.max_files,
        "use_signature": args.use_signature,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "alpha": args.alpha,
        "quantiles": args.quantiles,
        "random_state": args.random_state,
        "total_rows": int(len(df)),
        "benign_rows": benign_rows,
        "attack_rows": attack_rows,
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "train_normal_size": int(len(X_train_normal_df)),
        "val_normal_size": int(len(X_val_normal_df)),
        "test_size": int(len(X_test_df)),
        "test_benign_size": int((y_test == 0).sum()),
        "test_attack_size": int((y_test == 1).sum()),
    }
    save_metadata(result_dir, metadata)

    config = RFNoveltyConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        alpha=args.alpha,
    )
    scorer = RFNoveltyScorer(config=config)

    summary_rows = []

    # Train RF once
    t0 = time.perf_counter()
    scorer.fit(X_train_normal)
    train_elapsed = time.perf_counter() - t0
    print(f"\nRF novelty fit time: {train_elapsed:.6f}s")

    # Precompute RF scores on the full test set once
    t_score0 = time.perf_counter()
    scores_rf_all = scorer.score_samples(X_test)
    score_elapsed = time.perf_counter() - t_score0
    print(f"RF score precompute time: {score_elapsed:.6f}s")

    # Precompute signature stage once
    sig_hits = None
    sig_details = None
    sig_elapsed = 0.0

    if args.use_signature:
        sig_cfg = SignatureConfig()

        t_sig0 = time.perf_counter()
        sig_hits, sig_details = signature_predict(
            X_test_df,
            config=sig_cfg,
            return_details=True,
        )
        sig_elapsed = time.perf_counter() - t_sig0

        metrics_sig = compute_metrics(y_test, sig_hits, sig_elapsed)
        print_metrics("Signature Only", metrics_sig)

        summary_rows.append({
            "model": "signature_only",
            "quantile": args.quantiles[0],
            "train_time_s": 0.0,
            **metrics_sig,
        })

        q_tag0 = quantile_tag(args.quantiles[0])
        save_predictions(
            result_dir=result_dir,
            filename=f"signature_only_q{q_tag0}_predictions.csv",
            y_true=y_test,
            y_pred=sig_hits,
            scores=None,
        )

        rule_summary = sig_details.sum(axis=0).reset_index()
        rule_summary.columns = ["rule_name", "hit_count"]
        rule_summary.to_csv(
            os.path.join(result_dir, f"signature_rule_summary_q{q_tag0}.csv"),
            index=False,
        )

    # Evaluate thresholds
    for q in args.quantiles:
        q_tag = quantile_tag(q)

        scorer.fit_threshold(X_val_normal, quantile=q)
        threshold = scorer.threshold_

        # RF-only prediction from precomputed scores
        t1 = time.perf_counter()
        y_pred_rf, scores_rf = predict_rf_only(scores_rf_all, threshold)
        infer_elapsed_rf = time.perf_counter() - t1

        metrics_rf = compute_metrics(y_test, y_pred_rf, infer_elapsed_rf)
        print_metrics(f"RF Novelty Only (q={q})", metrics_rf)

        summary_rows.append({
            "model": "rf_novelty_only",
            "quantile": q,
            "train_time_s": float(train_elapsed),
            **metrics_rf,
        })

        save_predictions(
            result_dir=result_dir,
            filename=f"rf_only_q{q_tag}_predictions.csv",
            y_true=y_test,
            y_pred=y_pred_rf,
            scores=scores_rf,
        )

        if args.use_signature:
            t_h0 = time.perf_counter()
            hybrid_pred, hybrid_scores = predict_hybrid(
                scores_rf_all=scores_rf_all,
                threshold=threshold,
                sig_hits=sig_hits,
                delta=0.03,
            )
            hybrid_elapsed = time.perf_counter() - t_h0

            metrics_hybrid = compute_metrics(y_test, hybrid_pred, hybrid_elapsed)
            print_metrics(f"Hybrid Signature + RF Novelty (q={q})", metrics_hybrid)

            summary_rows.append({
                "model": "hybrid_signature_rf_novelty",
                "quantile": q,
                "train_time_s": float(train_elapsed),
                **metrics_hybrid,
            })

            save_predictions(
                result_dir=result_dir,
                filename=f"hybrid_q{q_tag}_predictions.csv",
                y_true=y_test,
                y_pred=hybrid_pred,
                scores=hybrid_scores,
            )

    save_summary(result_dir, summary_rows)
    print(f"\nSaved summary to: {os.path.join(result_dir, 'summary.csv')}")

    print("\nTop 20 features used:")
    for c in feature_cols[:20]:
        print(" -", c)


def main():
    parser = argparse.ArgumentParser(description="RF novelty experiment for IDS.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory or CSV path for CIC-IDS2017.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per CSV after loading.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of source CSV files.")
    parser.add_argument("--use-signature", action="store_true", help="Run signature-first hybrid experiment.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--quantiles", nargs="*", type=float, default=[0.95])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--result-dir", type=str, default="result", help="Directory to store experiment outputs.")
    args = parser.parse_args()
    run_rf_novelty_experiment(args)


if __name__ == "__main__":
    main()