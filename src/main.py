"""
main.py
-------
Central entry point for running NIDaaS experiments.

Primary proposed model:
- two_stage
- two_stage_best

Baselines:
- rf_novelty
- rf_novelty_best
- supervised_rf
"""

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from exp_rf_novelty import run_rf_novelty_experiment
from exp_supervised_rf import run_supervised_rf_experiment
from exp_two_stage_lr_rf import run_two_stage_lr_rf_experiment


def main():
    parser = argparse.ArgumentParser(description="NIDaaS experiment runner")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    # Proposed two-stage model
    two_stage_parser = subparsers.add_parser(
        "two_stage",
        help="Run the proposed two-stage LR -> RF detection model",
    )
    two_stage_parser.add_argument("--input-dir", type=str, required=True)
    two_stage_parser.add_argument("--max-rows-per-file", type=int, default=None)
    two_stage_parser.add_argument("--max-files", type=int, default=8)
    two_stage_parser.add_argument("--lr-c", type=float, default=1.0)
    two_stage_parser.add_argument("--lr-max-iter", type=int, default=1000)
    two_stage_parser.add_argument("--rf-n-estimators", type=int, default=200)
    two_stage_parser.add_argument("--rf-max-depth", type=int, default=12)
    two_stage_parser.add_argument("--rf-min-samples-leaf", type=int, default=5)
    two_stage_parser.add_argument("--screen-thresholds", nargs="*", type=float, default=[0.2])
    two_stage_parser.add_argument("--confirm-thresholds", nargs="*", type=float, default=[0.8, 0.95])
    two_stage_parser.add_argument("--random-state", type=int, default=42)
    two_stage_parser.add_argument("--result-dir", type=str, default="result_two_stage_lr_rf")

    two_stage_best_parser = subparsers.add_parser(
        "two_stage_best",
        help="Run the saved best two-stage configuration",
    )
    two_stage_best_parser.add_argument("--input-dir", type=str, required=True)
    two_stage_best_parser.add_argument("--max-rows-per-file", type=int, default=None)
    two_stage_best_parser.add_argument("--max-files", type=int, default=8)
    two_stage_best_parser.add_argument("--random-state", type=int, default=42)
    two_stage_best_parser.add_argument("--result-dir", type=str, default="result_two_stage_best")

    # Baselines kept for comparison
    rf_parser = subparsers.add_parser("rf_novelty", help="Run RF novelty baseline")
    rf_parser.add_argument("--input-dir", type=str, required=True)
    rf_parser.add_argument("--max-rows-per-file", type=int, default=None)
    rf_parser.add_argument("--max-files", type=int, default=None)
    rf_parser.add_argument("--use-signature", action="store_true")
    rf_parser.add_argument("--n-estimators", type=int, default=100)
    rf_parser.add_argument("--max-depth", type=int, default=10)
    rf_parser.add_argument("--min-samples-leaf", type=int, default=5)
    rf_parser.add_argument("--alpha", type=float, default=0.7)
    rf_parser.add_argument("--quantiles", nargs="*", type=float, default=[0.95])
    rf_parser.add_argument("--random-state", type=int, default=42)
    rf_parser.add_argument("--result-dir", type=str, default="result")

    supervised_rf_parser = subparsers.add_parser("supervised_rf", help="Run supervised RF baseline")
    supervised_rf_parser.add_argument("--input-dir", type=str, required=True)
    supervised_rf_parser.add_argument("--max-rows-per-file", type=int, default=None)
    supervised_rf_parser.add_argument("--max-files", type=int, default=None)
    supervised_rf_parser.add_argument("--n-estimators", type=int, default=200)
    supervised_rf_parser.add_argument("--max-depth", type=int, default=12)
    supervised_rf_parser.add_argument("--min-samples-leaf", type=int, default=5)
    supervised_rf_parser.add_argument("--thresholds", nargs="*", type=float, default=[0.8, 0.9, 0.95])
    supervised_rf_parser.add_argument("--random-state", type=int, default=42)
    supervised_rf_parser.add_argument("--result-dir", type=str, default="result_supervised_rf")

    args = parser.parse_args()

    if args.experiment == "two_stage":
        run_two_stage_lr_rf_experiment(args)

    elif args.experiment == "two_stage_best":
        args.lr_c = 1.0
        args.lr_max_iter = 1000
        args.rf_n_estimators = 200
        args.rf_max_depth = 12
        args.rf_min_samples_leaf = 5
        args.screen_thresholds = [0.2]
        args.confirm_thresholds = [0.8, 0.95]

        print("\nUsing saved best two-stage config:")
        print({
            "stage1": {"model": "logistic_regression", "threshold": 0.2},
            "stage2": {"model": "random_forest", "thresholds": [0.8, 0.95]},
        })

        run_two_stage_lr_rf_experiment(args)

    elif args.experiment == "rf_novelty":
        run_rf_novelty_experiment(args)

    elif args.experiment == "supervised_rf":
        run_supervised_rf_experiment(args)

    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")


if __name__ == "__main__":
    main()