"""
main.py
-------
Central entry point for running NIDaaS experiments.

Supported experiments:
- rf_novelty
- rf_novelty_best
"""

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from exp_rf_novelty import run_rf_novelty_experiment
from best_config import get_best_experiment_config


def main():
    parser = argparse.ArgumentParser(description="NIDaaS experiment runner")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    # Manual config
    rf_parser = subparsers.add_parser(
        "rf_novelty",
        help="Run Random-Forest novelty experiment with manual parameters",
    )
    rf_parser.add_argument("--input-dir", type=str, required=True, help="Directory or CSV path for CIC-IDS2017.")
    rf_parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per CSV after loading.")
    rf_parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of source CSV files.")
    rf_parser.add_argument("--use-signature", action="store_true", help="Run signature-first hybrid experiment.")
    rf_parser.add_argument("--n-estimators", type=int, default=100)
    rf_parser.add_argument("--max-depth", type=int, default=10)
    rf_parser.add_argument("--min-samples-leaf", type=int, default=5)
    rf_parser.add_argument("--alpha", type=float, default=0.7)
    rf_parser.add_argument("--quantiles", nargs="*", type=float, default=[0.95])
    rf_parser.add_argument("--random-state", type=int, default=42)
    rf_parser.add_argument("--result-dir", type=str, default="result", help="Directory to store experiment outputs.")

    # Saved best config
    best_parser = subparsers.add_parser(
        "rf_novelty_best",
        help="Run Random-Forest novelty experiment using saved best config",
    )
    best_parser.add_argument("--input-dir", type=str, required=True, help="Directory or CSV path for CIC-IDS2017.")
    best_parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per CSV after loading.")
    best_parser.add_argument("--max-files", type=int, default=8, help="Optional cap on number of source CSV files.")
    best_parser.add_argument("--random-state", type=int, default=42)
    best_parser.add_argument("--result-dir", type=str, default="result_best", help="Directory to store experiment outputs.")

    args = parser.parse_args()

    if args.experiment == "rf_novelty":
        run_rf_novelty_experiment(args)

    elif args.experiment == "rf_novelty_best":
        best_cfg = get_best_experiment_config()
        rf_cfg = best_cfg["rf_config"]

        args.use_signature = best_cfg.get("use_signature", True)
        args.n_estimators = rf_cfg["n_estimators"]
        args.max_depth = rf_cfg["max_depth"]
        args.min_samples_leaf = rf_cfg["min_samples_leaf"]
        args.alpha = rf_cfg["alpha"]
        args.quantiles = rf_cfg["quantiles"]
        args.random_state = args.random_state if args.random_state is not None else rf_cfg.get("random_state", 42)

        print("\nUsing saved best config:")
        print(best_cfg["rf_config"])
        print(best_cfg["signature_config"])

        run_rf_novelty_experiment(args)

    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")


if __name__ == "__main__":
    main()