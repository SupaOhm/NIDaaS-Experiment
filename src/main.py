"""
main.py
-------
Entry point for the NIDaaS research experiment repo.

Usage
-----
  # Run detection comparison experiment (with real data):
  python src/main.py --experiment detection --data data/

  # Run detection comparison (smoke test — no real data needed):
  python src/main.py --experiment detection --smoke

  # Run efficiency / dedup comparison experiment:
  python src/main.py --experiment efficiency --data data/

  # Run efficiency experiment (smoke test):
  python src/main.py --experiment efficiency --smoke

  # Run both experiments in sequence:
  python src/main.py --experiment all --smoke
"""

import sys
import os
import argparse

# --- OPENMP CONFLICT FIX ---
# XGBoost (via libomp) and PyTorch can sometimes conflict on macOS, 
# leading to deadlocks or "Error #15". This flag resolves it.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"     # Prevents thread-pool deadlock
# ---------------------------

# Ensure src/ is on the path regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(__file__))

from experiment_detection import run_detection_experiment
from experiment_efficiency import run_efficiency_experiment
from experiment_parallel_dedupe import run_parallel_experiment


def main():
    parser = argparse.ArgumentParser(
        description=(
            "NIDaaS Research Experiment Runner\n"
            "-----------------------------------\n"
            "Experiment 1 (detection):  compare Hybrid vs RF / XGBoost / LR\n"
            "Experiment 2 (efficiency): compare Bloom+Exact dedup vs baselines\n"
            "Experiment 3 (parallel):   compare Centralized vs Partitioned Multi-worker architecture\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=["detection", "1", "d", "efficiency", "2", "e", "parallel", "3", "p", "all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--data",
        default="data/",
        help="Path to CIC-IDS2017 CSV file or directory (default: data/)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run with synthetic data — no real dataset required. Good for testing setup.",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=10_000,
        help="Number of records to use in the efficiency experiment (default: 10000)",
    )
    parser.add_argument(
        "--dup-rate",
        type=float,
        default=0.3,
        help="Fraction of duplicate records to inject in the efficiency experiment (default: 0.3)",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=4,
        help="Number of multi-processing partitions for Experiment 3 (default: 4)",
    )

    args = parser.parse_args()

    if args.experiment in ("detection", "1", "d", "all"):
        run_detection_experiment(data_path=args.data, smoke=args.smoke)

    if args.experiment in ("efficiency", "2", "e", "all"):
        run_efficiency_experiment(
            data_path=args.data,
            smoke=args.smoke,
            n_records=args.n_records,
            dup_rate=args.dup_rate,
        )

    if args.experiment in ("parallel", "3", "p", "all"):
        # Smoke testing for parallel uses small n_records
        n_rec = 1000 if args.smoke else args.n_records
        run_parallel_experiment(
            data_path=args.data,
            n_records=n_rec,
            dup_rate=args.dup_rate,
            num_partitions=args.partitions
        )

    print("\n✓ All requested experiments finished.")
    print("  Check results/tables/ for CSVs and results/figures/ for plots.")


if __name__ == "__main__":
    main()
