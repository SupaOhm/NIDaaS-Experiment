"""
main.py
-------
Main entry point for the NIDaaS research experiment repository.

This script is responsible for:
1) Parsing command-line arguments
2) Dispatching the selected experiment
3) Passing runtime options such as dataset path, smoke mode, and profile

Supported experiment groups
---------------------------
1. detection
   Compare detection performance between:
   - Signature-only baseline
   - Classical ML baselines
   - Proposed Hybrid model (Signature + LSTM)

2. efficiency
   Compare deduplication efficiency between:
   - No dedup
   - Exact cache / traditional baseline
   - Proposed Bloom + Exact dedup approach

3. parallel
   Compare scalability between:
   - Centralized processing
   - Partitioned / multi-worker processing

4. all
   Run all experiment groups in sequence

Usage
-----
# Run detection experiment with real dataset
python src/main.py --experiment detection --data data/

# Run detection experiment with quick profile
python src/main.py --experiment detection --data data/ --profile fast

# Run detection experiment with full profile
python src/main.py --experiment detection --data data/ --profile full

# Run detection experiment in smoke mode (synthetic data, quick setup test)
python src/main.py --experiment detection --smoke

# Run efficiency experiment with real dataset
python src/main.py --experiment efficiency --data data/

# Run efficiency experiment in smoke mode
python src/main.py --experiment efficiency --smoke

# Run parallel experiment
python src/main.py --experiment parallel --data data/

# Run all experiments in smoke mode
python src/main.py --experiment all --smoke

Notes
-----
- Use --smoke when you want to verify that the pipeline runs correctly
  without requiring the full dataset.
- Use --profile fast for quicker CPU-friendly runs.
- Use --profile full for more complete final evaluation.
- Dataset paths are usually given relative to the project root, e.g. data/
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
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

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
    parser.add_argument(
    "--profile",
    choices=["fast", "full"],
    default="fast",
    help="Experiment profile: fast for quick CPU runs, full for final evaluation",
)

    args = parser.parse_args()

    if args.experiment in ("detection", "1", "d", "all"):
        run_detection_experiment(
        data_path=args.data,
        smoke=args.smoke,
        profile=args.profile,
    )

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
