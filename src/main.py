"""
main.py
-------
Experiment CLI Router targeting the 3 academic verification domains:
1=Detection Precision, 2=Deduplication Efficiency, 3=Pipeline Scalability.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from experiments.exp_detection import run_detection_experiment
from experiments.exp_dedupe_efficiency import run_efficiency_experiment
from experiments.exp_scaling import run_scaling_experiment
from experiments.exp_dedupe_search import run_dedupe_grid_search

def main():
    parser = argparse.ArgumentParser(
        description="NIDSaaS Research Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/main.py -e 1 --data data/        # Detection Precision (full dataset)
  python src/main.py -e 2 --data data/ -n 100000   # Dedupe Efficiency (100K records)
  python src/main.py -e 3 --data data/        # Scaling Sensitivity
""")
    
    parser.add_argument("-e", "--experiment", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Experiment: 1=Detection, 2=Dedup-Efficiency, 3=Scaling, 4=GridSearch")
    parser.add_argument("--data", type=str, default="data/",
                        help="Path to data directory with ISCX CSVs")
    parser.add_argument("-n", "--records", type=int, default=None,
                        help="Limit to N records (None = full dataset)")
    parser.add_argument("-d", "--duplicate", type=float, default=None,
                        help="Duplicate pressure ratio (0.0-1.0). For Experiment 2, appends sampled duplicate rows; for Experiment 4, controls synthetic dup ratio.")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    if args.experiment == 1:
        print(" EXPERIMENT 1: DETECTION PRECISION ")
        run_detection_experiment(args.data)
    elif args.experiment == 2:
        print(" EXPERIMENT 2: RAW CLEANED-EVENT DEDUPE EFFICIENCY & MEMORY BOUNDS ")
        run_efficiency_experiment(args.data, args.records, duplicate_ratio=args.duplicate)
    elif args.experiment == 3:
        print(" EXPERIMENT 3: PIPELINE SCALABILITY SENSITIVITY ")
        run_scaling_experiment(args.data, args.records)
    elif args.experiment == 4:
        print(" EXPERIMENT 4: DEDUPE GRID SEARCH ")
        run_dedupe_grid_search(args.data, args.records, duplicate_ratio=args.duplicate)
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
