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
        epilog=(
            "Available Experiments:\n"
            "  1 / detection   : Evaluates Snort, LSTM, and Hybrid Precision & F1\n"
            "  2 / efficiency  : Benchmarks fair Sliding-Window deduplication topologies\n"
            "  3 / scaling     : Evaluates Sensitivity to Window Size and Parallel CPU counts\n"
            "  4 / search      : Systematic Grid Search across Parameterized 2-Stage Architectures\n"
        )
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=["1", "detection", "2", "efficiency", "3", "scaling", "4", "search", "all"],
        default="all",
        help="Target experiment variant (default: all)"
    )
    parser.add_argument("--data", "-d", default="data/", help="Path to CIC-IDS2017 .csv files")
    parser.add_argument("--records", "-n", type=int, default=None, help="Limit number of records to process (default: all)")
    
    args = parser.parse_args()
    
    if args.experiment in ("1", "detection", "all"):
        run_detection_experiment(args.data, args.records)
        
    if args.experiment in ("2", "efficiency", "all"):
        run_efficiency_experiment(args.data, args.records)
        
    if args.experiment in ("3", "scaling", "all"):
        run_scaling_experiment(args.data, args.records)
        
    if args.experiment in ("4", "search", "all"):
        run_dedupe_grid_search(args.data, args.records)
        
if __name__ == "__main__":
    main()
