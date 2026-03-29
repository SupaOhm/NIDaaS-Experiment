"""
exp_dedupe_tune_exp2.py
-------------------------
Orchestrates a fair, bounded optimization search for the Experiment 2 deduplication pipeline.
Tuning targets:
1. Bloom+Exact (Proposed Stage-2 system)
2. Partitioned Bloom+Exact (Parallel Stage-2 system)

Includes fixed baseline anchor runs for fair comparison.
Logs all results to results/tables/exp2_tuning_log.csv.
"""
import os
import pandas as pd
import numpy as np
import time
from experiments.exp_dedupe_efficiency import run_efficiency_experiment

def run_tuning_campaign(data_path: str, n_records: int = 100000, confirmation_n: int = 500000):
    print("\n" + "="*80)
    print(" EXPERIMENT 2: BOUNDED OPTIMIZATION SEARCH ")
    print("="*80)
    
    # 1. Setup logging
    log_file = "results/tables/exp2_tuning_log.csv"
    os.makedirs("results/tables", exist_ok=True)
    
    # Define search space
    error_rates = [0.001, 0.01, 0.05]
    initial_capacities = [100, 1000, 5000, 10000]
    num_partitions_list = [2, 4, 8]
    duplicate_ratios = [0.1, 0.5, 1.0]
    
    trials = []
    trial_id = 1
    
    def _log_trial(mode, dup_ratio, err, cap, part, results, counts, notes=""):
        # Extract matching algorithm result
        alg_res = next((r for r in results if mode in r['Algorithm']), None)
        if not alg_res:
            # Handle No Dedupe or Hash-Only which might have slightly different names in results
            if mode == "No Dedupe":
                alg_res = next((r for r in results if "No Dedupe" in r['Algorithm']), None)
            elif mode == "Hash-Only":
                alg_res = next((r for r in results if "Hash-Only Exact" in r['Algorithm']), None)
        
        if not alg_res:
            print(f"      [!] Warning: Mode {mode} not found in results.")
            return

        # Correctness guard: Check for anomalous dedupe rates
        # (Compare against Hash-Only if available in this run)
        hash_only_res = next((r for r in results if "Hash-Only Exact" in r['Algorithm']), None)
        valid = True
        if hash_only_res:
            diff = abs(alg_res["Dedupe Rate (%)"] - hash_only_res["Dedupe Rate (%)"])
            if diff > 5.0: # arbitrary 5% threshold for "obviously inconsistent"
                notes += f" [Anomalous Dedupe Rate: diff={diff:.2f}%]"
                valid = False

        trial_data = {
            "trial_id": trial_id,
            "mode": mode,
            "duplicate_fraction": dup_ratio,
            "cleaned_rows_before_injection": counts["cleaned_rows_before_injection"],
            "rows_after_injection": counts["rows_after_injection"],
            "rows_processed": counts["rows_processed"],
            "bloom_error_rate": err if "Bloom" in mode else None,
            "bloom_initial_capacity": cap if "Bloom" in mode else None,
            "num_partitions": part if "Partitioned" in mode else None,
            "dedupe_rate_pct": alg_res["Dedupe Rate (%)"],
            "e2e_elapsed_s": alg_res["E2E Elapsed (s)"],
            "throughput_rec_s": alg_res["Throughput (rec/s)"],
            "peak_memory_mb": alg_res["Peak Memory (MB)"],
            "memory_delta_mb": alg_res["Memory Delta (MB)"],
            "notes": notes,
            "valid": valid
        }
        trials.append(trial_data)
        return trial_id + 1

    # PHASE 1: SEARCH (100k records)
    print(f"\n[PHASE 1] Starting bounded search on validation slice (n={n_records:,})...")
    
    # Pruned search space to stay within 30 trials
    # error_rates = [0.01, 0.05]
    # initial_capacities = [1000, 5000]
    # partitions = [2, 4, 8]
    # base (2) * d(3) + bloom(4) * d(3) + part(3) * d(3) = 6 + 12 + 9 = 27 trials
    
    for d in duplicate_ratios:
        print(f"\n--- Testing Duplicate Level: d={d} ---")
        
        # 1. Fixed Baseline Anchors
        res, cnt = run_efficiency_experiment(data_path, n_records=n_records, duplicate_ratio=d)
        trial_id = _log_trial("No Dedupe", d, None, None, None, res, cnt, "Baseline Anchor")
        trial_id = _log_trial("Hash-Only", d, None, None, None, res, cnt, "Baseline Anchor")
        
        # 2. Plain Bloom+Exact Search (4 trials per level)
        for err in [0.01, 0.05]:
            for cap in [1000, 5000]:
                print(f"      Trial {trial_id}: Bloom+Exact (err={err}, cap={cap})")
                res, cnt = run_efficiency_experiment(data_path, n_records=n_records, duplicate_ratio=d, 
                                                   error_rate=err, initial_capacity=cap)
                trial_id = _log_trial("Bloom+Exact", d, err, cap, 1, res, cnt)

        # 3. Partitioned Bloom+Exact Search (3 trials per level)
        # Use a stable error/capacity to explore partitioning
        for part in num_partitions_list:
            print(f"      Trial {trial_id}: Partitioned Bloom+Exact (err=0.01, cap=5000, part={part})")
            res, cnt = run_efficiency_experiment(data_path, n_records=n_records, duplicate_ratio=d, 
                                               error_rate=0.01, initial_capacity=5000, num_partitions=part)
            trial_id = _log_trial("Partitioned Bloom+Exact", d, 0.01, 5000, part, res, cnt)

    # Save search log
    df_trials = pd.DataFrame(trials)
    df_trials.to_csv(log_file, index=False)
    print(f"\n[Search Complete] Trial log saved to {log_file}")

    # PHASE 2: SELECTION & CONFIRMATION
    print(f"\n[PHASE 2] Identifying best candidates and running confirmation (n={confirmation_n:,})...")
    
    # Filter for valid runs
    df_valid = df_trials[df_trials['valid'] == True]
    
    # Best Bloom+Exact (smallest e2e_elapsed_s averaged across d levels or just absolute best)
    # We'll take the configuration that has the best throughput on average
    best_be = df_valid[df_valid['mode'] == 'Bloom+Exact'].groupby(['bloom_error_rate', 'bloom_initial_capacity'])['throughput_rec_s'].mean().idxmax()
    best_pbe = df_valid[df_valid['mode'] == 'Partitioned Bloom+Exact'].groupby(['bloom_error_rate', 'bloom_initial_capacity', 'num_partitions'])['throughput_rec_s'].mean().idxmax()
    
    print(f"  -> Best Bloom+Exact found: err={best_be[0]}, cap={best_be[1]}")
    print(f"  -> Best Partitioned Bloom+Exact found: err={best_pbe[0]}, cap={best_pbe[1]}, part={best_pbe[2]}")
    
    print(f"\nRunning Final Confirmation Test (n={confirmation_n})...")
    # We'll run them all for d=0.5 for the final head-to-head comparison
    res_final, cnt_final = run_efficiency_experiment(
        data_path, n_records=confirmation_n, duplicate_ratio=0.5,
        error_rate=best_be[0], initial_capacity=int(best_be[1]), num_partitions=int(best_pbe[2])
    )
    
    print("\n" + "="*80)
    print(" FINAL CONFIRMATION RESULTS (n=500k, d=0.5) ")
    print("="*80)
    df_final = pd.DataFrame(res_final)
    print(df_final.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("-n", "--records", type=int, default=100000)
    parser.add_argument("-c", "--confirmation", type=int, default=500000)
    args = parser.parse_args()
    
    run_tuning_campaign(args.data, args.records, args.confirmation)
