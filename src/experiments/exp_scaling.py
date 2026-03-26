"""
exp_scaling.py
--------------
Sensitivity analysis. Evaluates the Bloom+Exact pipeline under various 
Sliding Window (TTL) constraints and Thread Counts to prove the scalability claims.
"""
from dedupe.partitioned_bloom_exact import PartitionedDeduplicatorRunner
from dedupe.bloom_exact import BloomExactDeduplicator
from data.loader import load_dataset
from metrics.resource_tracker import BenchmarkTracker
import pandas as pd

def run_scaling_experiment(data_path: str, n_records: int = None):
    print("\n" + "="*70)
    print(" EXPERIMENT 3: SENSITIVITY & SCALING ANALYSIS ")
    print("="*70)
    
    df = load_dataset(data_path)
    if n_records is not None:
        df = df.head(n_records)
    records = df.to_dict(orient="records")
    
    # 1. Thread Scaling (Partition Variants)
    print("\n[exp_scaling] Testing Multi-Core Partitioning Throughput...")
    partition_results = []
    # Test across 1, 2, 4 cores
    for partitions in [1, 2, 4]:
        runner = PartitionedDeduplicatorRunner(num_partitions=partitions, window_size=50000)
        with BenchmarkTracker() as t:
            p_stats = runner.run_parallel(records)
        partition_results.append({
            "Workers": partitions,
            "Elapsed (s)": round(p_stats["e2e_elapsed_s"], 2),
            "Throughput": round(p_stats["total_records"] / max(0.001, p_stats["e2e_elapsed_s"]), 1)
        })
    print("\n" + pd.DataFrame(partition_results).to_string(index=False) + "\n")
    
    # 2. TTL Sliding Window Scaling
    print("[exp_scaling] Testing Absolute Memory Bounds over diverse Sliding Window (TTL) Sizes...")
    window_results = []
    # Test across different window capacities
    for w_size in [1000, 10000, 100000]:
        deduper = BloomExactDeduplicator(window_size=w_size)
        with BenchmarkTracker() as t:
            for r in records: deduper.process_record(r)
        
        window_results.append({
            "Window Size": w_size,
            "Peak Mem (MB)": round(t.peak_mem_mb, 2),
            "Dedupe Rate %": deduper.get_stats()["dedupe_rate_%"]
        })
    print("\n" + pd.DataFrame(window_results).to_string(index=False) + "\n")
    print("[exp_scaling] Experiment 3 completed successfully.\n")
