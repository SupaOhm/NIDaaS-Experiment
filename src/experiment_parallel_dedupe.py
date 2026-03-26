"""
experiment_parallel_dedupe.py
-----------------------------
Benchmarks centralized vs partitioned (parallel) deduplication architecture.
"""

import os
import sys
import time
import argparse
import tracemalloc
import numpy as np
import pandas as pd
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_dataset
from parallel_dedupe import PartitionedDeduplicatorRunner
from dedupe import Deduplicator, canonicalize, fingerprint
from metrics import efficiency_report, save_table, save_bar_chart
from experiment_efficiency import simulate_ids_processing

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")

def _run_centralized_condition(label: str, records: List[dict], dedup_fn) -> dict:
    import gc
    gc.collect()

    tracemalloc.start()
    t0 = time.perf_counter()
    num_ids_runs = 0
    
    for r in records:
        passed_to_ids, is_duplicate = dedup_fn(r)
        if passed_to_ids:
            num_ids_runs += 1
            simulate_ids_processing()
            
    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_delta_mb = peak_mem / 1e6
    
    return {
        "condition": label,
        "total_records": len(records),
        "ids_runs": num_ids_runs,
        "elapsed_s": elapsed,
        "mem_delta": mem_delta_mb
    }

def run_parallel_experiment(data_path, n_records=100_000, dup_rate=0.4, num_partitions=4):
    print("\n" + "=" * 65)
    print("EXPERIMENT 3 — Parallel Partitioned Architecture Comparison")
    print("=" * 65)

    # 1. Prepare Data with Tenant IDs
    df = load_dataset(data_path).sample(frac=1).reset_index(drop=True).head(n_records)
    n_dup = int(n_records * dup_rate)
    df_dups = df.sample(n=n_dup, replace=True)
    records = pd.concat([df, df_dups]).sample(frac=1).to_dict(orient="records")
    
    # Inject multi-tenant distribution
    tenant_ids = [f"Tenant_{i}" for i in range(1, 11)]  # 10 tenants
    for r in records:
        r["tenant_id"] = np.random.choice(tenant_ids)
        
    total = len(records)
    print(f"[experiment_parallel] Data ready: {total:,} records ({dup_rate*100:.1f}% DB-level duplicates). Assigned to 10 mock tenants.")
    
    del df, df_dups
    import gc; gc.collect()

    results = []

    # -------------------------------------------------------------
    # BASELINE 1: Centralized Hash-Only
    # -------------------------------------------------------------
    print("[experiment_parallel] Running: Centralized Hash-Only + IDS...")
    hash_cache = set()
    def logic_hash_only(r):
        fp = fingerprint(canonicalize(r))
        if fp in hash_cache: return False, True
        hash_cache.add(fp)
        return True, False
    
    res = _run_centralized_condition("Central Hash-Only", records, logic_hash_only)
    results.append(res)
    del hash_cache; gc.collect()

    # -------------------------------------------------------------
    # BASELINE 2: Centralized Bloom + Exact
    # -------------------------------------------------------------
    print("[experiment_parallel] Running: Centralized Bloom+Exact + IDS...")
    central_deduper = Deduplicator(capacity=total)
    def logic_centralized_bloom(r):
        is_dup = central_deduper.is_duplicate(r)
        return not is_dup, is_dup

    res = _run_centralized_condition("Central Bloom+Exact", records, logic_centralized_bloom)
    results.append(res)
    del central_deduper; gc.collect()

    # -------------------------------------------------------------
    # PROPOSED: Parallel Partitioned Bloom + Exact
    # -------------------------------------------------------------
    print(f"[experiment_parallel] Running: Parallel Partitioned Bloom+Exact ({num_partitions} Workers)...")
    import tracemalloc
    tracemalloc.start()
    
    runner = PartitionedDeduplicatorRunner(
        num_partitions=num_partitions,
        capacity_per_partition=int(total / num_partitions * 1.5)
    )
    parallel_stats = runner.run_parallel(records, algo="bloom_exact")
    
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    results.append({
        "condition": f"Parallel Bloom+Exact ({num_partitions}W)",
        "total_records": parallel_stats["total_records"],
        "ids_runs": parallel_stats["ids_runs"],
        "elapsed_s": parallel_stats["e2e_elapsed_s"],
        "mem_delta": peak_mem / 1e6
    })

    # -------------------------------------------------------------
    # Process & Save
    # -------------------------------------------------------------
    final_reports = []
    for r in results:
        dedup_rate = round(100 * (1 - r["ids_runs"] / r["total_records"]), 2)
        rep = efficiency_report(
            r["condition"], r["total_records"], r["elapsed_s"], r["mem_delta"],
            dedup_stats={"dedup_rate_%": dedup_rate}
        )
        final_reports.append(rep)

    out_csv = os.path.join(TABLES_DIR, "parallel_efficiency_results.csv")
    save_table(final_reports, out_csv)

    df_res = pd.DataFrame(final_reports)
    
    # Throughput graph
    save_bar_chart(
        df_res, "condition", "throughput_rec_s", 
        "Parallel Architecture Throughput Impact",
        os.path.join(FIGURES_DIR, "parallel_throughput_comparison.png"),
        "Records / Sec"
    )
    
    # Latency Graph
    save_bar_chart(
        df_res, "condition", "avg_latency_us", 
        "Parallel Architecture Average Latency",
        os.path.join(FIGURES_DIR, "parallel_latency_comparison.png"),
        "Microseconds"
    )

    print("\n[experiment_parallel] ✓ Experiment 3 (Parallel) complete.")
    return final_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/")
    parser.add_argument("--n-records", type=int, default=10_000)
    parser.add_argument("--partitions", type=int, default=4)
    args = parser.parse_args()
    run_parallel_experiment(args.data, n_records=args.n_records, num_partitions=args.partitions)
