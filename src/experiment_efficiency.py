"""
experiment_efficiency.py
------------------------
Experiment Group 2: Deduplication Efficiency & System Throughput.

This experiment compares 4 system configurations:
  1. No Dedup + IDS (Baseline)
  2. Exact-Hash Filter + IDS
  3. Bloom Filter ONLY + IDS (Risky - may cause misses)
  4. Bloom + Exact Filter + IDS (Proposed NIDaaS architecture)

Metrics:
  - Total latency (Filter + IDS)
  - Memory usage
  - System Throughput (records/sec)
  - Filter Accuracy (to show security loss in 'Bloom-Only' mode)
"""

import os
import sys
import time
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import tracemalloc
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_dataset
from preprocess import clean
from dedupe import Deduplicator, canonicalize, fingerprint
from metrics import efficiency_report, save_table, save_bar_chart

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")

# Simulated cost of running a complex IDS (Signature + LSTM) 
# per network flow. 50 microseconds is a realistic research placeholder.
SIMULATED_IDS_LATENCY_US = 50 

def _make_synthetic_records(n: int = 5_000, dup_rate: float = 0.3) -> list:
    base_templates = [
        {
            "Source IP": f"192.168.1.{i % 255}", "Destination IP": f"10.0.0.{i % 100}",
            "Source Port": 1024 + (i % 60000), "Destination Port": [80, 443, 22, 53][i % 4],
            "Protocol": i % 3, "Total Fwd Packets": np.random.randint(1, 500),
            "Total Backward Packets": np.random.randint(0, 300),
            "Total Length of Fwd Packets": np.random.randint(0, 10000),
            "Total Length of Bwd Packets": np.random.randint(0, 8000),
        } for i in range(int(n * (1 - dup_rate)))
    ]
    records = list(base_templates)
    for i in range(n - len(records)):
        records.append(base_templates[i % len(base_templates)])
    np.random.shuffle(records)
    return records


def simulate_ids_processing():
    """Simulates the heavy CPU work of the IDS component."""
    time.sleep(SIMULATED_IDS_LATENCY_US / 1_000_000)


def _run_experiment_condition(label, records, dedup_fn):
    """
    Standard runner for each efficiency condition.
    `dedup_fn` returns (passed_to_ids, is_duplicate).
    """
    import gc
    gc.collect()  # Force OS to reclaim memory from previous runs
    
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
        "total": len(records),
        "ids_runs": num_ids_runs,
        "elapsed_s": elapsed,
        "mem_delta": mem_delta_mb
    }


def run_efficiency_experiment(data_path, smoke=False, n_records=10000, dup_rate=0.4):
    print("\n" + "=" * 65)
    print("EXPERIMENT 2 — Deduplication & IDS Throughput Comparison")
    print("=" * 65)

    # 1. Prepare Data
    if smoke:
        from experiment_efficiency import _make_synthetic_records
        records = _make_synthetic_records(n=n_records, dup_rate=dup_rate)
    else:
        df = load_dataset(data_path).sample(frac=1).reset_index(drop=True).head(n_records)
        n_dup = int(n_records * dup_rate)
        df_dups = df.sample(n=n_dup, replace=True)
        records = pd.concat([df, df_dups]).sample(frac=1).to_dict(orient="records")
        # Free up massive DataFrame memory immediately
        del df, df_dups
        import gc; gc.collect()

    total = len(records)
    print(f"[experiment_efficiency] Data ready: {total:,} records ({dup_rate*100:.1f}% duplicates)")

    results = []

    # --- Condition 1: No Dedup + IDS ---
    def logic_no_dedup(r):
        return True, False  # Every record passes to IDS
    
    print("[experiment_efficiency] Running: No Dedup + IDS...")
    results.append(_run_experiment_condition("No Dedup + IDS", records, logic_no_dedup))

    # --- Condition 2: Exact Hash Only + IDS ---
    hash_cache = set()
    def logic_hash_only(r):
        fp = fingerprint(canonicalize(r))
        if fp in hash_cache: return False, True
        hash_cache.add(fp)
        return True, False
    
    print("[experiment_efficiency] Running: Hash-Only Filter + IDS...")
    results.append(_run_experiment_condition("Hash-Only + IDS", records, logic_hash_only))
    del hash_cache; import gc; gc.collect()

    # --- Condition 3: Bloom Filter ONLY + IDS ---
    from pybloom_live import BloomFilter
    bloom = BloomFilter(capacity=total, error_rate=0.01)
    def logic_bloom_only(r):
        fp = fingerprint(canonicalize(r))
        if fp in bloom: return False, True  # Risk: False positive means unique flow MISSES ids!
        bloom.add(fp)
        return True, False

    print("[experiment_efficiency] Running: Bloom-Only Filter + IDS...")
    results.append(_run_experiment_condition("Bloom-Only + IDS", records, logic_bloom_only))
    del bloom; import gc; gc.collect()

    # --- Condition 4: Bloom + Exact (Proposed) + IDS ---
    deduper = Deduplicator(capacity=total)
    def logic_proposed(r):
        is_dup = deduper.is_duplicate(r)
        return not is_dup, is_dup

    print("[experiment_efficiency] Running: Bloom+Exact (Proposed) + IDS...")
    results.append(_run_experiment_condition("Bloom+Exact + IDS", records, logic_proposed))
    del deduper; import gc; gc.collect()

    # --- Process and Save Results ---
    final_reports = []
    for r in results:
        final_reports.append(efficiency_report(
            r["condition"], r["total"], r["elapsed_s"], r["mem_delta"],
            dedup_stats={"dedup_rate_%": round(100 * (1 - r["ids_runs"]/r["total"]), 2)}
        ))

    save_table(final_reports, os.path.join(TABLES_DIR, "efficiency_results.csv"))
    
    # Visualisation
    df_res = pd.DataFrame(final_reports)
    save_bar_chart(df_res, "condition", "throughput_rec_s", "Overall System Throughput",
                   os.path.join(FIGURES_DIR, "throughput_comparison.png"), "Records / Sec")
    
    save_bar_chart(df_res, "condition", "memory_delta_mb", "Memory Consumption (Overhead)",
                   os.path.join(FIGURES_DIR, "memory_comparison.png"), "MB")

    print("\n[experiment_efficiency] ✓ Experiment 2 complete.")
    return final_reports

if __name__ == "__main__":
    run_efficiency_experiment("data/", smoke=True)
