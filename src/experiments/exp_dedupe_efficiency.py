"""
exp_dedupe_efficiency.py
------------------------
Validates the theoretical throughput and memory bounds of the Deduplication pipeline.
Compares O(1) Sliding-Window baselines against the NIDSaaS Bloom+Exact architecture.
"""
import pandas as pd
from data.loader import load_dataset
from metrics.resource_tracker import BenchmarkTracker
from dedupe.no_dedupe import NoDeduplicator
from dedupe.hash_exact import HashExactDeduplicator
from dedupe.bloom_only import BloomOnlyDeduplicator
from dedupe.bloom_exact import BloomExactDeduplicator
from dedupe.partitioned_bloom_exact import PartitionedDeduplicatorRunner

def run_efficiency_experiment(data_path: str, n_records: int = None, window_size: int = 100000):
    print("\n" + "="*70)
    print(" EXPERIMENT 2: DEDUPLICATION EFFICIENCY & MEMORY BOUNDS ")
    print("="*70)
    
    df = load_dataset(data_path)
    if n_records is not None:
        df = df.head(n_records)
    
    # We explicitly do NOT strip NaNs or Scale here because a production gateway 
    # deduplicates raw traffic before it even reaches the deep learning pipeline!
    records = df.to_dict(orient="records")
    total = len(records)
    print(f"\n[exp_efficiency] Loaded {total:,} raw network flow events for pipeline benchmark.")
    print(f"[exp_efficiency] TTL Eviction Sliding Window strictly bounded to W={window_size:,} records.")

    # Simulated downstream ML processing latency (e.g., PyTorch inference)
    IDS_LATENCY_S = 50 / 1_000_000 
        
    results = []

    def _benchmark(name, deduper_instance):
        print(f"  -> Benchmarking: {name}")
        with BenchmarkTracker() as tracker:
            for r in records:
                deduper_instance.process_record(r)
                
        stats = deduper_instance.get_stats()
        total_elapsed = tracker.elapsed_s + (stats["ids_evaluations"] * IDS_LATENCY_S)
        
        results.append({
            "Algorithm": name,
            "Records Processed": stats["total_records"],
            "Dedupe Rate (%)": stats["dedupe_rate_%"],
            "E2E Elapsed (s)": round(total_elapsed, 4),
            "Throughput (rec/s)": round(stats["total_records"] / max(0.0001, total_elapsed), 1),
            "Peak Memory (MB)": round(tracker.peak_mem_mb, 2)
        })

    _benchmark("No Dedupe (Direct IDS)", NoDeduplicator(window_size))
    _benchmark("Hash-Only Exact", HashExactDeduplicator(window_size))
    _benchmark("Bloom-Only Filter", BloomOnlyDeduplicator(window_size, error_rate=0.01))
    _benchmark("Bloom+Exact (NIDSaaS)", BloomExactDeduplicator(window_size, error_rate=0.01))
    
    # ── Partitioned Variant ──
    print(f"  -> Benchmarking: Partitioned Bloom+Exact (Multiprocessing 4W)")
    runner = PartitionedDeduplicatorRunner(num_partitions=4, window_size=window_size)
    with BenchmarkTracker() as tracker:
        p_stats = runner.run_parallel(records)
        
    p_elapsed = p_stats["e2e_elapsed_s"] + (p_stats["ids_evaluations"] * IDS_LATENCY_S)
    results.append({
        "Algorithm": "Partitioned Bloom+Exact",
        "Records Processed": p_stats["total_records"],
        "Dedupe Rate (%)": p_stats["dedupe_rate_%"],
        "E2E Elapsed (s)": round(p_elapsed, 4),
        "Throughput (rec/s)": round(p_stats["total_records"] / max(0.0001, p_elapsed), 1),
        "Peak Memory (MB)": round(tracker.peak_mem_mb, 2)
    })

    print("\n" + "="*85)
    print(" PIPELINE THROUGHPUT & EFFICIENCY RESULTS ")
    print("="*85)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("="*85 + "\n")
    return results
