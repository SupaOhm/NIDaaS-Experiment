"""
exp_dedupe_efficiency.py
------------------------
Validates the theoretical throughput and memory bounds of the Deduplication pipeline.
Compares O(1) Sliding-Window baselines against the NIDSaaS Bloom+Exact architecture.
"""
import hashlib
import os
import pandas as pd
from data.loader import load_dataset
from data.preprocess import clean_for_experiment2
from metrics.resource_tracker import BenchmarkTracker
from dedupe.no_dedupe import NoDeduplicator
from dedupe.hash_exact import HashExactDeduplicator, HashArraySearchWorstCaseDeduplicator
from dedupe.bloom_only import BloomOnlyDeduplicator
from dedupe.bloom_exact import BloomExactDeduplicator
from dedupe.partitioned_bloom_exact import PartitionedDeduplicatorRunner


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_compact_dedupe_records(df: pd.DataFrame):
    """
    Build compact records once so all algorithms reuse the same scoped fingerprint payload.
    Record format: ((tenant, log_type), fingerprint_bytes)
    """
    src_col = _first_present(df, ["Source IP", "Src IP", "src_ip", "source_ip"])
    dst_col = _first_present(df, ["Destination IP", "Dst IP", "dst_ip", "destination_ip"])
    fwd_col = _first_present(df, ["Total Length of Fwd Packets", "Fwd Bytes", "fwd_bytes"])
    tenant_col = _first_present(df, ["tenant_id", "Tenant ID", "Source IP", "Src IP", "src_ip"])
    log_col = _first_present(df, ["log_type", "Log Type", "Protocol", "protocol"])

    n = len(df)
    default_empty = [""] * n

    src_vals = df[src_col].fillna("").astype(str).tolist() if src_col else default_empty
    dst_vals = df[dst_col].fillna("").astype(str).tolist() if dst_col else default_empty
    fwd_vals = df[fwd_col].fillna("").astype(str).tolist() if fwd_col else default_empty
    tenant_vals = df[tenant_col].fillna("").astype(str).tolist() if tenant_col else default_empty
    log_vals = df[log_col].fillna("").astype(str).tolist() if log_col else default_empty

    return [
        (
            (tenant_vals[i], log_vals[i]),
            hashlib.sha256(f"{src_vals[i]}|{dst_vals[i]}|{fwd_vals[i]}".encode("utf-8")).digest(),
        )
        for i in range(n)
    ]


def _apply_duplicate_pressure(df: pd.DataFrame, duplicate_ratio: float | None, random_state: int = 42) -> pd.DataFrame:
    """
    Apply configurable duplicate pressure by appending sampled existing rows.

    duplicate_ratio=r means append r * N duplicate rows to N base rows.
    - r=0.0 -> unchanged
    - r=1.0 -> append N duplicates (roughly doubles benchmark input size)
    """
    if duplicate_ratio is None:
        return df
    if duplicate_ratio < 0.0 or duplicate_ratio > 1.0:
        raise ValueError("[exp_efficiency] duplicate_ratio must be in [0.0, 1.0]. Example: -d 0.5")
    if duplicate_ratio == 0.0 or df.empty:
        return df

    base_n = len(df)
    extra_n = int(base_n * duplicate_ratio)
    if extra_n <= 0:
        return df

    dup_rows = df.sample(n=extra_n, replace=True, random_state=random_state)
    out = pd.concat([df, dup_rows], ignore_index=True)
    # Keep arrival ordering realistic and deterministic.
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

def run_efficiency_experiment(data_path: str, n_records: int = None, window_size: int = 100000, duplicate_ratio: float | None = None, 
                             error_rate: float = 0.01, initial_capacity: int = 1000, num_partitions: int = 4):
    print("\n" + "="*70)
    print(" EXPERIMENT 2: RAW CLEANED-EVENT DEDUPE EFFICIENCY & MEMORY BOUNDS ")
    print("="*70)
    
    df = load_dataset(data_path)
    if n_records is not None:
        df = df.head(n_records)

    # Experiment 2 default path: clean only, no aggregation.
    df = clean_for_experiment2(df)
    clean_n = len(df)

    df = _apply_duplicate_pressure(df, duplicate_ratio)
    pressured_n = len(df)

    records = _build_compact_dedupe_records(df)
    total = len(records)
    include_worst_case = os.getenv("EXP2_INCLUDE_WORST_CASE", "0").strip().lower() in {"1", "true", "yes", "on"}
    include_bloom_only = os.getenv("EXP2_INCLUDE_BLOOM_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}

    print(f"\n[exp_efficiency][clean-only] Loaded {total:,} cleaned raw flow events for dedupe benchmark (no aggregation).")
    if duplicate_ratio is not None:
        print(
            f"[exp_efficiency] Duplicate pressure applied: d={duplicate_ratio:.2f} | "
            f"base_rows={clean_n:,} -> benchmark_rows={pressured_n:,}"
        )
    print(f"[exp_efficiency] TTL Eviction Sliding Window strictly bounded to W={window_size:,} records.")
    print(
        "[exp_efficiency] Note: The default Hash-Only baseline is a practical full-cache exact cache (set-based, no TTL).\n"
        "Bloom+Exact retains scoped sliding-window behavior; Bloom-Only is optional diagnostic."
    )
    if include_bloom_only:
        print("[exp_efficiency] Optional diagnostic baseline enabled: Bloom-Only Filter.")
    if include_worst_case:
        print("[exp_efficiency] Optional supplementary baseline enabled: Hash-Only Array Search (Worst-Case Exact).")

    # Simulated downstream ML processing latency (e.g., PyTorch inference)
    IDS_LATENCY_S = 50 / 1_000_000 
        
    results = []

    def _benchmark(name, deduper_instance):
        print(f"  -> Benchmarking: {name}")
        with BenchmarkTracker() as tracker:
            deduper_instance.process_records(records)
        stats = deduper_instance.get_stats()
        print(f"     [Rows] processed={stats['total_records']:,} (input={total:,})")
        # Print diagnostic info if available (for Bloom-based dedupers)
        if hasattr(deduper_instance, 'get_diagnostic_info'):
            print(f"     {deduper_instance.get_diagnostic_info()}")
        # Add simulated IDS cost for every record that reaches IDS stage (including No Dedupe baseline)
        # This simulates downstream IDS cost after dedupe.
        total_elapsed = tracker.elapsed_s + (stats["ids_evaluations"] * IDS_LATENCY_S)
        mem_delta = getattr(tracker, "mem_delta_mb", None)
        results.append({
            "Algorithm": name,
            "Records Processed": stats["total_records"],
            "Dedupe Rate (%)": stats["dedupe_rate_%"],
            "E2E Elapsed (s)": round(total_elapsed, 4),
            "Throughput (rec/s)": round(stats["total_records"] / max(0.0001, total_elapsed), 1),
            "Peak Memory (MB)": round(tracker.peak_mem_mb, 2),
            "Memory Delta (MB)": round(mem_delta, 2) if mem_delta is not None else ""
        })

    _benchmark("No Dedupe (Direct IDS)", NoDeduplicator(window_size))
    _benchmark("Hash-Only Exact Cache", HashExactDeduplicator(window_size=None))
    if include_worst_case:
        _benchmark("Hash-Only Array Search (Worst-Case Exact)", HashArraySearchWorstCaseDeduplicator(window_size=None))
    if include_bloom_only:
        _benchmark("Bloom-Only Filter", BloomOnlyDeduplicator(window_size, error_rate=error_rate))
    _benchmark("Bloom+Exact (NIDSaaS)", BloomExactDeduplicator(window_size, error_rate=error_rate, initial_capacity=initial_capacity))
    
    # ── Partitioned Variant ──
    print(f"  -> Benchmarking: Partitioned Bloom+Exact (Multiprocessing {num_partitions}W)")
    runner = PartitionedDeduplicatorRunner(num_partitions=num_partitions, window_size=window_size, error_rate=error_rate, initial_capacity=initial_capacity)
    with BenchmarkTracker() as tracker:
        p_stats = runner.run_parallel(records)
    print(f"     [Rows] processed={p_stats['total_records']:,} (input={total:,})")
    p_elapsed = p_stats["e2e_elapsed_s"] + (p_stats["ids_evaluations"] * IDS_LATENCY_S)
    mem_delta = getattr(tracker, "mem_delta_mb", None)
    results.append({
        "Algorithm": "Partitioned Bloom+Exact",
        "Records Processed": p_stats["total_records"],
        "Dedupe Rate (%)": p_stats["dedupe_rate_%"],
        "E2E Elapsed (s)": round(p_elapsed, 4),
        "Throughput (rec/s)": round(p_stats["total_records"] / max(0.0001, p_elapsed), 1),
        "Peak Memory (MB)": round(tracker.peak_mem_mb, 2),
        "Memory Delta (MB)": round(mem_delta, 2) if mem_delta is not None else ""
    })

    print("\n" + "="*105)
    print(" PIPELINE THROUGHPUT & EFFICIENCY RESULTS (CLEAN-ONLY, NO AGGREGATION) ")
    print("="*105)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("="*105 + "\n")
    
    counts = {
        "cleaned_rows_before_injection": clean_n,
        "rows_after_injection": pressured_n,
        "rows_processed": total
    }
    return results, counts
