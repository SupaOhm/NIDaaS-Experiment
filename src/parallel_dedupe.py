"""
parallel_dedupe.py
------------------
Implements the multi-worker parallel deduplication simulation.
Models the Kafka-Partition -> Spark-Executor relationship.
"""

import time
import hashlib
import multiprocessing
from typing import Dict, List, Any

# Must import inside or before worker to avoid serialization issues
try:
    from pybloom_live import BloomFilter
    _BLOOM_AVAILABLE = True
except ImportError:
    import sys
    print("pybloom_live missing. Exiting.")
    sys.exit(1)

from dedupe import canonicalize, fingerprint

# Import IDS simulation timing constant
from experiment_efficiency import SIMULATED_IDS_LATENCY_US

def simulate_ids_processing():
    time.sleep(SIMULATED_IDS_LATENCY_US / 1_000_000)

def worker_process(worker_id: int, records: List[dict], capacity: int, error_rate: float, algo: str) -> Dict[str, Any]:
    """
    Simulates a standalone stream-processing executor executing deduplication.
    Supported algorithms for benchmark: 'bloom_exact', 'hash_only'
    """
    import gc
    gc.collect()

    stats = {
        "worker_id": worker_id,
        "total": 0,
        "deduped": 0,
        "bloom_hits": 0,
        "ids_runs": 0,
        "elapsed_s": 0.0
    }
    
    t0 = time.perf_counter()
    
    if algo == "bloom_exact":
        bloom = BloomFilter(capacity=max(capacity, 1000), error_rate=error_rate)
        exact_cache = {}
        for r in records:
            stats["total"] += 1
            fp = fingerprint(canonicalize(r))
            is_duplicate = False
            if fp in bloom:
                stats["bloom_hits"] += 1
                if fp in exact_cache:
                    stats["deduped"] += 1
                    is_duplicate = True
            
            if not is_duplicate:
                bloom.add(fp)
                exact_cache[fp] = True
                stats["ids_runs"] += 1
                simulate_ids_processing()
                
        # Clean local refs
        del bloom, exact_cache

    elif algo == "hash_only":
        hash_cache = set()
        for r in records:
            stats["total"] += 1
            fp = fingerprint(canonicalize(r))
            if fp in hash_cache:
                stats["deduped"] += 1
            else:
                hash_cache.add(fp)
                stats["ids_runs"] += 1
                simulate_ids_processing()
        del hash_cache

    elif algo == "no_dedup":
        for r in records:
            stats["total"] += 1
            stats["ids_runs"] += 1
            simulate_ids_processing()
            
    stats["elapsed_s"] = time.perf_counter() - t0
    
    del records
    gc.collect()
    
    return stats


class PartitionedDeduplicatorRunner:
    def __init__(self, num_partitions=4, capacity_per_partition=100_000, error_rate=0.01):
        self.num_partitions = num_partitions
        self.capacity_per_partition = capacity_per_partition
        self.error_rate = error_rate
        
    def partition_data(self, records: List[dict]) -> Dict[int, List[dict]]:
        """
        Simulates Kafka routing based on Tenant ID.
        Ensures identical flows hit the exact same worker to maintain state consistency.
        """
        partitions = {i: [] for i in range(self.num_partitions)}
        for r in records:
            tid = str(r.get("tenant_id", "default_tenant"))
            # Determine partition using MD5 to avoid Python hash() seed variations across runs, 
            # ensuring consistent deterministic assignment for performance measurement.
            hash_val = int(hashlib.md5(tid.encode('utf-8')).hexdigest(), 16)
            pid = hash_val % self.num_partitions
            partitions[pid].append(r)
        return partitions

    def run_parallel(self, records: List[dict], algo: str = "bloom_exact"):
        partitions = self.partition_data(records)
        
        # Package arguments for Pool maps (each element is for one worker)
        args_list = []
        for pid in range(self.num_partitions):
            args_list.append((
                pid, 
                partitions[pid], 
                self.capacity_per_partition, 
                self.error_rate,
                algo
            ))
            
        t_start = time.perf_counter()
        
        with multiprocessing.Pool(processes=self.num_partitions) as pool:
            results = pool.starmap(worker_process, args_list)
            
        t_end = time.perf_counter()
        
        aggregated = {
            "total_records": sum(r["total"] for r in results),
            "ids_runs": sum(r["ids_runs"] for r in results),
            "deduped": sum(r["deduped"] for r in results),
            "max_worker_elapsed_s": max(r["elapsed_s"] for r in results) if results else 0,
            "e2e_elapsed_s": t_end - t_start
        }
        
        return aggregated
