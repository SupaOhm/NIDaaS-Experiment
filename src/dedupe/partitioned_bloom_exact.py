"""
partitioned_bloom_exact.py
--------------------------
Simulates a distributed environment (Kafka -> Spark executors) where 
multi-tenant deduplication is routed via `hash(tenant_id) % N`.
Because there is NO shared state between workers, Python GIL is sidestepped
and operations perform completely in parallel.
"""

import time
import multiprocessing
from typing import Dict, List, Any

from dedupe.bloom_exact import BloomExactDeduplicator

def parallel_worker(worker_id: int, records: List[dict], window_size: int, error_rate: float, initial_capacity: int = 1000) -> Dict[str, Any]:
    deduper = BloomExactDeduplicator(window_size=window_size, error_rate=error_rate, initial_capacity=initial_capacity)
    
    t0 = time.perf_counter()
    for r in records:
        deduper.process_record(r)
    elapsed_s = time.perf_counter() - t0
    
    stats = deduper.get_stats()
    stats["worker_id"] = worker_id
    stats["elapsed_s"] = elapsed_s
    
    return stats


class PartitionedDeduplicatorRunner:
    """Orchestrator for offline multiprocessing benchmark."""
    def __init__(self, num_partitions=4, window_size=100_000, error_rate=0.01, initial_capacity=1000):
        self.num_partitions = num_partitions
        self.window_size = window_size
        self.error_rate = error_rate
        self.initial_capacity = initial_capacity
        self._mp_ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else multiprocessing.get_context()
        
    def partition_data(self, records: List[dict | tuple]) -> Dict[int, List[dict | tuple]]:
        partitions = [[] for _ in range(self.num_partitions)]
        n = self.num_partitions
        for r in records:
            if isinstance(r, tuple) and len(r) == 2:
                tid = r[0][0]
            else:
                tid = str(r.get("Source IP", "default_tenant"))
            # Use native hash for fast, in-run stable partition assignment.
            pid = hash(tid) % n
            partitions[pid].append(r)
        return {i: partitions[i] for i in range(n)}

    def run_parallel(self, records: List[dict]) -> Dict[str, Any]:
        partitions = self.partition_data(records)
        
        args_list = []
        for pid in range(self.num_partitions):
             args_list.append((pid, partitions[pid], self.window_size, self.error_rate, self.initial_capacity))
             
        t_start = time.perf_counter()
        with self._mp_ctx.Pool(processes=self.num_partitions) as pool:
             results = pool.starmap(parallel_worker, args_list)
        t_end = time.perf_counter()
        
        aggregated = {
            "total_records": sum(r["total_records"] for r in results),
            "duplicates_dropped": sum(r["duplicates_dropped"] for r in results),
            "false_positives": sum(r["false_positives"] for r in results),
            "ids_evaluations": sum(r["ids_evaluations"] for r in results),
            "max_worker_elapsed_s": max(r["elapsed_s"] for r in results) if results else 0,
            "e2e_elapsed_s": t_end - t_start
        }
        aggregated["dedupe_rate_%"] = round(100 * (aggregated["duplicates_dropped"] / max(1, aggregated["total_records"])), 2)
        return aggregated
