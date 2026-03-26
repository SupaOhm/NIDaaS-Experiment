"""
exp_dedupe_search.py
--------------------
Systematic grid search evaluating 2-stage deduplication topologies against the absolute benchmark 
of a monolithic Hashmap Exact deduplicator. Designed to discover theoretically optimal throughput scaling 
while profiling hardware contentions and memory bottlenecks.
"""

import time
import hashlib
import collections
import multiprocessing
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Any
from copy import deepcopy

try:
    from pybloom_live import BloomFilter
except ImportError:
    BloomFilter = None


def generate_synthetic_workload(num_records: int, dup_ratio: float, bursty: bool, tenants: int = 10) -> List[dict]:
    """
    Generates tailored datasets exploring theoretical failure modes of deduplication topologies.
    """
    num_unique = int(num_records * (1.0 - dup_ratio))
    base_records = []
    
    print(f"    [Synth] Generating {num_records:,} records (Dup Ratio: {dup_ratio:.2f}, Bursty: {bursty}, Tenants: {tenants})")
    
    # Generate unique flows
    for i in range(num_unique):
        base_records.append({
            "Source IP": f"10.0.0.{random.randint(1, tenants)}",
            "Destination IP": f"192.168.1.{i % 255}",
            "Total Length of Fwd Packets": str(random.randint(40, 1500)),
            "_id": str(i)
        })
        
    records = []
    if bursty:
        # High volume of rapid repeating identical packets sequentially
        pool = base_records.copy()
        while len(records) < num_records:
            if not pool: pool = base_records.copy()
            chosen = random.choice(pool)
            
            # Burst 1 to 50 identical records consecutively
            burst_len = min(random.randint(1, 50), num_records - len(records))
            records.extend([chosen] * burst_len)
    else:
        # Uniformly scattered duplicates
        records = base_records.copy()
        while len(records) < num_records:
            records.append(random.choice(base_records))
            
    # Do NOT strictly shuffle if representing time-series, but scattered duplicates imply random arrival.
    # We shuffle to prevent all uniques loading at the very start.
    random.shuffle(records)
    return records


def _get_signature(r: dict) -> str:
    sig = f"{r.get('Source IP','')}-{r.get('Destination IP','')}-{r.get('Total Length of Fwd Packets','')}"
    return hashlib.sha256(sig.encode('utf-8')).hexdigest()


# ==============================================================================
# WORKER EXECUTION CORE (Runs in isolated process)
# ==============================================================================
def parallel_topology_worker(worker_id: int, records: List[dict], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates a discrete executing node.
    If 'exact_map' is mapped to 'global', this worker only executes the Bloom array 
    and returns its output to the Master Coordinator over IPC.
    """
    import tracemalloc
    tracemalloc.start()
    t_start = time.perf_counter()
    
    # Topology State Flags
    use_bloom = config.get("use_bloom", False)
    b_cap = config.get("bloom_capacity", 10000)
    b_err = config.get("bloom_error_rate", 0.01)
    b_rot = config.get("bloom_rotation_window", 0) # 0 = no rotation
    
    use_exact = config.get("use_exact", False)
    exact_type = config.get("exact_type", "dict") # 'dict' or 'list'
    e_window = config.get("exact_window", 100_000)
    
    # Infrastructure
    bloom = BloomFilter(capacity=max(b_cap, 1000), error_rate=b_err) if use_bloom else None
    bloom_inserts = 0
    
    if exact_type == "list":
        exact_cache = []
    else:
        exact_cache = {}
        exact_queue = collections.deque()
        
    stats = {
        "worker_id": worker_id,
        "processed": 0,
        "bloom_drops": 0,
        "exact_drops": 0,
        "false_positives": 0,  # difficult to measure perfectly if we don't have the holistic picture, calculated roughly
        "passed_to_next_stage": 0
    }
    
    # Payload buffer if this node feeds to a Global Exact Cache Map
    forwarding_buffer = []
    is_global_exact = config.get("exact_scope", "local") == "global"
    
    for r in records:
        stats["processed"] += 1
        fp = _get_signature(r)
        
        # 1. Exact Cache O(1) Sliding Window Eviction
        if use_exact and not is_global_exact:
            if exact_type == "list":
                if len(exact_cache) >= e_window:
                    exact_cache.pop(0)
            else:
                if len(exact_queue) >= e_window:
                    old_fp = exact_queue.popleft()
                    if old_fp in exact_cache:
                        del exact_cache[old_fp]

        # 2. Bloom Filter Synchronous Flush (Emulates Sliding Window)
        if use_bloom and b_rot > 0 and bloom_inserts >= b_rot:
            bloom = BloomFilter(capacity=max(b_cap, 1000), error_rate=b_err)
            bloom_inserts = 0

        # 3. Two-Stage Deduplication Logic
        is_duplicate = False
        survived_bloom = True
        
        # Step A: Bloom Filter Evaluation
        if use_bloom:
            if fp in bloom:
                survived_bloom = False
                stats["bloom_drops"] += 1
                if not use_exact and not is_global_exact:
                    is_duplicate = True
                    
        # Step B: Exact Cache Verification (or Primary Filter)
        if not is_duplicate:
            if use_exact or is_global_exact:
                # Query Exact Cache if: 
                # (1) Bloom is disabled (Exact is primary), Or
                # (2) Bloom Hit (Exact is verification)
                needs_exact_query = (not use_bloom) or (not survived_bloom)
                
                if needs_exact_query:
                    if is_global_exact:
                        # Forward for IPC verification (completely handled remotely)
                        forwarding_buffer.append(r)
                        stats["passed_to_next_stage"] += 1
                        is_duplicate = True 
                    else:
                        in_exact = (fp in exact_cache) if exact_type == "dict" else (fp in exact_cache)
                        if in_exact:
                            stats["exact_drops"] += 1
                            is_duplicate = True
                        elif use_bloom:
                            stats["false_positives"] += 1

        # Step C: Accept New Flow
        if not is_duplicate:
            if use_bloom and survived_bloom:
                bloom.add(fp)
                bloom_inserts += 1
                
            if is_global_exact:
                # Bloom missed, so the record is new. Send over IPC to update remote state.
                forwarding_buffer.append(r)
                stats["passed_to_next_stage"] += 1
            elif use_exact:
                if exact_type == "list":
                    exact_cache.append(fp)
                else:
                    exact_cache[fp] = True
                    exact_queue.append(fp)
                stats["passed_to_next_stage"] += 1
                    
    elapsed_s = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    stats["elapsed_s"] = elapsed_s
    stats["peak_mem_mb"] = peak_mem / 1e6
    return stats, forwarding_buffer


# ==============================================================================
# COORDINATOR
# ==============================================================================
class GridSearchOrchestrator:
    def __init__(self):
        self.results = []
        
    def partition_records(self, records: List[dict], num_partitions: int, strategy: str) -> Dict[int, List[dict]]:
        partitions = {i: [] for i in range(num_partitions)}
        if num_partitions == 1:
            partitions[0] = records
            return partitions
            
        for r in records:
            if strategy == "tenant":
                routing_key = str(r.get("Source IP", ""))
            elif strategy == "hash":
                routing_key = str(_get_signature(r))
            else: # Random (Simulation of round-robin Kafka)
                routing_key = str(random.randint(1, 10000))
                
            pid = int(hashlib.md5(routing_key.encode()).hexdigest(), 16) % num_partitions
            partitions[pid].append(r)
        return partitions

    def evaluate_topology(self, name: str, config: Dict[str, Any], records: List[dict]):
        print(f"\nEvaluating Context: {name}")
        t_start = time.perf_counter()
        
        num_partitions = config.get("num_partitions", 1)
        part_strategy = config.get("partitioning_strategy", "tenant")
        
        # Distribute workload over designated strategy
        partitions = self.partition_records(records, num_partitions, part_strategy)
        
        args_list = []
        for pid in range(num_partitions):
            args_list.append((pid, partitions[pid], config))
            
        worker_stats = []
        forwarded_buffers = []
        
        # Parallel Execution Map
        with multiprocessing.Pool(processes=num_partitions) as pool:
            results = pool.starmap(parallel_topology_worker, args_list)
            
        for stats, f_buffer in results:
            worker_stats.append(stats)
            forwarded_buffers.extend(f_buffer)
            
        t_parallel = time.perf_counter() - t_start
        
        # Stage 3: Global Exact Emulation (if exact_scope == global)
        global_exact_drops = 0
        global_elapsed = 0.0
        
        if config.get("exact_scope", "local") == "global" and config.get("use_exact", False):
            # Evaluate the forwarded subset against a unified single-thread dictionary
            t_g = time.perf_counter()
            global_cache = {}
            global_queue = collections.deque()
            e_window = config.get("exact_window", 100_000)
            
            for r in forwarded_buffers:
                fp = _get_signature(r)
                if len(global_queue) >= e_window:
                    old_fp = global_queue.popleft()
                    if old_fp in global_cache: del global_cache[old_fp]
                    
                if fp in global_cache:
                    global_exact_drops += 1
                else:
                    global_cache[fp] = True
                    global_queue.append(fp)
                    
            global_elapsed = time.perf_counter() - t_g
            
        total_elapsed = t_parallel + global_elapsed
        
        # Aggregate Statistics
        t_records = sum(s["processed"] for s in worker_stats)
        if config.get("use_exact", False):
            t_drops = sum(s["exact_drops"] for s in worker_stats) + global_exact_drops
        elif config.get("use_bloom", False):
            t_drops = sum(s["bloom_drops"] for s in worker_stats)
        else:
            t_drops = 0
            
        max_worker_mem = max(s["peak_mem_mb"] for s in worker_stats)
        
        throughput = t_records / max(0.001, total_elapsed)
        sys_info = {
            "Topology_Name": name,
            "Total_Records": t_records,
            "Total_Dropped": t_drops,
            "Dup_Rate_%": round(100 * (t_drops / max(1, t_records)), 2),
            "E2E_Elapsed_s": round(total_elapsed, 4),
            "Throughput_rec_s": round(throughput, 1),
            "Peak_Mem_MB_per_Worker": round(max_worker_mem, 2),
            "Bloom_Config": f"p={config.get('num_partitions',1)} r={config.get('bloom_rotation_window',0)}",
            "Exact_Config": f"{config.get('exact_scope','local')} {config.get('exact_type','dict')}",
            "Score": throughput # Used for ranking
        }
        
        print(f"  -> Throughput: {sys_info['Throughput_rec_s']:,} r/s | E2E Mem: {sys_info['Peak_Mem_MB_per_Worker']} MB")
        self.results.append(sys_info)
        return sys_info


# ==============================================================================
# PIPELINE STAGES
# ==============================================================================
from data.loader import load_dataset

def run_dedupe_grid_search(data_path: str, smoke_limit: int = None):
    print("\n" + "="*80)
    print(" SYSTEMATIC GRID SEARCH: REAL WORLD CIC-IDS2017 EVALUATION ")
    print("="*80)
    
    orch = GridSearchOrchestrator()
    if not BloomFilter:
        print("[!] PyBloom Live not installed. Aborting.")
        return

    # Workload generation
    print("\n[exp_search] Ingesting Primary Main Dataset...")
    df = load_dataset(data_path)
    if smoke_limit is not None:
        df = df.head(smoke_limit)
        
    main_records = df.to_dict(orient="records")
    
    import os
    dataset_name = os.path.basename(data_path.rstrip('/'))
    if not dataset_name:
        dataset_name = data_path
        
    workloads = {
        f"Dataset ({dataset_name})": main_records
    }

    # Baseline: Monolithic Exact Hashmap (Dictionary)
    base_config_exact = {
        "num_partitions": 1, "use_bloom": False, "use_exact": True, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
    }
    
    # Baseline: Monolithic List (Naive Array)
    base_config_list = {
        "num_partitions": 1, "use_bloom": False, "use_exact": True, "exact_type": "list", "exact_scope": "local", "exact_window": 100_000
    }

    configs_to_test = [
        ("Exp2: No Dedupe", {
            "num_partitions": 1, "use_bloom": False, "use_exact": False, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
        }),
        ("Exp2: Bloom-Only", {
            "num_partitions": 1, "use_bloom": True, "bloom_capacity": 500000, "bloom_error_rate": 0.01,
            "use_exact": False, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
        }),
        ("Base: Exact Only (Exp2)", base_config_exact),
        ("Base: Global Array", base_config_list),
        
        ("Cfg A: Central B+E", {
            "num_partitions": 1, "use_bloom": True, "bloom_capacity": 500000, "bloom_error_rate": 0.01,
            "use_exact": True, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
        }),
        
        ("Cfg B: Part B+E (IPC)", {
            "num_partitions": 4, "partitioning_strategy": "hash", 
            "use_bloom": True, "bloom_capacity": 200000, "bloom_error_rate": 0.01,
            "use_exact": True, "exact_type": "dict", "exact_scope": "global", "exact_window": 100_000
        }),
        
        ("Cfg C: Part B+E (Tenant)", {
            "num_partitions": 4, "partitioning_strategy": "tenant", 
            "use_bloom": True, "bloom_capacity": 200000, "bloom_error_rate": 0.01,
            "use_exact": True, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
        }),
        
        ("Cfg D: Part B+E (Hash)", {
            "num_partitions": 4, "partitioning_strategy": "hash", 
            "use_bloom": True, "bloom_capacity": 200000, "bloom_error_rate": 0.01,
            "use_exact": True, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
        }),
        
        ("Cfg E: Rot Part B+E", {
            "num_partitions": 4, "partitioning_strategy": "hash", 
            "use_bloom": True, "bloom_capacity": 100000, "bloom_error_rate": 0.01, "bloom_rotation_window": 20000,
            "use_exact": True, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
        }),
        
        ("Cfg F: Part B+E (8C)", {
            "num_partitions": 8, "partitioning_strategy": "hash", 
            "use_bloom": True, "bloom_capacity": 100000, "bloom_error_rate": 0.01,
            "use_exact": True, "exact_type": "dict", "exact_scope": "local", "exact_window": 100_000
        })
    ]

    for wk_name, records in workloads.items():
        print(f"\n[{wk_name.upper()}] Running configuration matrix on {len(records):,} flows...")
        for c_name, cfg in configs_to_test:
            # Skip O(N) array evaluation on massive lists 
            # to prevent unbounded processing spans
            if "Array/List" in c_name and (smoke_limit is None or smoke_limit > 50000):
                print(f"Skipping {c_name} (Too slow for massive dataset)")
                continue
                
            orch.evaluate_topology(f"{wk_name} | {c_name}", cfg, records)

    df_results = pd.DataFrame(orch.results)
    df_results = df_results.sort_values(by="Score", ascending=False)
    
    # Save the output formally
    print("\n" + "="*80)
    print(" RANKED RESULTS - PHASE 1 & 2 CONCLUSIVE SWEEP ")
    print("="*80)
    print(df_results.drop(columns=["Score"]).to_string(index=False))
    
    # Write to local CSV for graphing/reporting
    import os
    os.makedirs("results/tables", exist_ok=True)
    df_results.to_csv("results/tables/dedupe_grid_search_ranking.csv", index=False)
    
    return df_results
