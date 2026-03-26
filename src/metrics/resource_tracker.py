"""
resource_tracker.py
-------------------
Safely benchmarks algorithm throughput and isolated memory footprint 
via Python's built-in tracemalloc, avoiding OS paging distortions.
"""
import time
import tracemalloc

class BenchmarkTracker:
    def __init__(self):
        self.t_start = 0
        self.elapsed_s = 0.0
        self.peak_mem_mb = 0.0

    def __enter__(self):
        import gc
        gc.collect()
        tracemalloc.start()
        self.t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_s = time.perf_counter() - self.t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mem_mb = peak / 1e6
