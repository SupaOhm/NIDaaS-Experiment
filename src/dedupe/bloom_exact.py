import sys
import collections
try:
    from pybloom_live import BloomFilter
except ImportError:
    print("pybloom_live missing. Exiting.")
    sys.exit(1)

from dedupe.base import BaseDeduplicator, _get_flow_signature

class BloomExactDeduplicator(BaseDeduplicator):
    """
    Proposed 2-Stage Pipeline (Bloom + Exact Hash).
    Zero False Positives while maintaining O(1) Sliding-Window memory bounds.
    """
    def __init__(self, window_size: int = 100_000, error_rate: float = 0.01, **kwargs):
        super().__init__(window_size, **kwargs)
        self.error_rate = error_rate
        
        self.bloom = BloomFilter(capacity=max(self.window_size, 1000), error_rate=self.error_rate)
        self.insertions_since_flush = 0
        
        self.exact_cache = {}
        self.exact_queue = collections.deque()

    def _flush_bloom_if_needed(self):
        if self.insertions_since_flush >= self.window_size:
            self.bloom = BloomFilter(capacity=max(self.window_size, 1000), error_rate=self.error_rate)
            self.insertions_since_flush = 0

    def process_record(self, record: dict) -> bool:
        self.total_records += 1
        fp = _get_flow_signature(record)
        
        # 1. Exact Cache O(1) Sliding Window Eviction
        if len(self.exact_queue) >= self.window_size:
            oldest_fp = self.exact_queue.popleft()
            if oldest_fp in self.exact_cache:
                del self.exact_cache[oldest_fp]
                
        # 2. Bloom Filter Synchronous Flush (Emulates Sliding Window)
        self._flush_bloom_if_needed()
        
        # 3. Two-Stage Deduplication Logic
        is_duplicate = False
        
        if fp in self.bloom:
            # Stage 2: Exact verification to prevent False Positives
            if fp in self.exact_cache:
                self.duplicates += 1
                is_duplicate = True
            else:
                self.false_positives += 1
                
        if not is_duplicate:
            self.bloom.add(fp)
            self.insertions_since_flush += 1
            
            self.exact_cache[fp] = True
            self.exact_queue.append(fp)
            
            self.ids_evaluations += 1
            
        return is_duplicate
