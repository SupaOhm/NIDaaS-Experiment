import collections
from dedupe.base import BaseDeduplicator, _get_flow_signature

class HashExactDeduplicator(BaseDeduplicator):
    """
    Standard Hash Cache. 
    Strict O(1) sliding window eviction to prevent unbounded OOM at scale.
    """
    def __init__(self, window_size: int = 100_000, **kwargs):
        super().__init__(window_size, **kwargs)
        self.cache = {}
        self.queue = collections.deque()

    def process_record(self, record: dict) -> bool:
        self.total_records += 1
        fp = _get_flow_signature(record)
        
        # O(1) Eviction check
        if len(self.queue) >= self.window_size:
            oldest_fp = self.queue.popleft()
            if oldest_fp in self.cache:
                del self.cache[oldest_fp]
                
        # O(1) Dictionary Hash Lookup
        if fp in self.cache:
            self.duplicates += 1
            return True
        else:
            self.cache[fp] = True
            self.queue.append(fp)
            self.ids_evaluations += 1
            return False
