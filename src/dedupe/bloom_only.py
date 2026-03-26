import sys
try:
    from pybloom_live import BloomFilter
except ImportError:
    print("pybloom_live missing. Exiting.")
    sys.exit(1)

from dedupe.base import BaseDeduplicator, _get_flow_signature

class BloomOnlyDeduplicator(BaseDeduplicator):
    """
    Uses an approximate pre-filter.
    Because standard Bloom Filters cannot delete individual items, 
    the sliding window is implemented as a synchronous periodic flush.
    """
    def __init__(self, window_size: int = 100_000, error_rate: float = 0.01, **kwargs):
        super().__init__(window_size, **kwargs)
        self.error_rate = error_rate
        self.bloom = BloomFilter(capacity=max(self.window_size, 1000), error_rate=self.error_rate)
        self.insertions_since_flush = 0

    def _flush_if_needed(self):
        if self.insertions_since_flush >= self.window_size:
            self.bloom = BloomFilter(capacity=max(self.window_size, 1000), error_rate=self.error_rate)
            self.insertions_since_flush = 0

    def process_record(self, record: dict) -> bool:
        self.total_records += 1
        fp = _get_flow_signature(record)
        
        self._flush_if_needed()
        
        if fp in self.bloom:
            # We don't have an exact cache to check, so we MUST assume it's a duplicate.
            # This causes False Positives (dropping unique packets permanently)
            self.duplicates += 1
            return True
        else:
            self.bloom.add(fp)
            self.insertions_since_flush += 1
            self.ids_evaluations += 1
            return False
