import sys
from typing import Dict, Any
try:
    from pybloom_live import ScalableBloomFilter
except ImportError:
    print("pybloom_live missing. Exiting.")
    sys.exit(1)

from dedupe.base import BaseDeduplicator, _extract_scope_and_fingerprint


def _estimate_bloom_only_memory(scoped_state: dict) -> dict:
    """Estimate memory used by Bloom filters (no exact cache).
    ScalableBloomFilter grows dynamically, so we estimate based on current filters."""
    num_scopes = len(scoped_state)
    total_bytes = 0
    for state in scoped_state.values():
        # ScalableBloomFilter has a Python object overhead + internal bitarray
        total_bytes += sys.getsizeof(state.bloom)
        # Rough estimate: current filter chains grow, estimate via capacity if available
        if hasattr(state.bloom, 'bit_array'):
            total_bytes += len(state.bloom.bit_array) // 8
    
    return {
        'num_scopes': num_scopes,
        'bytes_blooms': total_bytes,
        'total_estimate_mb': total_bytes / 1e6
    }


class _ScopedBloomState:
    __slots__ = ("bloom", "insertions_since_flush")

    def __init__(self, error_rate: float):
        # ScalableBloomFilter grows lazily as items are added
        self.bloom = ScalableBloomFilter(initial_capacity=1000, error_rate=error_rate)
        self.insertions_since_flush = 0

class BloomOnlyDeduplicator(BaseDeduplicator):
    """
    Approximate Bloom-only deduplicator with per-scope scalable filters.
    
    Uses ScalableBloomFilter for lazy memory growth per (tenant, log_type) scope.
    TTL implemented via periodic flush (resetting filter at window boundaries).
    """
    def __init__(self, window_size: int = 100_000, error_rate: float = 0.01, **kwargs):
        super().__init__(window_size, **kwargs)
        self.error_rate = error_rate
        self._scoped_state = {}

    def _get_or_create_state(self, scope_key):
        state = self._scoped_state.get(scope_key)
        if state is None:
            state = _ScopedBloomState(self.error_rate)
            self._scoped_state[scope_key] = state
        return state

    def get_diagnostic_info(self) -> str:
        """Return diagnostic information about Bloom instances and memory usage."""
        mem_est = _estimate_bloom_only_memory(self._scoped_state)
        return (f"[BloomOnly Diagnostic] Scopes: {mem_est['num_scopes']}, "
                f"Est. state: {mem_est['total_estimate_mb']:.2f}MB (lazy-growth ScalableBloomFilter)")

    def process_record(self, record: dict | tuple) -> bool:
        self.total_records += 1
        scope_key, fp = _extract_scope_and_fingerprint(record)
        state = self._get_or_create_state(scope_key)

        bloom = state.bloom
        if fp in bloom:
            # We don't have an exact cache to check, so we MUST assume it's a duplicate.
            # This causes False Positives (dropping unique packets permanently)
            self.duplicates += 1
            return True

        if state.insertions_since_flush >= self.window_size:
            # TTL flush: reset filter for next window
            state.bloom = ScalableBloomFilter(initial_capacity=1000, error_rate=self.error_rate)
            state.insertions_since_flush = 0
            bloom = state.bloom

        bloom.add(fp)
        state.insertions_since_flush += 1
        self.ids_evaluations += 1
        return False
