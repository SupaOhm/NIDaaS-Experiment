import sys
import collections
from typing import Dict, Any
try:
    from pybloom_live import ScalableBloomFilter
except ImportError:
    print("pybloom_live missing. Exiting.")
    sys.exit(1)

from dedupe.base import BaseDeduplicator, _extract_scope_and_fingerprint


def _estimate_scoped_state_memory(scoped_state: dict) -> dict:
    """
    Estimate memory used by dedupe state (ScalableBloomFilter + exact caches).
    ScalableBloomFilter grows dynamically, so we estimate based on current state.
    """
    num_scopes = len(scoped_state)
    total_bytes = 0
    total_fps_cached = 0
    
    for state in scoped_state.values():
        # Bloom filter overhead
        total_bytes += sys.getsizeof(state.bloom)
        if hasattr(state.bloom, 'bit_array'):
            total_bytes += len(state.bloom.bit_array) // 8
        
        # Exact cache
        num_fps = len(state.exact_cache)
        total_fps_cached += num_fps
        bytes_per_fp = 32  # SHA-256 digest
        total_bytes += num_fps * bytes_per_fp
        
        # Exact queue pointers
        total_bytes += len(state.exact_queue) * 8
    
    return {
        'num_scopes': num_scopes,
        'bytes_blooms': total_bytes,
        'num_fps_in_cache': total_fps_cached,
        'total_estimate_mb': total_bytes / 1e6
    }


class _ScopedBloomExactState:
    __slots__ = ("bloom", "insertions_since_flush", "exact_cache", "exact_queue")

    def __init__(self, error_rate: float, initial_capacity: int = 1000):
        # ScalableBloomFilter grows lazily as items are added
        self.bloom = ScalableBloomFilter(initial_capacity=initial_capacity, error_rate=error_rate)
        self.insertions_since_flush = 0
        self.exact_cache = {}
        self.exact_queue = collections.deque()

class BloomExactDeduplicator(BaseDeduplicator):
    """
    Proposed 2-Stage Pipeline (ScalableBloomFilter + Exact Hash).
    Zero False Positives while maintaining O(1) Sliding-Window memory bounds.
    Per-scope Blooms use lazy growth to avoid over-allocation on small scopes.
    """
    def __init__(self, window_size: int = 100_000, error_rate: float = 0.01, initial_capacity: int = 1000, **kwargs):
        super().__init__(window_size, **kwargs)
        self.error_rate = error_rate
        self.initial_capacity = initial_capacity
        self._scoped_state = {}

    def _get_or_create_state(self, scope_key):
        state = self._scoped_state.get(scope_key)
        if state is None:
            state = _ScopedBloomExactState(self.error_rate, self.initial_capacity)
            self._scoped_state[scope_key] = state
        return state

    def get_diagnostic_info(self) -> str:
        """
        Return diagnostic information about Bloom instances and memory usage.
        """
        mem_est = _estimate_scoped_state_memory(self._scoped_state)
        return (f"[BloomExact Diagnostic] Scopes: {mem_est['num_scopes']}, "
                f"FPs cached: {mem_est['num_fps_in_cache']}, "
                f"Est. state: {mem_est['total_estimate_mb']:.2f}MB (lazy-growth ScalableBloomFilter)")

    def process_record(self, record: dict | tuple) -> bool:
        self.total_records += 1

        # Fast path: Experiment 2 precomputes compact tuple records.
        if isinstance(record, tuple) and len(record) == 2:
            scope_key, fp = record
        else:
            scope_key, fp = _extract_scope_and_fingerprint(record)

        state = self._scoped_state.get(scope_key)
        if state is None:
            state = _ScopedBloomExactState(self.error_rate, self.initial_capacity)
            self._scoped_state[scope_key] = state

        exact_queue = state.exact_queue
        exact_cache = state.exact_cache

        # 1) Stage-1 Bloom lookup (without pre-evicting state)
        bloom = state.bloom

        # 2) Two-stage dedupe: Bloom screening then exact verification
        if fp in bloom:
            if fp in exact_cache:
                self.duplicates += 1
                return True
            self.false_positives += 1

        # Evict only when accepting a new record to preserve true N-record lookback.
        if len(exact_queue) >= self.window_size:
            old_fp = exact_queue.popleft()
            exact_cache.pop(old_fp, None)

        # Rotate bloom on insertion boundaries, not before evaluating current record.
        if state.insertions_since_flush >= self.window_size:
            state.bloom = ScalableBloomFilter(initial_capacity=self.initial_capacity, error_rate=self.error_rate)
            state.insertions_since_flush = 0
            bloom = state.bloom

        bloom.add(fp)
        state.insertions_since_flush += 1

        exact_cache[fp] = True
        exact_queue.append(fp)
        self.ids_evaluations += 1
        return False
