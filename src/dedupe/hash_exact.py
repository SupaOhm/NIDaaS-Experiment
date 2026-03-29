"""
hash_exact.py
-------------
Exact-only dedupe baselines for Experiment 2.

Default baseline (HashExactDeduplicator):
- Practical exact cache using hashed membership (Python set)
- Full-cache, no-TTL semantics
- Represents a realistic exact-only competitor for Bloom-based methods

Optional supplementary baseline (HashArraySearchWorstCaseDeduplicator):
- Pathological O(n) linear array search
- Full-cache, no-TTL semantics
- Intended only for worst-case/debug comparisons, not default reporting
"""

from dedupe.base import BaseDeduplicator, _extract_scope_and_fingerprint


class HashExactDeduplicator(BaseDeduplicator):
    """
    Practical exact-only baseline.

    - No scoping (all records share one global cache)
    - No window/TTL eviction (full-cache mode: window_size=None)
    - O(1) average membership via Python set
    """

    def __init__(self, window_size: int = None, **kwargs):
        # Full-cache semantics are intentional for this exact-only comparator.
        super().__init__(window_size=None, **kwargs)
        self.seen_fingerprints = set()

    def process_record(self, record: dict | tuple) -> bool:
        self.total_records += 1
        _, fp = _extract_scope_and_fingerprint(record)

        if fp in self.seen_fingerprints:
            self.duplicates += 1
            return True

        self.seen_fingerprints.add(fp)
        self.ids_evaluations += 1
        return False


class HashArraySearchWorstCaseDeduplicator(BaseDeduplicator):
    """
    Optional pathological exact-only baseline.

    - No scoping
    - No TTL/window eviction
    - O(n) linear membership per record via list scan
    """

    def __init__(self, window_size: int = None, **kwargs):
        super().__init__(window_size=None, **kwargs)
        self.seen_fingerprints = []

    def process_record(self, record: dict | tuple) -> bool:
        self.total_records += 1
        _, fp = _extract_scope_and_fingerprint(record)

        for cached_fp in self.seen_fingerprints:
            if cached_fp == fp:
                self.duplicates += 1
                return True

        self.seen_fingerprints.append(fp)
        self.ids_evaluations += len(self.seen_fingerprints)
        return False
