"""
dedupe.py
---------
Two-stage deduplication engine from the NIDaaS paper.

Stage 1 — Bloom Filter (fast approximate pre-filter):
  Checks whether a fingerprint has been seen before.
  May produce false positives (says "duplicate" when not), but never misses a real duplicate.
  This makes it a safe pre-filter: false positives are caught by Stage 2.

Stage 2 — Exact Hash Cache (SHA-256):
  Only runs if Stage 1 returns "possibly seen".
  Provides ground-truth duplicate detection with zero false positives.

Flow:
  record → canonicalize → fingerprint
         → Bloom check: not seen → pass through (new record)
         → Bloom check: possibly seen → exact cache check
                          → exact match → deduplicate
                          → no exact match → pass through (false positive)

This two-stage design is the paper's efficiency contribution:
  - Bloom filter is O(k) hash computations, very fast, memory-efficient
  - Exact cache only checked for a small fraction of records
  - Together they reduce redundant detection workload
"""

import hashlib
import json
import time
from typing import Dict, List, Tuple

try:
    from pybloom_live import BloomFilter
    _BLOOM_AVAILABLE = True
except ImportError:
    _BLOOM_AVAILABLE = False


def canonicalize(record: dict, log_type: str = "flow") -> str:
    """
    Standardizes a record into a canonical string based on the 
    NIDSaaS paper's Section III-B-4 definitions.
    """
    # Use lowercase for consistency
    lt = log_type.lower()
    
    # Common fields
    tid = str(record.get("tenant_id", "default_tenant"))
    
    if lt == "firewall":
        # Paper: tenant_id, event_time_utc, src_ip, dst_ip, src_port, dst_port, protocol, action
        fields = ["event_time_utc", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "action"]
    elif lt == "dns":
        # Paper: tenant_id, event_time_utc, src_ip, query_name, query_type, response_code
        fields = ["event_time_utc", "src_ip", "query_name", "query_type", "response_code"]
    elif lt == "auth":
        # Paper: tenant_id, event_time_utc, username, src_ip, action, result
        fields = ["event_time_utc", "username", "src_ip", "action", "result"]
    else:  # Default to IPFIX/Flow
        # Paper: tenant_id, flow_start_utc, flow_end_utc, src_ip, dst_ip, packet_count, byte_count
        # Note: mapping CIC-IDS2017 keys to Paper naming
        fields = ["Flow Duration", "Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol"]
    
    # Build canonical string
    vals = [tid, lt] + [str(record.get(f, "")) for f in fields]
    return "|".join(vals)

def fingerprint(canonical_str: str) -> str:
    """Computes a deterministic SHA-256 fingerprint (H(C(e')) in the paper)."""
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class Deduplicator:
    """
    Two-stage Bloom filter + exact hash cache deduplicator.

    Parameters
    ----------
    capacity     : expected number of unique items (Bloom filter sizing param)
    error_rate   : target false-positive rate for the Bloom filter (e.g. 0.01 = 1%)

    Attributes
    ----------
    stats : dict tracking pass/deduplicated counts and timing
    """

    def __init__(self, capacity: int = 100_000, error_rate: float = 0.01):
        if not _BLOOM_AVAILABLE:
            raise ImportError(
                "pybloom-live is required for Deduplicator. "
                "Install it with: pip install pybloom-live"
            )
        self.bloom = BloomFilter(capacity=capacity, error_rate=error_rate)
        self.exact_cache: Dict[str, bool] = {}   # sha256 → True
        self.stats = {
            "total":         0,
            "passed":        0,
            "deduplicated":  0,
            "bloom_hits":    0,      # how many times bloom said "seen"
            "false_positives": 0,    # bloom said "seen" but exact cache disagreed
            "total_time_ns": 0.0,
        }

    def is_duplicate(self, record: dict) -> bool:
        """
        Check whether this record is a duplicate.

        Returns True if duplicate (should be skipped), False if new (should be processed).
        Side effect: updates internal Bloom filter and exact cache if new.
        """
        t0 = time.perf_counter_ns()
        self.stats["total"] += 1

        canon = canonicalize(record)
        fp    = fingerprint(canon)

        # Stage 1: Bloom filter
        if fp in self.bloom:
            # Possibly seen before — verify with exact cache
            self.stats["bloom_hits"] += 1

            if fp in self.exact_cache:
                # Confirmed duplicate
                self.stats["deduplicated"] += 1
                self.stats["total_time_ns"] += time.perf_counter_ns() - t0
                return True
            else:
                # Bloom false positive — treat as new
                self.stats["false_positives"] += 1

        # Not a duplicate — register it
        self.bloom.add(fp)
        self.exact_cache[fp] = True
        self.stats["passed"] += 1
        self.stats["total_time_ns"] += time.perf_counter_ns() - t0
        return False

    def process_batch(self, records: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Filter a batch of records, separating unique from duplicate records.

        Returns
        -------
        (unique_records, duplicate_records)
        """
        unique = []
        dupes  = []
        for r in records:
            if self.is_duplicate(r):
                dupes.append(r)
            else:
                unique.append(r)
        return unique, dupes

    def summary(self) -> dict:
        """Return a readable performance summary dict."""
        total = self.stats["total"]
        total_ns = self.stats["total_time_ns"]
        avg_latency_us = (total_ns / total / 1000) if total > 0 else 0.0
        throughput = (total / (total_ns / 1e9)) if total_ns > 0 else 0.0
        return {
            "total_records":    total,
            "passed":           self.stats["passed"],
            "deduplicated":     self.stats["deduplicated"],
            "dedup_rate_%":     round(100 * self.stats["deduplicated"] / total, 2) if total else 0,
            "bloom_hits":       self.stats["bloom_hits"],
            "false_positives":  self.stats["false_positives"],
            "fp_rate_%":        round(100 * self.stats["false_positives"] / max(self.stats["bloom_hits"], 1), 2),
            "avg_latency_us":   round(avg_latency_us, 3),
            "throughput_rec_s": round(throughput, 1),
        }
