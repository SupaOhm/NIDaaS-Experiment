"""
test_dedupe.py
--------------
Unit tests for the two-stage Bloom + exact hash deduplication module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from dedupe import Deduplicator, canonicalize, fingerprint


# ── Helper fixtures ────────────────────────────────────────────────────────────

def make_record(src_ip="1.1.1.1", dst_ip="2.2.2.2",
                src_port=1111, dst_port=80, proto=6,
                fwd_pkts=10, bwd_pkts=5, fwd_bytes=500, bwd_bytes=200):
    return {
        "Source IP": src_ip, "Destination IP": dst_ip,
        "Source Port": src_port, "Destination Port": dst_port,
        "Protocol": proto,
        "Total Fwd Packets": fwd_pkts, "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_bytes, "Total Length of Bwd Packets": bwd_bytes,
    }


# ── Canonicalisation tests ─────────────────────────────────────────────────────

def test_canonicalize_deterministic():
    """Same record → same canonical string every time."""
    r = make_record()
    assert canonicalize(r) == canonicalize(r)


def test_canonicalize_different_records():
    """Different records → different canonical strings."""
    r1 = make_record(src_ip="1.1.1.1")
    r2 = make_record(src_ip="1.1.1.2")
    assert canonicalize(r1) != canonicalize(r2)


def test_fingerprint_length():
    """SHA-256 fingerprint must be 64 hex characters."""
    r = make_record()
    fp = fingerprint(canonicalize(r))
    assert len(fp) == 64


# ── Deduplicator tests ─────────────────────────────────────────────────────────

def test_first_occurrence_is_not_duplicate():
    """A record seen for the first time must not be flagged as duplicate."""
    deduper = Deduplicator(capacity=100)
    r = make_record()
    assert deduper.is_duplicate(r) is False


def test_second_occurrence_is_duplicate():
    """The same record seen a second time must be flagged as duplicate."""
    deduper = Deduplicator(capacity=100)
    r = make_record()
    deduper.is_duplicate(r)   # first time
    assert deduper.is_duplicate(r) is True   # second time


def test_unique_records_all_pass():
    """All distinct records must pass through (none flagged as duplicates)."""
    deduper = Deduplicator(capacity=200)
    records = [make_record(src_port=1000 + i) for i in range(100)]
    results = [deduper.is_duplicate(r) for r in records]
    assert all(r is False for r in results)


def test_duplicate_batch():
    """process_batch correctly separates unique and duplicate records."""
    deduper = Deduplicator(capacity=200)
    unique_records = [make_record(src_port=2000 + i) for i in range(5)]
    duplicate_records = [unique_records[0], unique_records[2]]
    all_records = unique_records + duplicate_records

    passed, duped = deduper.process_batch(all_records)
    assert len(passed) == 5
    assert len(duped) == 2


def test_stats_counting():
    """Summary stats must accurately count total, passed, and deduplicated."""
    deduper = Deduplicator(capacity=200)
    r = make_record()
    deduper.is_duplicate(r)
    deduper.is_duplicate(r)  # duplicate
    deduper.is_duplicate(make_record(src_port=9999))  # unique

    s = deduper.summary()
    assert s["total_records"] == 3
    assert s["passed"] == 2
    assert s["deduplicated"] == 1


def test_bloom_false_positive_rate():
    """
    Bloom filter FP rate should stay within a reasonable bound for a small test set.
    Note: This is a statistical test — it should pass reliably at the error_rate=0.01 setting.
    """
    n = 500
    deduper = Deduplicator(capacity=n, error_rate=0.01)
    # Insert n unique records
    for i in range(n):
        deduper.is_duplicate(make_record(src_port=10000 + i))
    s = deduper.summary()
    # False-positive rate should be low (well under 5%)
    assert s["fp_rate_%"] < 5.0, f"FP rate too high: {s['fp_rate_%']}%"
