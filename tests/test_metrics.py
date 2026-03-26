"""
test_metrics.py
---------------
Unit tests for the metrics.py module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from metrics import classification_report_dict, efficiency_report


# ── classification_report_dict tests ──────────────────────────────────────────

def test_perfect_predictions():
    """Perfect predictions → accuracy=1.0, F1=1.0, FAR=0.0."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    result = classification_report_dict(y_true, y_pred, "TestModel")
    assert result["accuracy"]  == 1.0
    assert result["f1_score"]  == 1.0
    assert result["far"]       == 0.0
    assert result["model"]     == "TestModel"


def test_all_wrong_predictions():
    """All wrong predictions → accuracy=0.0, recall=0.0."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    result = classification_report_dict(y_true, y_pred)
    assert result["accuracy"] == 0.0
    assert result["recall"]   == 0.0


def test_far_calculation():
    """
    FAR = FP / (FP + TN).
    y_true = [0, 0, 1], y_pred = [1, 0, 1]
    TN=1, FP=1, TP=1, FN=0 → FAR = 1/(1+1) = 0.5
    """
    y_true = np.array([0, 0, 1])
    y_pred = np.array([1, 0, 1])
    result = classification_report_dict(y_true, y_pred)
    assert result["far"] == pytest.approx(0.5, rel=1e-3)


def test_result_keys():
    """Result dict must contain all required keys."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    result = classification_report_dict(y_true, y_pred, "AModel")
    expected_keys = {"model", "accuracy", "precision", "recall", "f1_score", "far"}
    assert expected_keys.issubset(result.keys())


def test_values_are_rounded():
    """All numeric values must be floats (rounded)."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    result = classification_report_dict(y_true, y_pred)
    for k in ("accuracy", "precision", "recall", "f1_score", "far"):
        assert isinstance(result[k], float), f"{k} is not a float"


# ── efficiency_report tests ────────────────────────────────────────────────────

def test_efficiency_throughput():
    """Throughput = total_records / elapsed_seconds."""
    r = efficiency_report("TestCond", total_records=1000, elapsed_seconds=2.0, memory_delta_mb=10.0)
    assert r["throughput_rec_s"] == pytest.approx(500.0, rel=1e-3)


def test_efficiency_latency():
    """Mean latency = elapsed / records * 1e6 microseconds."""
    r = efficiency_report("TestCond", total_records=1000, elapsed_seconds=1.0, memory_delta_mb=5.0)
    assert r["avg_latency_us"] == pytest.approx(1000.0, rel=1e-3)


def test_efficiency_non_negative():
    """Throughput and latency must be non-negative."""
    r = efficiency_report("TestCond", total_records=100, elapsed_seconds=0.5, memory_delta_mb=2.0)
    assert r["throughput_rec_s"] >= 0
    assert r["avg_latency_us"]   >= 0


def test_efficiency_with_dedup_stats():
    """Dedup stats are correctly merged into the efficiency report."""
    dedup_stats = {"dedup_rate_%": 30.0, "fp_rate_%": 0.5}
    r = efficiency_report("BloomExact", 500, 0.1, 3.0, dedup_stats=dedup_stats)
    assert r["dedup_rate_%"]    == 30.0
    assert r["bloom_fp_rate_%"] == 0.5
