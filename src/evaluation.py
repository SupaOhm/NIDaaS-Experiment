"""
evaluation.py
-------------
Evaluation helpers for IDS experiments.

Main responsibilities:
- Compute classification metrics
- Compute system-efficiency metrics such as latency and throughput
- Print experiment results in a readable format
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, elapsed_s: float) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    throughput = len(y_true) / elapsed_s if elapsed_s > 0 else 0.0
    avg_latency_ms = (elapsed_s / len(y_true)) * 1000.0 if len(y_true) > 0 else 0.0

    return {
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total_time_s": float(elapsed_s),
        "avg_latency_ms": float(avg_latency_ms),
        "throughput_eps": float(throughput),
    }


def print_metrics(title: str, metrics: dict):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:.6f}")
        else:
            print(f"  {k:<20} {v}")