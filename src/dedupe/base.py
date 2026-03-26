"""
base.py
-------
Enforces the strict, fair API for all Deduplication algorithms.
"""

import hashlib
from typing import Dict, Any

def _get_flow_signature(r: dict) -> str:
    """Canonical 5-tuple payload signature proxy."""
    sig = f"{r.get('Source IP','')}-{r.get('Destination IP','')}-{r.get('Total Length of Fwd Packets','')}"
    return hashlib.sha256(sig.encode('utf-8')).hexdigest()

class BaseDeduplicator:
    def __init__(self, window_size: int = 100_000, **kwargs):
        """
        window_size: The capacity of the sliding temporal window. 
        Using record-counts instead of physical wall-clock time ensures 
        academic fairness during offline batch simulation loops.
        """
        self.window_size = window_size
        self.total_records = 0
        self.duplicates = 0
        self.false_positives = 0
        self.ids_evaluations = 0

    def process_record(self, record: dict) -> bool:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "duplicates_dropped": self.duplicates,
            "false_positives": self.false_positives,
            "ids_evaluations": self.ids_evaluations,
            "dedupe_rate_%": round(100 * (self.duplicates / max(1, self.total_records)), 2)
        }
