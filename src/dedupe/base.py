"""
base.py
-------
Enforces the strict, fair API for all Deduplication algorithms.
"""

import hashlib
from typing import Dict, Any, Iterable, Tuple

_TENANT_KEYS = ("tenant_id", "Tenant ID", "Source IP", "Src IP", "src_ip")
_LOG_TYPE_KEYS = ("log_type", "Log Type", "Protocol", "protocol")


def _get_flow_signature(r: dict | tuple) -> bytes:
    """
    Canonical fingerprint for dedupe.
    Returns bytes to reduce memory and comparison overhead versus hex strings.
    """
    if isinstance(r, tuple) and len(r) == 2:
        return r[1]

    precomputed = r.get("_dedupe_fp")
    if precomputed is not None:
        return precomputed

    src = str(r.get("Source IP", ""))
    dst = str(r.get("Destination IP", ""))
    fwd_len = str(r.get("Total Length of Fwd Packets", ""))
    payload = f"{src}|{dst}|{fwd_len}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _get_scope_key(r: dict | tuple) -> Tuple[str, str]:
    """
    Tenant-scoped and log-type-scoped namespace for dedupe state.
    """
    if isinstance(r, tuple) and len(r) == 2:
        return r[0]

    precomputed = r.get("_dedupe_scope")
    if precomputed is not None:
        return precomputed

    tenant = ""
    for key in _TENANT_KEYS:
        if key in r:
            val = r.get(key)
            tenant = "" if val is None else str(val)
            break

    log_type = ""
    for key in _LOG_TYPE_KEYS:
        if key in r:
            val = r.get(key)
            log_type = "" if val is None else str(val)
            break

    return tenant, log_type


def _extract_scope_and_fingerprint(r: dict | tuple) -> Tuple[Tuple[str, str], bytes]:
    return _get_scope_key(r), _get_flow_signature(r)

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

    def process_records(self, records: Iterable[dict | tuple]) -> None:
        for record in records:
            self.process_record(record)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "duplicates_dropped": self.duplicates,
            "false_positives": self.false_positives,
            "ids_evaluations": self.ids_evaluations,
            "dedupe_rate_%": round(100 * (self.duplicates / max(1, self.total_records)), 2)
        }
