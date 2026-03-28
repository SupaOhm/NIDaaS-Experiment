"""
signature_detector.py
---------------------
Implements rule-based signature detection for CIC-IDS2017-style flow features.

This version can automatically load the current best signature settings
from best_config.py. If best_config.py is unavailable, it falls back to
safe defaults.

You can still override any value manually, for example:
    SignatureConfig(high_flow_packets_s_thr=900_000.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

# Try loading saved best config
try:
    from best_config import get_best_experiment_config

    _BEST_SIGNATURE_CFG = get_best_experiment_config().get("signature_config", {})
except Exception:
    _BEST_SIGNATURE_CFG = {}


def _best(name: str, default):
    return _BEST_SIGNATURE_CFG.get(name, default)


@dataclass
class SignatureConfig:
    # Auto-load from best_config.py if available
    enable_high_rate_rule: bool = _best("enable_high_rate_rule", True)
    enable_syn_flood_rule: bool = _best("enable_syn_flood_rule", False)
    enable_udp_flood_rule: bool = _best("enable_udp_flood_rule", False)
    enable_icmp_flood_rule: bool = _best("enable_icmp_flood_rule", False)
    enable_scan_like_rule: bool = _best("enable_scan_like_rule", False)

    high_flow_packets_s_thr: float = _best("high_flow_packets_s_thr", 1_000_000.0)
    high_flow_bytes_s_thr: float = _best("high_flow_bytes_s_thr", 0.0)

    syn_count_thr: float = _best("syn_count_thr", 20.0)
    ack_count_max: float = _best("ack_count_max", 0.0)
    syn_flood_packets_s_thr: float = _best("syn_flood_packets_s_thr", 5_000.0)
    syn_flood_total_fwd_pkts_thr: float = _best("syn_flood_total_fwd_pkts_thr", 40.0)

    udp_packets_s_thr: float = _best("udp_packets_s_thr", 20_000.0)
    udp_total_fwd_pkts_thr: float = _best("udp_total_fwd_pkts_thr", 80.0)
    udp_total_fwd_bytes_thr: float = _best("udp_total_fwd_bytes_thr", 8_000.0)

    icmp_packets_s_thr: float = _best("icmp_packets_s_thr", 10_000.0)
    icmp_total_fwd_pkts_thr: float = _best("icmp_total_fwd_pkts_thr", 40.0)

    scan_duration_max: float = _best("scan_duration_max", 50_000.0)
    scan_total_pkts_max: float = _best("scan_total_pkts_max", 3.0)
    scan_total_bytes_max: float = _best("scan_total_bytes_max", 300.0)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Safely get a numeric column; return default if missing."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _protocol_mask(df: pd.DataFrame, protocol_name: str) -> pd.Series:
    if "Protocol" not in df.columns:
        return pd.Series(False, index=df.index)

    s = df["Protocol"]
    s_num = pd.to_numeric(s, errors="coerce")

    if protocol_name.upper() == "ICMP":
        return s_num.eq(1) | s.astype(str).str.upper().eq("ICMP")
    if protocol_name.upper() == "TCP":
        return s_num.eq(6) | s.astype(str).str.upper().eq("TCP")
    if protocol_name.upper() == "UDP":
        return s_num.eq(17) | s.astype(str).str.upper().eq("UDP")

    return pd.Series(False, index=df.index)


def detect(
    df_features: pd.DataFrame,
    config: SignatureConfig | None = None,
    return_details: bool = False,
) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
    cfg = config or SignatureConfig()

    flow_packets_s = _num(df_features, "Flow Packets/s")
    flow_bytes_s = _num(df_features, "Flow Bytes/s")

    syn = _num(df_features, "SYN Flag Count")
    ack = _num(df_features, "ACK Flag Count")
    flow_duration = _num(df_features, "Flow Duration")

    total_fwd_pkts = _num(df_features, "Total Fwd Packets")
    total_bwd_pkts = _num(df_features, "Total Backward Packets")
    total_len_fwd = _num(df_features, "Total Length of Fwd Packets")
    total_len_bwd = _num(df_features, "Total Length of Bwd Packets")

    total_pkts = total_fwd_pkts + total_bwd_pkts
    total_bytes = total_len_fwd + total_len_bwd

    is_tcp = _protocol_mask(df_features, "TCP")
    is_udp = _protocol_mask(df_features, "UDP")
    is_icmp = _protocol_mask(df_features, "ICMP")

    # Current best round: packets/s only
    rule_high_rate = pd.Series(False, index=df_features.index)
    if cfg.enable_high_rate_rule:
        rule_high_rate = (
            is_tcp
            & (flow_packets_s >= cfg.high_flow_packets_s_thr)
        )

    rule_syn_flood = pd.Series(False, index=df_features.index)
    if cfg.enable_syn_flood_rule:
        rule_syn_flood = (
            is_tcp
            & (syn >= cfg.syn_count_thr)
            & (ack <= cfg.ack_count_max)
            & (
                (flow_packets_s >= cfg.syn_flood_packets_s_thr)
                | (total_fwd_pkts >= cfg.syn_flood_total_fwd_pkts_thr)
            )
        )

    rule_udp_flood = pd.Series(False, index=df_features.index)
    if cfg.enable_udp_flood_rule:
        rule_udp_flood = (
            is_udp
            & (
                (
                    (flow_packets_s >= cfg.udp_packets_s_thr)
                    & (total_fwd_pkts >= cfg.udp_total_fwd_pkts_thr)
                )
                | (
                    (flow_packets_s >= cfg.udp_packets_s_thr)
                    & (total_len_fwd >= cfg.udp_total_fwd_bytes_thr)
                )
            )
        )

    rule_icmp_flood = pd.Series(False, index=df_features.index)
    if cfg.enable_icmp_flood_rule:
        rule_icmp_flood = (
            is_icmp
            & (
                (flow_packets_s >= cfg.icmp_packets_s_thr)
                | (total_fwd_pkts >= cfg.icmp_total_fwd_pkts_thr)
            )
        )

    rule_scan_like = pd.Series(False, index=df_features.index)
    if cfg.enable_scan_like_rule:
        rule_scan_like = (
            is_tcp
            & (syn >= 1)
            & (ack <= 0)
            & (flow_duration <= cfg.scan_duration_max)
            & (total_pkts <= cfg.scan_total_pkts_max)
            & (total_bytes <= cfg.scan_total_bytes_max)
        )

    details = pd.DataFrame(
        {
            "rule_high_rate": rule_high_rate.astype(int),
            "rule_syn_flood": rule_syn_flood.astype(int),
            "rule_udp_flood": rule_udp_flood.astype(int),
            "rule_icmp_flood": rule_icmp_flood.astype(int),
            "rule_scan_like": rule_scan_like.astype(int),
        },
        index=df_features.index,
    )

    y_pred = details.any(axis=1).astype(int).to_numpy()

    if return_details:
        return y_pred, details
    return y_pred


def predict(
    df_features: pd.DataFrame,
    config: SignatureConfig | None = None,
    return_details: bool = False,
) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
    return detect(df_features, config=config, return_details=return_details)