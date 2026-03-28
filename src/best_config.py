"""
best_config.py
--------------
Stores the current best experiment configuration for the RF novelty + signature hybrid prototype.

Current best result:
- model: hybrid_signature_rf_novelty
- quantile: 0.95
- signature proxy: packets/s only
- rf_n_estimators: 100
- rf_max_depth: 10
"""

from __future__ import annotations


def get_best_experiment_config() -> dict:
    return {
        "experiment_name": "rf_novelty_hybrid_current_best",
        "best_result": {
            "model": "hybrid_signature_rf_novelty",
            "quantile": 0.95,
            "f1": 0.4476864738964949,
            "precision": 0.8995377108124304,
            "recall": 0.29799775006853785,
            "far": 0.06195064009502441,
        },
        "data": {
            "input_dir": "data/",
            "max_files": 8,
        },
        "rf_config": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "alpha": 0.7,
            "quantiles": [0.95],
            "random_state": 42,
        },
        "signature_config": {
            "enable_high_rate_rule": True,
            "enable_syn_flood_rule": False,
            "enable_udp_flood_rule": False,
            "enable_icmp_flood_rule": False,
            "enable_scan_like_rule": False,
            "high_flow_packets_s_thr": 1_000_000.0,
            "high_flow_bytes_s_thr": 0.0,
            "syn_count_thr": 20.0,
            "ack_count_max": 0.0,
            "syn_flood_packets_s_thr": 5_000.0,
            "syn_flood_total_fwd_pkts_thr": 40.0,
            "udp_packets_s_thr": 20_000.0,
            "udp_total_fwd_pkts_thr": 80.0,
            "udp_total_fwd_bytes_thr": 8_000.0,
            "icmp_packets_s_thr": 10_000.0,
            "icmp_total_fwd_pkts_thr": 40.0,
            "scan_duration_max": 50_000.0,
            "scan_total_pkts_max": 3.0,
            "scan_total_bytes_max": 300.0,
        },
        "notes": {
            "signature_mode": "simulated Snort-like signature proxy on CIC-IDS2017 CSV",
            "best_tradeoff": "Hybrid improves F1 and recall over RF-only with a moderate FAR increase.",
            "timing_note": "Fast experiment mode precomputes RF scores, so total_time_s and throughput_eps should not be used as final speed metrics.",
        },
    }


BEST_CONFIG = get_best_experiment_config()
