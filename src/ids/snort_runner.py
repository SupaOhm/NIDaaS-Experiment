"""
snort_runner.py
---------------
Simulates the Snort Signature-based detection stage for experimental evaluation.
"""
import numpy as np
import pandas as pd

class SnortSignatureDetector:
    def __init__(self):
        # Academic port/protocol heuristic representing generic deterministic Snort rules
        self.malicious_ports = {21, 22, 23, 3389, 4444}
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Emulates a Snort deterministic signature pass. 
        Returns an array of 0 (Pass to LSTM) or 1 (Snort Blocked).
        """
        predictions = np.zeros(len(df), dtype=int)
        
        # Heuristic 1: Known bad ports
        port_col = 'Destination Port' if 'Destination Port' in df.columns else ' Destination Port'
        if port_col in df.columns:
            ports = df[port_col].values
            for i in range(len(ports)):
                if ports[i] in self.malicious_ports:
                    predictions[i] = 1
        
        # Heuristic 2: High volume packet streams (DDoS Rule Proxy)
        bwd_pkts = 'Total Backward Packets' if 'Total Backward Packets' in df.columns else ' Total Backward Packets'
        if bwd_pkts in df.columns:
            pkts = df[bwd_pkts].values
            for i in range(len(pkts)):
                if pkts[i] > 10000:
                    predictions[i] = 1

        return predictions
