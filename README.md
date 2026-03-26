# NIDSaaS-DDA Architecture: PyTorch Experimental Prototype

This repository contains the refactored, strictly academic implementation investigating the throughput and detection scaling of a Cloud-based Intrusion Detection System as a Service (IDSaaS).

## Methodological Integrity
Following a severe audit, this implementation strictly guarantees:
1. **Zero Fake Sequencing**: The data pipeline groups connection flows strictly chronologically to simulate a gateway "macro-sensor", ensuring the LSTM sequence representations are temporally valid.
2. **Leak-Free Validation**: StandardScalers are fitted *exclusively* on the Train temporal bounds.
3. **Fair O(1) Sliding-Window Deduplication**: By enforcing a strict uniform window capacity ($N$ limit equivalent to TTL) across all data structures, naive unbounded Hash/Dict approaches are physically capped to mirror real-world OOM prevention policies.

## Repository Architecture

The workspace is deeply modularised to facilitate clean ablation studies and component isolation:
```text
NIDaaS-Experiment/
├── data/                       # Mount CIC-IDS2017 Raw CSV files here
├── src/
│   ├── data/                   # Strict Data pipeline & Chronological Loading
│   ├── dedupe/                 # Fair-Bounded TTL deduplication (O(1) constraints)
│   ├── ids/                    # Snort deterministic signatures + PyTorch LSTM Anomalies
│   ├── metrics/                # Hardware-isolated Tracemalloc & F1 scoring
│   ├── experiments/            # Isolated runtime environment drivers
│   └── main.py                 # Multi-experiment orchestrator
```

## Running the Experiments

By default, the experiments search for data inside `./data/`. For rapid debugging, append `--smoke` to any run to artificially constrain dataset evaluation windows to under ~10,000 records.

### 1. Hybrid Detection Evaluation
Proves the complementary nature of the `Snort` (Signature) array and the `LSTM` (Forecasting Auto-Calibrated Anomaly) engine.
```bash
python src/main.py -e 1 --data data/
```

### 2. Micro-Batch Dedupe Efficiency Bounds
Benchmarks physical processing latency overhead incurred by complex bloom filtering versus naive hashing, strictly restricted by realistic sliding window policies.
```bash
python src/main.py -e 2 --data data/
```

### 3. Pipeline Scaling & Multiprocessing
Stresses the proposed Paritioned-worker architecture, measuring runtime speedup correlations distributed over $K$ physical cores (Kafka-Partition pattern simulation).
```bash
python src/main.py -e 3 --data data/
```

## Environment Prerequisites

- macOS Metal Performance Shaders (`mps`) support is automatically handled if present.
- `pip install -r requirements.txt` required.
- All evaluation logs omit GUI window blocking mechanisms for pure terminal execution.