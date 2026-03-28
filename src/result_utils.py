"""
result_utils.py
---------------
Utilities for experiment result storage.

Main responsibilities:
- Create a timestamped result directory
- Save experiment metadata
- Save summary tables
- Save prediction outputs
- Provide filename-safe quantile tags
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def create_result_dir(current_dir: str, base_dir: str = "result") -> str:
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    run_id = datetime.now().strftime("rf_novelty_%Y%m%d_%H%M%S")
    result_dir = os.path.join(project_root, base_dir, run_id)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_metadata(result_dir: str, metadata: dict):
    path = os.path.join(result_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_summary(result_dir: str, rows: list[dict]):
    path = os.path.join(result_dir, "summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)


def save_predictions(
    result_dir: str,
    filename: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
):
    out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })

    if scores is not None:
        out["score"] = scores

    out.to_csv(os.path.join(result_dir, filename), index=False)


def quantile_tag(q: float) -> str:
    return str(q).replace(".", "p")