"""
metrics.py
----------
Shared metrics computation and result-saving utilities for NIDSaaS-DDA.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from typing import Dict, List, Any

def classification_report(model_name, y_true, y_pred):
    """
    Compute standard classification metrics.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "far": round(far, 4),
        "confusion_matrix": cm # Keep for plotting
    }

def save_confusion_matrix(cm, save_path, model_name):
    """
    Saves a heatmap of the confusion matrix.
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_f1_comparison(results, save_path):
    """
    Saves a bar chart comparing F1 scores across models.
    """
    df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='model', y='f1_score', palette='viridis')
    plt.title("Model F1-Score Comparison")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def efficiency_report(label, total, elapsed, mem_delta, dedup_stats=None):
    """
    Standardizes efficiency reporting.
    """
    throughput = total / elapsed if elapsed > 0 else 0
    latency = (elapsed / total * 1e6) if total > 0 else 0
    
    res = {
        "condition": label,
        "total_records": total,
        "elapsed_s": round(elapsed, 4),
        "throughput_rec_s": round(throughput, 1),
        "avg_latency_us": round(latency, 2),
        "memory_delta_mb": round(mem_delta, 2)
    }
    if dedup_stats:
        res.update(dedup_stats)
    return res

def save_table(rows, path):
    df = pd.DataFrame(rows)
    # Drop confusion matrix if it exists (don't save in CSV)
    if 'confusion_matrix' in df.columns:
        df = df.drop(columns=['confusion_matrix'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[metrics] Saved table to {path}")
    print(df.to_string(index=False))

def save_bar_chart(df, x, y, title, save_path, ylabel=""):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=x, y=y, palette='magma')
    plt.title(title)
    if ylabel: plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
