"""
evaluator.py
------------
Produces standardized academic metrics (F1, Precision, Recall, False Alarm Rate).
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_classification_report(model_name: str, y_true, y_pred) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("[metrics/evaluator] y_true and y_pred rank mismatch.")
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN = cm[0, 0]
    FP = cm[0, 1]
    
    far = FP / max(1, (FP + TN))
    
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "far": round(far, 4)
    }

def print_metrics_table(results_list: list):
    if not results_list: return
    df = pd.DataFrame(results_list)
    print("\n" + "="*70)
    print(" DETECTION PERFORMANCE EVALUATION ")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
