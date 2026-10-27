"""Shared utilities: paths, seeding, metrics, result I/O."""
import json
import os
import random
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SEED = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def compute_metrics(y_true, y_pred, n_classes: int) -> dict:
    """Accuracy, macro precision/recall/F1, weighted F1, MCC, FPR."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    labels = list(range(n_classes))
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        out["fpr"] = fp / (fp + tn) if (fp + tn) else 0.0
        out["detection_rate"] = tp / (tp + fn) if (tp + fn) else 0.0
    else:
        # macro one-vs-rest FPR
        fprs = []
        total = cm.sum()
        for i in labels:
            fp = cm[:, i].sum() - cm[i, i]
            tn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
        out["fpr_macro"] = float(np.mean(fprs))
    out["confusion_matrix"] = cm.tolist()
    return out


def save_results(name: str, payload: dict) -> str:
    payload = dict(payload)
    payload["_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[saved] {path}")
    return path


def print_table(rows: list, headers: list) -> None:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
