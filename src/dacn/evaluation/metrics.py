"""Metric helpers. Positive class denotes anomaly."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(labels, scores, threshold=None):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be same-length 1D arrays")
    if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain binary 0/1 values")
    if not np.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    if len(np.unique(labels)) < 2:
        raise ValueError("AUROC requires both normal and anomalous samples")
    out = {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores))}
    if threshold is not None:
        pred = (scores >= float(threshold)).astype(int)
        out.update(precision=float(precision_score(labels, pred, zero_division=0)), recall=float(recall_score(labels, pred, zero_division=0)), f1=float(f1_score(labels, pred, zero_division=0)), threshold=float(threshold))
    return out
