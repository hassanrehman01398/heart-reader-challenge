"""Evaluation metrics for multi-label ECG classification."""

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    multilabel_confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute Macro F1, Macro AUC, and per-class metrics.

    Parameters
    ----------
    y_true       : (N, C) binary ground-truth labels
    y_pred_proba : (N, C) predicted probabilities
    threshold    : binarisation threshold

    Returns
    -------
    dict with macro_f1, macro_auc, per_class_f1, per_class_auc
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_auc = roc_auc_score(y_true, y_pred_proba, average="macro")

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_auc = roc_auc_score(y_true, y_pred_proba, average=None)

    return {
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_auc": per_class_auc.tolist(),
    }


def find_best_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Grid-search the threshold that maximises macro-F1 on validation data."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def print_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
):
    y_pred = (y_pred_proba >= threshold).astype(int)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
