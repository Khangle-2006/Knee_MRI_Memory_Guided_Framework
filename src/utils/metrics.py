"""
Classification metrics: AUC, Accuracy, Sensitivity, Specificity, F1
and Youden's-J optimal threshold.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    f1_score, recall_score, roc_curve,
)


def get_optimal_threshold(y_true, y_probs):
    """
    Find the threshold maximizing Youden's J statistic:
        J = Sensitivity + Specificity - 1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr  # = tpr + (1 - fpr) - 1
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"AUC": auc, "ACC": acc, "SENS": sens,
            "SPEC": spec, "F1": f1, "Thresh": threshold}
