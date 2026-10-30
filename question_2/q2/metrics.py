"""Five-class metrics and ordinal error rates with explicit denominators."""
import numpy as np
from sklearn.metrics import confusion_matrix


def metrics_from_confusion(matrix):
    cm = np.asarray(matrix, dtype=np.int64)
    total = int(cm.sum())
    if total == 0:
        raise ValueError("Cannot score an empty split")
    diagonal = np.diag(cm)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(5, dtype=float), where=denominator != 0)
    distances = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    errors = int(cm[distances > 0].sum())
    adjacent = int(cm[distances == 1].sum())
    distant = int(cm[distances >= 2].sum())
    return {"accuracy": float(diagonal.sum() / total), "macro_f1": float(f1.mean()),
            "mae": float((cm * distances).sum() / total), "per_class_f1": f1.tolist(),
            "adjacent_error_rate": adjacent / total, "distant_error_rate": distant / total,
            "adjacent_share_of_errors": adjacent / errors if errors else 0.0,
            "distant_share_of_errors": distant / errors if errors else 0.0,
            "extreme_1_vs_5_error_rate": float((cm[0, 4] + cm[4, 0]) / total),
            "n": total, "confusion_matrix": cm.tolist()}


def classification_metrics(labels, predictions):
    return metrics_from_confusion(confusion_matrix(labels, predictions, labels=list(range(5))))
