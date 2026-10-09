"""
Classification Metrics Evaluation Module for QuantLab Pro.
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Per-Class metrics safely.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ClassificationMetrics(BaseModel):
    """Pydantic model holding evaluated classification metrics."""
    accuracy: float = Field(description="Accuracy score (0.0 to 1.0).")
    precision: float = Field(description="Macro-averaged Precision score.")
    recall: float = Field(description="Macro-averaged Recall score.")
    f1_score: float = Field(description="Macro-averaged F1 score.")
    confusion_matrix: list[list[int]] = Field(description="2x2 confusion matrix [[TN, FP], [FN, TP]].")
    per_class_report: dict[str, Any] = Field(description="Precision, Recall, F1 breakdown per target class.")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics safely handling edge cases (single class, empty arrays).
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        
    Returns:
        ClassificationMetrics object.
    """
    if len(y_true) == 0:
        return ClassificationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=[[0, 0], [0, 0]],
            per_class_report={},
        )
        
    acc = float(accuracy_score(y_true, y_pred))
    
    # Zero division handling for precision/recall/f1
    prec = float(precision_score(y_true, y_pred, zero_division=0, average="macro"))
    rec = float(recall_score(y_true, y_pred, zero_division=0, average="macro"))
    f1 = float(f1_score(y_true, y_pred, zero_division=0, average="macro"))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_list = cm.tolist()
    
    report_dict = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    
    return ClassificationMetrics(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        confusion_matrix=cm_list,
        per_class_report=report_dict,
    )
