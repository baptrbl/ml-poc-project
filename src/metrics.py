"""Metrics used to evaluate the student-risk model."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return stable classification metrics for comparing models."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got {y_true_array.shape} and {y_pred_array.shape}."
        )

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
    }
