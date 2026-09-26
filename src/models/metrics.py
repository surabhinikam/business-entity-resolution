"""
Evaluation metrics for Entity Resolution supervised matching.

Primary metric: F0.5 score (weights precision 2x over recall).
Diagnostic metrics: Precision, Recall, F1, Confusion Matrix, PR-AUC, ROC-AUC,
and threshold sweep evaluations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def compute_f_beta(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    beta: float = 0.5,
) -> float:
    """
    Computes F-beta score with safe zero-division handling.

    F0.5 = (1 + 0.25) * (Precision * Recall) / (0.25 * Precision + Recall)
    Weights precision twice as much as recall.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    return float(fbeta_score(y_t, y_p, beta=beta, zero_division=0.0))


@dataclass
class ThresholdEvaluation:
    """Evaluation metrics computed at a specific decision threshold."""
    threshold: float
    predicted_positives: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    f05: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "predicted_positives": self.predicted_positives,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f05": self.f05,
        }


@dataclass
class SupervisedEvaluationReport:
    """Comprehensive evaluation report for supervised matching model."""
    pr_auc: float
    roc_auc: float
    total_samples: int
    positive_samples: int
    negative_samples: int
    threshold_evaluations: List[ThresholdEvaluation]
    default_metrics_at_05: ThresholdEvaluation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "total_samples": self.total_samples,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "default_metrics_at_05": self.default_metrics_at_05.to_dict(),
            "threshold_sweep": [te.to_dict() for te in self.threshold_evaluations],
        }


def evaluate_predictions_at_threshold(
    y_true: Union[np.ndarray, Sequence[int]],
    y_prob: Union[np.ndarray, Sequence[float]],
    threshold: float = 0.5,
) -> ThresholdEvaluation:
    """
    Computes precision, recall, F1, F0.5, and confusion matrix counts
    at a given decision threshold.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_p_prob >= threshold).astype(int)

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_t, y_pred, zero_division=0.0))
    rec = float(recall_score(y_t, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0.0))
    f05 = compute_f_beta(y_t, y_pred, beta=0.5)

    return ThresholdEvaluation(
        threshold=round(float(threshold), 4),
        predicted_positives=tp + fp,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=prec,
        recall=rec,
        f1=f1,
        f05=f05,
    )


def evaluate_matching_probabilities(
    y_true: Union[np.ndarray, Sequence[int]],
    y_prob: Union[np.ndarray, Sequence[float]],
    thresholds: Optional[Sequence[float]] = None,
) -> SupervisedEvaluationReport:
    """
    Evaluates continuous matching probabilities across a sweep of thresholds.

    Computes:
    - Primary metric: F0.5 at each threshold
    - Diagnostic metrics: PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix
    - Threshold sweep (without freezing or prematurely optimizing threshold)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}")

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))

    # Safe PR-AUC and ROC-AUC
    if n_pos > 0 and n_neg > 0:
        pr_auc = float(average_precision_score(y_t, y_p))
        roc_auc = float(roc_auc_score(y_t, y_p))
    else:
        pr_auc = 0.0
        roc_auc = 0.0

    if thresholds is None:
        thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    evaluations: List[ThresholdEvaluation] = [
        evaluate_predictions_at_threshold(y_t, y_p, thresh) for thresh in thresholds
    ]

    default_05 = evaluate_predictions_at_threshold(y_t, y_p, 0.50)

    return SupervisedEvaluationReport(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        total_samples=len(y_t),
        positive_samples=n_pos,
        negative_samples=n_neg,
        threshold_evaluations=evaluations,
        default_metrics_at_05=default_05,
    )
