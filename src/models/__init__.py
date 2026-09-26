"""
Supervised Machine Learning Models and Evaluation for Business Entity Resolution.
"""

from src.models.metrics import (
    compute_f_beta,
    ThresholdEvaluation,
    SupervisedEvaluationReport,
    evaluate_predictions_at_threshold,
    evaluate_matching_probabilities,
)
from src.models.baseline_model import (
    BaselineMatchingModel,
    NON_FEATURE_COLUMNS,
)
from src.models.post_processing import (
    ThresholdOptimizationResult,
    optimize_threshold_oof,
    MatchPostProcessor,
    compute_prediction_diagnostics,
)
from src.models.phase4_interactions import (
    ALL_INTERACTION_FEATURES,
    BASELINE_FEATURES,
    EXPERIMENT_SPECS,
    EXP_A_FEATURES,
    EXP_B_FEATURES,
    EXP_C_FEATURES,
    EXP_ALL_FEATURES,
    compute_interaction_features,
)

__all__ = [
    "compute_f_beta",
    "ThresholdEvaluation",
    "SupervisedEvaluationReport",
    "evaluate_predictions_at_threshold",
    "evaluate_matching_probabilities",
    "BaselineMatchingModel",
    "NON_FEATURE_COLUMNS",
    "ThresholdOptimizationResult",
    "optimize_threshold_oof",
    "MatchPostProcessor",
    "compute_prediction_diagnostics",
    "ALL_INTERACTION_FEATURES",
    "BASELINE_FEATURES",
    "EXPERIMENT_SPECS",
    "EXP_A_FEATURES",
    "EXP_B_FEATURES",
    "EXP_C_FEATURES",
    "EXP_ALL_FEATURES",
    "compute_interaction_features",
]

