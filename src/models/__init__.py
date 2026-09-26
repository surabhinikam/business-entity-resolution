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
from src.models.cross_validation import (
    CVFoldMetrics,
    CVResult,
    EntityDisjointKFoldSplit,
    create_entity_disjoint_kfold_splits,
    run_cv_experiment,
)
from src.models.validation_analysis import (
    categorize_error_patterns,
    categorize_false_negative,
    categorize_false_positive,
    compute_entity_level_analysis,
    compute_score_distributions,
    compute_subgroup_metrics,
    sweep_threshold_metrics,
)
from src.models.model_family_comparison import (
    ErrorOverlapSummary,
    ModelFamilyResult,
    ModelFoldMetrics,
    build_model_for_family,
    compute_model_family_error_overlap,
    run_model_family_cv,
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
    "CVFoldMetrics",
    "CVResult",
    "EntityDisjointKFoldSplit",
    "create_entity_disjoint_kfold_splits",
    "run_cv_experiment",
    "categorize_error_patterns",
    "categorize_false_negative",
    "categorize_false_positive",
    "compute_entity_level_analysis",
    "compute_score_distributions",
    "compute_subgroup_metrics",
    "sweep_threshold_metrics",
    "ErrorOverlapSummary",
    "ModelFamilyResult",
    "ModelFoldMetrics",
    "build_model_for_family",
    "compute_model_family_error_overlap",
    "run_model_family_cv",
]

