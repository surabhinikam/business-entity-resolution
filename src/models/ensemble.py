"""
Phase 6: Multi-Model Ensembling & Calibration Experiment Framework.

Provides:
1. Probability blending (weighted average) and rank averaging with tie-breaking.
2. Comprehensive multi-threshold evaluation across entity-disjoint folds.
3. Fold-level stability analysis (F0.5 @ 0.90 per fold, mean, standard deviation).
4. Error overlap and exchange analysis against standalone LightGBM baseline.
5. Subgroup diagnostics across candidate sources (source2/source3) and countries (IN/US).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.feature_schema import LABEL_COLUMN, PAIR_ID_COLUMNS
from src.models.metrics import evaluate_predictions_at_threshold
from src.models.validation_analysis import (
    categorize_false_negative,
    categorize_false_positive,
    compute_subgroup_metrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Blending Operations
# =============================================================================

def blend_weighted_probabilities(
    predictions_map: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Computes a weighted linear blend of prediction probabilities from multiple models.

    Args:
        predictions_map: Dict mapping model name to 1D float array of predicted probabilities.
        weights: Dict mapping model name to positive float blend weight.

    Returns:
        1D float array of blended probabilities bounded to [0.0, 1.0].
    """
    for model_name in weights:
        if model_name not in predictions_map:
            raise KeyError(f"Model '{model_name}' specified in weights not found in predictions_map.")

    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError(f"Sum of blend weights must be > 0, got {total_weight}")

    normalized_weights = {m: w / total_weight for m, w in weights.items()}

    first_model = next(iter(weights))
    n_samples = len(predictions_map[first_model])
    blended = np.zeros(n_samples, dtype=np.float64)

    for model_name, w in normalized_weights.items():
        arr = predictions_map[model_name]
        if len(arr) != n_samples:
            raise ValueError(
                f"Prediction length mismatch: '{first_model}' has {n_samples}, "
                f"'{model_name}' has {len(arr)}"
            )
        blended += w * arr

    return np.clip(blended, 0.0, 1.0)


def rank_transform(probs: np.ndarray, method: str = "average") -> np.ndarray:
    """
    Transforms continuous prediction probabilities into normalized percentile ranks in [0, 1].

    Implementation:
    Uses scipy.stats.rankdata with 'average' tie-breaking.
    Maps ranks 1..N linearly to [0.0, 1.0] via (rank - 1.0) / (N - 1.0).
    For N=144,048, lowest probability maps to 0.0, highest maps to 1.0.
    """
    n = len(probs)
    if n <= 1:
        return np.ones_like(probs, dtype=np.float64)
    ranks = stats.rankdata(probs, method=method)
    return (ranks - 1.0) / (n - 1.0)


def blend_rank_average(
    predictions_map: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Computes normalized rank average across multiple models.

    Args:
        predictions_map: Dict mapping model name to 1D float array of predicted probabilities.
        weights: Optional dict mapping model name to blend weight. Defaults to equal weighting.

    Returns:
        1D float array of averaged percentile ranks in [0.0, 1.0].
    """
    if weights is None:
        weights = {m: 1.0 for m in predictions_map}

    for model_name in weights:
        if model_name not in predictions_map:
            raise KeyError(f"Model '{model_name}' specified in weights not found in predictions_map.")

    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError(f"Sum of rank weights must be > 0, got {total_weight}")

    normalized_weights = {m: w / total_weight for m, w in weights.items()}

    first_model = next(iter(weights))
    n_samples = len(predictions_map[first_model])
    blended_rank = np.zeros(n_samples, dtype=np.float64)

    for model_name, w in normalized_weights.items():
        arr = predictions_map[model_name]
        if len(arr) != n_samples:
            raise ValueError(
                f"Prediction length mismatch: '{first_model}' has {n_samples}, "
                f"'{model_name}' has {len(arr)}"
            )
        norm_r = rank_transform(arr)
        blended_rank += w * norm_r

    return np.clip(blended_rank, 0.0, 1.0)


# =============================================================================
# 2. Data Structures & Configurations
# =============================================================================

@dataclass
class EnsembleDefinition:
    """Specification of an ensemble candidate."""
    name: str
    description: str
    method: str  # 'probability_blend', 'rank_average', or 'baseline'
    weights: Dict[str, float]
    num_models: int = field(init=False)

    def __post_init__(self):
        self.num_models = len(self.weights)


# Standard 10 Ensembles required in Phase 6B
STANDARD_ENSEMBLES: List[EnsembleDefinition] = [
    # 1. Baseline
    EnsembleDefinition(
        name="lightgbm_baseline",
        description="Standalone LightGBM reference (num_leaves=31, est=300, lr=0.03)",
        method="baseline",
        weights={"lightgbm": 1.0},
    ),
    # 2. 0.75 LGB + 0.25 XGB
    EnsembleDefinition(
        name="blend_lgb0.75_xgb0.25",
        description="0.75 LightGBM + 0.25 XGBoost probability blend",
        method="probability_blend",
        weights={"lightgbm": 0.75, "xgboost": 0.25},
    ),
    # 3. 0.50 LGB + 0.50 XGB
    EnsembleDefinition(
        name="blend_lgb0.50_xgb0.50",
        description="0.50 LightGBM + 0.50 XGBoost equal probability blend",
        method="probability_blend",
        weights={"lightgbm": 0.50, "xgboost": 0.50},
    ),
    # 4. 0.75 LGB + 0.25 CatBoost
    EnsembleDefinition(
        name="blend_lgb0.75_cat0.25",
        description="0.75 LightGBM + 0.25 CatBoost probability blend",
        method="probability_blend",
        weights={"lightgbm": 0.75, "catboost": 0.25},
    ),
    # 5. 0.50 LGB + 0.50 CatBoost
    EnsembleDefinition(
        name="blend_lgb0.50_cat0.50",
        description="0.50 LightGBM + 0.50 CatBoost equal probability blend",
        method="probability_blend",
        weights={"lightgbm": 0.50, "catboost": 0.50},
    ),
    # 6. 1/3 LGB + 1/3 XGB + 1/3 CatBoost
    EnsembleDefinition(
        name="blend_three_model_equal",
        description="1/3 LightGBM + 1/3 XGBoost + 1/3 CatBoost equal probability blend",
        method="probability_blend",
        weights={"lightgbm": 1.0 / 3.0, "xgboost": 1.0 / 3.0, "catboost": 1.0 / 3.0},
    ),
    # 7. Rank average: LGB + XGB
    EnsembleDefinition(
        name="rank_lgb_xgb",
        description="Rank averaging of LightGBM + XGBoost",
        method="rank_average",
        weights={"lightgbm": 0.50, "xgboost": 0.50},
    ),
    # 8. Rank average: LGB + CatBoost
    EnsembleDefinition(
        name="rank_lgb_cat",
        description="Rank averaging of LightGBM + CatBoost",
        method="rank_average",
        weights={"lightgbm": 0.50, "catboost": 0.50},
    ),
    # 9. Rank average: All three
    EnsembleDefinition(
        name="rank_all_three",
        description="Rank averaging of all three models (LightGBM, XGBoost, CatBoost)",
        method="rank_average",
        weights={"lightgbm": 1.0 / 3.0, "xgboost": 1.0 / 3.0, "catboost": 1.0 / 3.0},
    ),
    # 10. Precision-oriented blend: 0.70 LGB + 0.20 XGB + 0.10 CatBoost
    EnsembleDefinition(
        name="blend_precision_oriented",
        description="0.70 LightGBM + 0.20 XGBoost + 0.10 CatBoost precision-weighted blend",
        method="probability_blend",
        weights={"lightgbm": 0.70, "xgboost": 0.20, "catboost": 0.10},
    ),
]


@dataclass
class EnsembleFoldMetric:
    """Fold-level evaluation for an ensemble."""
    fold_idx: int
    f05_80: float
    f05_82: float
    f05_85: float
    f05_90: float
    f05_95: float
    precision_90: float
    recall_90: float
    pr_auc: float
    roc_auc: float


@dataclass
class EnsembleResult:
    """Complete evaluation outcome for an ensemble configuration."""
    definition: EnsembleDefinition
    scores: np.ndarray
    oof_pr_auc: float
    oof_roc_auc: float
    metrics_at_thresholds: Dict[float, Dict[str, Any]]
    fold_metrics: List[EnsembleFoldMetric]
    mean_f05_90: float
    std_f05_90: float
    optimal_f05: float
    optimal_threshold: float
    optimal_precision: float
    optimal_recall: float
    optimal_tp: int
    optimal_fp: int
    optimal_fn: int
    optimal_pred_matches: int


# =============================================================================
# 3. Evaluation Engine
# =============================================================================

def evaluate_ensemble(
    definition: EnsembleDefinition,
    predictions_map: Dict[str, np.ndarray],
    y_true: np.ndarray,
    fold_assignments: np.ndarray,
    thresholds: Sequence[float] = (0.80, 0.82, 0.85, 0.90, 0.95),
) -> EnsembleResult:
    """
    Evaluates an ensemble configuration over pooled OOF and per-fold partitions.

    Args:
        definition: EnsembleDefinition configuration.
        predictions_map: Map of model name to aligned 1D probability array.
        y_true: Ground truth binary labels (0 or 1).
        fold_assignments: 1D int array indicating fold index (0..K-1) for each sample.
        thresholds: Fixed operational thresholds to evaluate.

    Returns:
        EnsembleResult containing full diagnostic metrics.
    """
    # 1. Compute blended scores
    if definition.method == "rank_average":
        scores = blend_rank_average(predictions_map, definition.weights)
    elif definition.method in ("probability_blend", "baseline"):
        scores = blend_weighted_probabilities(predictions_map, definition.weights)
    else:
        raise ValueError(f"Unknown ensemble method: {definition.method}")

    # 2. Overall PR-AUC and ROC-AUC
    oof_pr_auc = float(average_precision_score(y_true, scores))
    oof_roc_auc = float(roc_auc_score(y_true, scores))

    # 3. Multi-threshold metrics
    metrics_at_thresh: Dict[float, Dict[str, Any]] = {}
    for t in thresholds:
        t_float = round(float(t), 2)
        eval_t = evaluate_predictions_at_threshold(y_true, scores, threshold=t_float)
        metrics_at_thresh[t_float] = {
            "precision": round(eval_t.precision, 4),
            "recall": round(eval_t.recall, 4),
            "f05": round(eval_t.f05, 4),
            "f1": round(eval_t.f1, 4),
            "tp": int(eval_t.true_positives),
            "fp": int(eval_t.false_positives),
            "fn": int(eval_t.false_negatives),
            "tn": int(eval_t.true_negatives),
            "predicted_matches": int(eval_t.predicted_positives),
        }

    # 4. Fold-level validation
    unique_folds = sorted(np.unique(fold_assignments).tolist())
    fold_metrics_list: List[EnsembleFoldMetric] = []
    f05_90_folds: List[float] = []

    for f_idx in unique_folds:
        mask = (fold_assignments == f_idx)
        y_f = y_true[mask]
        s_f = scores[mask]

        e80 = evaluate_predictions_at_threshold(y_f, s_f, threshold=0.80)
        e82 = evaluate_predictions_at_threshold(y_f, s_f, threshold=0.82)
        e85 = evaluate_predictions_at_threshold(y_f, s_f, threshold=0.85)
        e90 = evaluate_predictions_at_threshold(y_f, s_f, threshold=0.90)
        e95 = evaluate_predictions_at_threshold(y_f, s_f, threshold=0.95)

        pr_auc_f = float(average_precision_score(y_f, s_f)) if np.sum(y_f) > 0 else 0.0
        roc_auc_f = float(roc_auc_score(y_f, s_f)) if np.sum(y_f) > 0 else 0.0

        fold_metrics_list.append(
            EnsembleFoldMetric(
                fold_idx=int(f_idx),
                f05_80=e80.f05,
                f05_82=e82.f05,
                f05_85=e85.f05,
                f05_90=e90.f05,
                f05_95=e95.f05,
                precision_90=e90.precision,
                recall_90=e90.recall,
                pr_auc=pr_auc_f,
                roc_auc=roc_auc_f,
            )
        )
        f05_90_folds.append(e90.f05)

    mean_f05_90 = float(np.mean(f05_90_folds))
    std_f05_90 = float(np.std(f05_90_folds))

    # 5. Optimal Threshold Sweep (0.50 -> 0.999)
    # For rank average, candidates concentrate in top 1%, so sweep up to 0.999
    if definition.method == "rank_average":
        sweep_grid = list(np.linspace(0.50, 0.98, 49)) + list(np.linspace(0.981, 0.999, 50))
    else:
        sweep_grid = list(np.linspace(0.50, 0.99, 50))

    best_f05 = -1.0
    best_thresh = 0.50
    best_eval = None

    for t in sweep_grid:
        t_val = round(float(t), 4)
        e = evaluate_predictions_at_threshold(y_true, scores, threshold=t_val)
        if e.f05 > best_f05:
            best_f05 = e.f05
            best_thresh = t_val
            best_eval = e

    assert best_eval is not None

    return EnsembleResult(
        definition=definition,
        scores=scores,
        oof_pr_auc=oof_pr_auc,
        oof_roc_auc=oof_roc_auc,
        metrics_at_thresholds=metrics_at_thresh,
        fold_metrics=fold_metrics_list,
        mean_f05_90=mean_f05_90,
        std_f05_90=std_f05_90,
        optimal_f05=round(best_f05, 4),
        optimal_threshold=best_thresh,
        optimal_precision=round(best_eval.precision, 4),
        optimal_recall=round(best_eval.recall, 4),
        optimal_tp=int(best_eval.true_positives),
        optimal_fp=int(best_eval.false_positives),
        optimal_fn=int(best_eval.false_negatives),
        optimal_pred_matches=int(best_eval.predicted_positives),
    )


# =============================================================================
# 4. Error Overlap and Exchange Analysis
# =============================================================================

@dataclass
class EnsembleErrorExchange:
    """Detailed error exchange between an ensemble and LightGBM baseline at tau=0.90."""
    ensemble_name: str
    threshold: float
    total_positives: int
    total_negatives: int
    lgb_fps: int
    lgb_fns: int
    ens_fps: int
    ens_fns: int
    corrected_fps: int  # LGB was FP -> Ensemble is TN
    corrected_fns: int  # LGB was FN -> Ensemble is TP
    introduced_fps: int  # LGB was TN -> Ensemble is FP
    introduced_fns: int  # LGB was TP -> Ensemble is FN
    net_error_change: int  # (corrected_fps + corrected_fns) - (introduced_fps + introduced_fns)
    error_overlap_df: pl.DataFrame


def compute_ensemble_error_exchange(
    df: pl.DataFrame,
    lgb_probs: np.ndarray,
    ens_scores: np.ndarray,
    ensemble_name: str,
    threshold: float = 0.90,
    label_col: str = LABEL_COLUMN,
) -> EnsembleErrorExchange:
    """
    Computes pair-by-pair error exchange between baseline LightGBM and an ensemble at a given threshold.
    """
    y_true = df[label_col].to_numpy().astype(int)
    n = len(y_true)

    lgb_pred = (lgb_probs >= threshold).astype(int)
    ens_pred = (ens_scores >= threshold).astype(int)

    # Errors
    lgb_fp_mask = (y_true == 0) & (lgb_pred == 1)
    lgb_fn_mask = (y_true == 1) & (lgb_pred == 0)

    ens_fp_mask = (y_true == 0) & (ens_pred == 1)
    ens_fn_mask = (y_true == 1) & (ens_pred == 0)

    corrected_fp_mask = lgb_fp_mask & (ens_pred == 0)
    corrected_fn_mask = lgb_fn_mask & (ens_pred == 1)

    introduced_fp_mask = (y_true == 0) & (lgb_pred == 0) & (ens_pred == 1)
    introduced_fn_mask = (y_true == 1) & (lgb_pred == 1) & (ens_pred == 0)

    any_error_mask = lgb_fp_mask | lgb_fn_mask | ens_fp_mask | ens_fn_mask
    error_indices = np.where(any_error_mask)[0]

    error_rows: List[Dict[str, Any]] = []
    for idx in error_indices:
        r = df.row(int(idx), named=True)
        lbl = int(y_true[idx])
        l_p = float(lgb_probs[idx])
        e_s = float(ens_scores[idx])
        l_pred = int(lgb_pred[idx])
        e_pred = int(ens_pred[idx])

        # Status
        if corrected_fp_mask[idx]:
            status = "corrected_fp"
            err_type = "FP"
            err_cat = categorize_false_positive(r)
        elif corrected_fn_mask[idx]:
            status = "corrected_fn"
            err_type = "FN"
            err_cat = categorize_false_negative(r)
        elif introduced_fp_mask[idx]:
            status = "introduced_fp"
            err_type = "FP"
            err_cat = categorize_false_positive(r)
        elif introduced_fn_mask[idx]:
            status = "introduced_fn"
            err_type = "FN"
            err_cat = categorize_false_negative(r)
        elif lgb_fp_mask[idx] and ens_fp_mask[idx]:
            status = "shared_fp"
            err_type = "FP"
            err_cat = categorize_false_positive(r)
        else:
            status = "shared_fn"
            err_type = "FN"
            err_cat = categorize_false_negative(r)

        error_rows.append({
            "pair_idx": int(idx),
            "source1_entity_id": r["source1_entity_id"],
            "candidate_entity_id": r["candidate_entity_id"],
            "candidate_source": r["candidate_source"],
            "label": lbl,
            "prob_lightgbm": round(l_p, 4),
            "pred_lightgbm": l_pred,
            "score_ensemble": round(e_s, 4),
            "pred_ensemble": e_pred,
            "status": status,
            "error_type": err_type,
            "error_category": err_cat,
        })

    error_df = pl.DataFrame(error_rows) if error_rows else pl.DataFrame()

    c_fp = int(np.sum(corrected_fp_mask))
    c_fn = int(np.sum(corrected_fn_mask))
    i_fp = int(np.sum(introduced_fp_mask))
    i_fn = int(np.sum(introduced_fn_mask))

    return EnsembleErrorExchange(
        ensemble_name=ensemble_name,
        threshold=threshold,
        total_positives=int(np.sum(y_true == 1)),
        total_negatives=int(np.sum(y_true == 0)),
        lgb_fps=int(np.sum(lgb_fp_mask)),
        lgb_fns=int(np.sum(lgb_fn_mask)),
        ens_fps=int(np.sum(ens_fp_mask)),
        ens_fns=int(np.sum(ens_fn_mask)),
        corrected_fps=c_fp,
        corrected_fns=c_fn,
        introduced_fps=i_fp,
        introduced_fns=i_fn,
        net_error_change=(c_fp + c_fn) - (i_fp + i_fn),
        error_overlap_df=error_df,
    )
