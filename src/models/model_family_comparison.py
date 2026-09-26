"""
Phase 5: Controlled Model Family Comparison Framework (LightGBM vs XGBoost vs CatBoost).

Provides:
1. Standardized model factories with comparable conservative hyperparameters.
2. Cross-validation runner over identical entity-disjoint folds.
3. Detailed multi-threshold evaluation (0.82, 0.85, 0.90, 0.95, optimal).
4. Error overlap and diversity analysis (shared vs unique FPs/FNs, corrected vs introduced errors).
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.feature_schema import ALL_FEATURE_NAMES, LABEL_COLUMN, PAIR_ID_COLUMNS
from src.models.cross_validation import EntityDisjointKFoldSplit
from src.models.metrics import evaluate_predictions_at_threshold
from src.models.phase4_interactions import BASELINE_FEATURES
from src.models.validation_analysis import (
    categorize_false_negative,
    categorize_false_positive,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Controlled Hyperparameter Configurations
# =============================================================================

# LightGBM: Frozen reference from Phase 4D
LIGHTGBM_PARAMS: Dict[str, Any] = {
    "num_leaves": 31,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "binary",
    "verbose": -1,
}

# XGBoost: Comparable conservative configuration
# Note: min_child_weight=1.0 is conservative and appropriate for the 0.006 class imbalance.
# max_depth=6 provides capacity comparable to LightGBM num_leaves=31 (2^5=32).
XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

# CatBoost: Comparable conservative configuration
# Note: depth=6 symmetric trees, 300 iterations, lr=0.03.
CATBOOST_PARAMS: Dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
}


def build_model_for_family(family: str, custom_params: Optional[Dict[str, Any]] = None) -> Any:
    """Builds a classifier instance for the specified model family."""
    fam = family.lower().strip()
    if fam == "lightgbm":
        params = dict(LIGHTGBM_PARAMS)
        if custom_params:
            params.update(custom_params)
        return lgb.LGBMClassifier(**params)
    elif fam == "xgboost":
        params = dict(XGBOOST_PARAMS)
        if custom_params:
            params.update(custom_params)
        return xgb.XGBClassifier(**params)
    elif fam == "catboost":
        params = dict(CATBOOST_PARAMS)
        if custom_params:
            params.update(custom_params)
        return cb.CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unsupported model family: '{family}'. Supported: 'lightgbm', 'xgboost', 'catboost'.")


# =============================================================================
# 2. Result Data Structures
# =============================================================================

@dataclass
class ModelFoldMetrics:
    """Per-fold metrics for a model family."""
    fold_idx: int
    train_time_sec: float
    f05_82: float
    f05_85: float
    f05_90: float
    f05_95: float
    precision_90: float
    recall_90: float
    pr_auc: float
    roc_auc: float


@dataclass
class ModelFamilyResult:
    """Comprehensive evaluation results for a single model family."""
    family_name: str
    params: Dict[str, Any]
    fold_metrics: List[ModelFoldMetrics]
    oof_probabilities: np.ndarray
    oof_y_true: np.ndarray
    total_train_time_sec: float
    avg_fold_train_time_sec: float
    peak_memory_mb: float

    # Overall OOF Metrics
    oof_pr_auc: float
    oof_roc_auc: float

    # Metrics at Key Thresholds
    metrics_at_thresholds: Dict[float, Dict[str, float]]  # t -> {p, r, f05, f1, tp, fp, fn, tn}

    # Fold Summary Metrics
    mean_f05_90: float
    std_f05_90: float
    per_fold_f05_90: List[float]

    # OOF Optimal F0.5
    optimal_f05: float
    optimal_threshold: float
    optimal_precision: float
    optimal_recall: float


# =============================================================================
# 3. Cross-Validation Runner
# =============================================================================

def run_model_family_cv(
    splits: List[EntityDisjointKFoldSplit],
    family_name: str,
    feature_cols: Sequence[str] = BASELINE_FEATURES,
    custom_params: Optional[Dict[str, Any]] = None,
    thresholds: Sequence[float] = (0.82, 0.85, 0.90, 0.95),
    label_col: str = LABEL_COLUMN,
) -> ModelFamilyResult:
    """
    Executes entity-disjoint cross-validation for a specific model family.

    Guarantees:
    - Evaluated on the exact same folds.
    - OOF predictions sorted back to match original DataFrame order.
    - Zero S1 leakage, zero pair leakage verified by splits.
    """
    n_splits = len(splits)
    total_pairs = sum(s.validation.height for s in splits)

    oof_probs = np.zeros(total_pairs, dtype=np.float64)
    oof_y_true = np.zeros(total_pairs, dtype=np.int32)
    oof_indices = np.zeros(total_pairs, dtype=np.int64)

    fold_metrics_list: List[ModelFoldMetrics] = []
    total_train_time = 0.0
    current_offset = 0

    tracemalloc.start()
    t_start_total = time.time()

    for split in splits:
        fold_idx = split.fold_idx
        n_val = split.validation.height

        # Feature matrix & labels
        X_train = split.train.select(feature_cols).to_numpy()
        y_train = split.train[label_col].to_numpy().astype(int)
        X_val = split.validation.select(feature_cols).to_numpy()
        y_val = split.validation[label_col].to_numpy().astype(int)
        if "_row_id" in split.validation.columns:
            val_row_ids = split.validation["_row_id"].to_numpy().astype(np.int64)
        elif "_row_idx" in split.validation.columns:
            val_row_ids = split.validation["_row_idx"].to_numpy().astype(np.int64)
        else:
            val_row_ids = np.arange(current_offset, current_offset + n_val, dtype=np.int64)

        model = build_model_for_family(family_name, custom_params=custom_params)

        t_fold_start = time.time()
        model.fit(X_train, y_train)
        fold_time = time.time() - t_fold_start
        total_train_time += fold_time

        val_probs = model.predict_proba(X_val)[:, 1]

        oof_probs[current_offset:current_offset + n_val] = val_probs
        oof_y_true[current_offset:current_offset + n_val] = y_val
        oof_indices[current_offset:current_offset + n_val] = val_row_ids
        current_offset += n_val

        # Fold diagnostics
        f_eval_82 = evaluate_predictions_at_threshold(y_val, val_probs, threshold=0.82)
        f_eval_85 = evaluate_predictions_at_threshold(y_val, val_probs, threshold=0.85)
        f_eval_90 = evaluate_predictions_at_threshold(y_val, val_probs, threshold=0.90)
        f_eval_95 = evaluate_predictions_at_threshold(y_val, val_probs, threshold=0.95)
        f_pr_auc = float(average_precision_score(y_val, val_probs))
        f_roc_auc = float(roc_auc_score(y_val, val_probs))

        fold_metrics_list.append(
            ModelFoldMetrics(
                fold_idx=fold_idx,
                train_time_sec=fold_time,
                f05_82=f_eval_82.f05,
                f05_85=f_eval_85.f05,
                f05_90=f_eval_90.f05,
                f05_95=f_eval_95.f05,
                precision_90=f_eval_90.precision,
                recall_90=f_eval_90.recall,
                pr_auc=f_pr_auc,
                roc_auc=f_roc_auc,
            )
        )
        logger.info(
            "[%s] Fold %d/%d (%.1fs): P@0.90=%.4f, R@0.90=%.4f, F0.5@0.90=%.4f, PR-AUC=%.4f",
            family_name, fold_idx + 1, n_splits, fold_time,
            f_eval_90.precision, f_eval_90.recall, f_eval_90.f05, f_pr_auc,
        )

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_bytes / (1024 * 1024)

    # Sort OOF predictions back to original DataFrame row order
    sort_order = np.argsort(oof_indices)
    oof_probs_sorted = oof_probs[sort_order]
    oof_y_true_sorted = oof_y_true[sort_order]

    oof_pr_auc = float(average_precision_score(oof_y_true_sorted, oof_probs_sorted))
    oof_roc_auc = float(roc_auc_score(oof_y_true_sorted, oof_probs_sorted))

    # Evaluate at specified thresholds
    metrics_at_thresh: Dict[float, Dict[str, float]] = {}
    for t in thresholds:
        e = evaluate_predictions_at_threshold(oof_y_true_sorted, oof_probs_sorted, threshold=float(t))
        metrics_at_thresh[round(float(t), 2)] = {
            "precision": round(e.precision, 4),
            "recall": round(e.recall, 4),
            "f05": round(e.f05, 4),
            "f1": round(e.f1, 4),
            "tp": int(e.true_positives),
            "fp": int(e.false_positives),
            "fn": int(e.false_negatives),
            "tn": int(e.true_negatives),
        }

    # Find OOF optimal F0.5
    sweep_grid = [round(t, 2) for t in np.linspace(0.50, 0.99, 50)]
    best_f05 = -1.0
    best_thresh = 0.50
    best_prec = 0.0
    best_rec = 0.0

    for t in sweep_grid:
        e = evaluate_predictions_at_threshold(oof_y_true_sorted, oof_probs_sorted, threshold=t)
        if e.f05 > best_f05:
            best_f05 = e.f05
            best_thresh = t
            best_prec = e.precision
            best_rec = e.recall

    per_fold_f05 = [m.f05_90 for m in fold_metrics_list]

    active_params = LIGHTGBM_PARAMS if family_name == "lightgbm" else (
        XGBOOST_PARAMS if family_name == "xgboost" else CATBOOST_PARAMS
    )
    if custom_params:
        active_params = dict(active_params)
        active_params.update(custom_params)

    return ModelFamilyResult(
        family_name=family_name,
        params=active_params,
        fold_metrics=fold_metrics_list,
        oof_probabilities=oof_probs_sorted,
        oof_y_true=oof_y_true_sorted,
        total_train_time_sec=total_train_time,
        avg_fold_train_time_sec=total_train_time / n_splits,
        peak_memory_mb=peak_mem_mb,
        oof_pr_auc=oof_pr_auc,
        oof_roc_auc=oof_roc_auc,
        metrics_at_thresholds=metrics_at_thresh,
        mean_f05_90=float(np.mean(per_fold_f05)),
        std_f05_90=float(np.std(per_fold_f05)),
        per_fold_f05_90=per_fold_f05,
        optimal_f05=round(best_f05, 4),
        optimal_threshold=best_thresh,
        optimal_precision=round(best_prec, 4),
        optimal_recall=round(best_rec, 4),
    )


# =============================================================================
# 4. Error Overlap and Complementarity Analysis
# =============================================================================

@dataclass
class ErrorOverlapSummary:
    """Detailed overlap, intersection, and category analysis across model families."""
    threshold: float
    total_positives: int
    total_negatives: int

    # False Positive Sets
    fp_counts: Dict[str, int]
    shared_fp_all: int
    unique_fp: Dict[str, int]

    # False Negative Sets
    fn_counts: Dict[str, int]
    shared_fn_all: int
    unique_fn: Dict[str, int]

    # Relative to LightGBM
    corrected_fn_by_alt: Dict[str, int]  # Positives LGBM missed (FN) that Alt got right (TP)
    corrected_fp_by_alt: Dict[str, int]  # Negatives LGBM flagged (FP) that Alt rejected (TN)
    introduced_fn_by_alt: Dict[str, int] # Positives LGBM got right (TP) that Alt missed (FN)
    introduced_fp_by_alt: Dict[str, int] # Negatives LGBM rejected (TN) that Alt flagged (FP)

    # Detailed Error Record DataFrames
    error_overlap_df: pl.DataFrame


def compute_model_family_error_overlap(
    df: pl.DataFrame,
    predictions: Dict[str, np.ndarray],
    threshold: float = 0.90,
    label_col: str = LABEL_COLUMN,
) -> ErrorOverlapSummary:
    """
    Computes set intersections, unique errors, and error categories across model families
    at a specific decision threshold (default: 0.90).
    """
    y_true = df[label_col].to_numpy().astype(int)
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    models = list(predictions.keys())

    # Build binary prediction vectors
    pred_bins = {m: (predictions[m] >= threshold).astype(int) for m in models}

    # FP sets (indices where y_true == 0 and pred == 1)
    fp_sets = {m: set(np.where((y_true == 0) & (pred_bins[m] == 1))[0]) for m in models}

    # FN sets (indices where y_true == 1 and pred == 0)
    fn_sets = {m: set(np.where((y_true == 1) & (pred_bins[m] == 0))[0]) for m in models}

    # Set intersections
    all_fps = set.intersection(*fp_sets.values()) if fp_sets else set()
    all_fns = set.intersection(*fn_sets.values()) if fn_sets else set()

    unique_fps = {
        m: len(fp_sets[m] - set.union(*[fp_sets[other] for other in models if other != m]))
        for m in models
    }
    unique_fns = {
        m: len(fn_sets[m] - set.union(*[fn_sets[other] for other in models if other != m]))
        for m in models
    }

    # Relative to LightGBM
    lgb_fp = fp_sets.get("lightgbm", set())
    lgb_fn = fn_sets.get("lightgbm", set())

    corrected_fn: Dict[str, int] = {}
    corrected_fp: Dict[str, int] = {}
    introduced_fn: Dict[str, int] = {}
    introduced_fp: Dict[str, int] = {}

    for alt in [m for m in models if m != "lightgbm"]:
        corrected_fn[alt] = len(lgb_fn - fn_sets[alt])
        corrected_fp[alt] = len(lgb_fp - fp_sets[alt])
        introduced_fn[alt] = len(fn_sets[alt] - lgb_fn)
        introduced_fp[alt] = len(fp_sets[alt] - lgb_fp)

    # Build detailed row-level error DataFrame
    union_errors = set.union(*fp_sets.values(), *fn_sets.values())
    error_indices = sorted(list(union_errors))

    sub_df = df[error_indices]

    rows: List[Dict[str, Any]] = []
    for orig_idx in error_indices:
        r = df.row(orig_idx, named=True)
        lbl = int(y_true[orig_idx])
        row_dict: Dict[str, Any] = {
            "pair_idx": orig_idx,
            "source1_entity_id": r.get("source1_entity_id", ""),
            "candidate_entity_id": r.get("candidate_entity_id", ""),
            "candidate_source": r.get("candidate_source", ""),
            "label": lbl,
        }

        # Model probabilities & predictions
        for m in models:
            p = float(predictions[m][orig_idx])
            row_dict[f"prob_{m}"] = round(p, 4)
            row_dict[f"pred_{m}"] = 1 if p >= threshold else 0

        # Error type
        if lbl == 0:
            row_dict["error_type"] = "FP"
            row_dict["error_category"] = categorize_false_positive(r)
        else:
            row_dict["error_type"] = "FN"
            row_dict["error_category"] = categorize_false_negative(r)

        # Flag sharing
        row_dict["in_lgb"] = orig_idx in (lgb_fp | lgb_fn)
        for alt in [m for m in models if m != "lightgbm"]:
            row_dict[f"in_{alt}"] = orig_idx in (fp_sets[alt] | fn_sets[alt])

        rows.append(row_dict)

    error_overlap_df = pl.DataFrame(rows)

    return ErrorOverlapSummary(
        threshold=threshold,
        total_positives=total_pos,
        total_negatives=total_neg,
        fp_counts={m: len(fp_sets[m]) for m in models},
        shared_fp_all=len(all_fps),
        unique_fp=unique_fps,
        fn_counts={m: len(fn_sets[m]) for m in models},
        shared_fn_all=len(all_fns),
        unique_fn=unique_fns,
        corrected_fn_by_alt=corrected_fn,
        corrected_fp_by_alt=corrected_fp,
        introduced_fn_by_alt=introduced_fn,
        introduced_fp_by_alt=introduced_fp,
        error_overlap_df=error_overlap_df,
    )
