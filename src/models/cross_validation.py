"""
Entity-Disjoint K-Fold Cross-Validation Framework for Business Entity Resolution.

Design & Guarantees:
- Grouping Unit: source1_entity_id.
- Zero Leakage: The same S1 entity NEVER appears in both training and validation sets of any fold.
- Candidate-Pair Disjoint: Zero candidate pair overlap across validation folds.
- Stratification: Stratifies S1 entities by positive match count (0, 1, 2+) to ensure
  balanced class ratios and positive presence across all folds.
- Deterministic: Seed-controlled reproducibility (seed=42).
- Out-of-Fold (OOF) Evaluation: Generates complete out-of-fold probability vectors
  for un-biased, global ranking (PR-AUC, ROC-AUC) and threshold calibration.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl

from src.features.feature_schema import (
    ALL_FEATURE_NAMES,
    LABEL_COLUMN,
    PAIR_ID_COLUMNS,
)
from src.features.training_dataset import (
    compute_split_stats,
    DatasetSplitStats,
)
from src.models.baseline_model import BaselineMatchingModel
from src.models.metrics import (
    ThresholdEvaluation,
    evaluate_matching_probabilities,
    evaluate_predictions_at_threshold,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Data Structures
# =============================================================================

@dataclass
class EntityDisjointKFoldSplit:
    """Represents a single entity-disjoint cross-validation fold."""
    fold_idx: int
    train: pl.DataFrame
    validation: pl.DataFrame
    train_stats: DatasetSplitStats
    validation_stats: DatasetSplitStats
    s1_overlap_count: int  # Must be 0
    candidate_pair_overlap_count: int  # Must be 0


@dataclass
class CVFoldMetrics:
    """Metrics evaluated on a single validation fold."""
    fold_idx: int
    pr_auc: float
    roc_auc: float
    f05_50: float
    precision_50: float
    recall_50: float
    f1_50: float
    f05_90: float
    precision_90: float
    recall_90: float
    f1_90: float
    tp_90: int
    fp_90: int
    tn_90: int
    fn_90: int
    train_time: float


@dataclass
class CVResult:
    """Complete summary of a K-Fold cross-validation experiment."""
    config_name: str
    feature_names: List[str]
    lgbm_params: Dict[str, Any]
    fold_metrics: List[CVFoldMetrics]
    oof_probabilities: np.ndarray
    oof_y_true: np.ndarray
    oof_indices: np.ndarray  # Indices matching original dataframe order
    mean_pr_auc: float
    std_pr_auc: float
    mean_roc_auc: float
    std_roc_auc: float
    mean_f05_90: float
    std_f05_90: float
    oof_f05_50: float
    oof_prec_50: float
    oof_rec_50: float
    oof_f05_90: float
    oof_prec_90: float
    oof_rec_90: float
    oof_opt_f05: float
    oof_opt_threshold: float
    oof_opt_prec: float
    oof_opt_rec: float
    oof_pr_auc: float
    oof_roc_auc: float
    total_train_time: float
    avg_fold_train_time: float
    feature_importances: pl.DataFrame


# =============================================================================
# 2. Entity-Disjoint K-Fold Splitting Logic
# =============================================================================

def create_entity_disjoint_kfold_splits(
    labeled_df: pl.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    label_col: str = LABEL_COLUMN,
) -> List[EntityDisjointKFoldSplit]:
    """
    Partitions a labeled candidate dataset into n_splits entity-disjoint folds
    grouped strictly by source1_entity_id and stratified by positive match count.

    Guarantees:
    - Zero S1 leakage across train and validation for all folds.
    - Zero candidate pair overlap across validation sets.
    - Exhaustive partition: Every candidate pair is in validation exactly once.
    - Positive presence: Every fold receives a balanced share of positives.
    - Determinism: Fixed seed guarantees reproducible fold assignment.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    for col in PAIR_ID_COLUMNS:
        if col not in labeled_df.columns:
            raise ValueError(f"Missing required identity column: '{col}'")
    if label_col not in labeled_df.columns:
        raise ValueError(f"Missing required label column: '{label_col}'")

    # Add temporary original index column for reconstructing OOF order
    if "_row_id" not in labeled_df.columns:
        df_indexed = labeled_df.with_columns(
            pl.int_range(0, labeled_df.height).alias("_row_id")
        )
    else:
        df_indexed = labeled_df

    all_s1_entities = sorted(df_indexed["source1_entity_id"].unique().to_list())
    rng = random.Random(seed)

    # Count positive candidate pairs per S1 entity for stratification
    pos_counts = (
        df_indexed.filter(pl.col(label_col) == 1)
        .group_by("source1_entity_id")
        .len()
    )
    pos_dict = dict(zip(pos_counts["source1_entity_id"].to_list(), pos_counts["len"].to_list()))

    # Stratified buckets: 0 positives, 1 positive, 2+ positives
    bucket_0: List[str] = []
    bucket_1: List[str] = []
    bucket_2plus: List[str] = []

    for s1_id in all_s1_entities:
        c = pos_dict.get(s1_id, 0)
        if c == 0:
            bucket_0.append(s1_id)
        elif c == 1:
            bucket_1.append(s1_id)
        else:
            bucket_2plus.append(s1_id)

    # Deterministic shuffle within each bucket
    rng.shuffle(bucket_0)
    rng.shuffle(bucket_1)
    rng.shuffle(bucket_2plus)

    # Distribute S1 entities round-robin across folds
    fold_s1_sets: List[Set[str]] = [set() for _ in range(n_splits)]
    for bucket in [bucket_0, bucket_1, bucket_2plus]:
        for i, s1_id in enumerate(bucket):
            fold_s1_sets[i % n_splits].add(s1_id)

    splits: List[EntityDisjointKFoldSplit] = []

    for fold_idx in range(n_splits):
        val_s1 = fold_s1_sets[fold_idx]
        train_s1 = set().union(*[fold_s1_sets[j] for j in range(n_splits) if j != fold_idx])

        # Strict disjointness verification
        s1_overlap = len(train_s1.intersection(val_s1))
        if s1_overlap != 0:
            raise RuntimeError(f"Fold {fold_idx}: S1 leakage detected ({s1_overlap} overlapping entities)!")

        val_df = df_indexed.filter(pl.col("source1_entity_id").is_in(list(val_s1)))
        train_df = df_indexed.filter(pl.col("source1_entity_id").is_in(list(train_s1)))

        # Candidate pair overlap check
        pair_overlap = int(
            train_df.select(PAIR_ID_COLUMNS)
            .join(val_df.select(PAIR_ID_COLUMNS), on=PAIR_ID_COLUMNS, how="inner")
            .height
        )
        if pair_overlap != 0:
            raise RuntimeError(f"Fold {fold_idx}: Pair leakage detected ({pair_overlap} overlapping pairs)!")

        train_stats = compute_split_stats(train_df, label_col=label_col)
        val_stats = compute_split_stats(val_df, label_col=label_col)

        if val_stats.positive_pairs == 0:
            raise RuntimeError(f"Fold {fold_idx}: Validation set has 0 positive pairs!")

        splits.append(
            EntityDisjointKFoldSplit(
                fold_idx=fold_idx,
                train=train_df,
                validation=val_df,
                train_stats=train_stats,
                validation_stats=val_stats,
                s1_overlap_count=s1_overlap,
                candidate_pair_overlap_count=pair_overlap,
            )
        )

    logger.info(
        "Created %d entity-disjoint folds across %d total pairs (%d S1 entities).",
        n_splits, df_indexed.height, len(all_s1_entities),
    )
    return splits


# =============================================================================
# 3. Cross-Validation Model Evaluation
# =============================================================================

def run_cv_experiment(
    splits: List[EntityDisjointKFoldSplit],
    config_name: str,
    feature_names: List[str],
    lgbm_params: Dict[str, Any],
    label_col: str = LABEL_COLUMN,
) -> CVResult:
    """
    Executes entity-disjoint K-fold cross-validation for a specific LightGBM configuration.

    Computes fold-level metrics, aggregates out-of-fold (OOF) probabilities,
    and returns comprehensive CV results.
    """
    n_splits = len(splits)
    total_pairs = sum(s.validation.height for s in splits)

    oof_probs = np.zeros(total_pairs, dtype=np.float32)
    oof_y_true = np.zeros(total_pairs, dtype=np.int32)
    oof_indices = np.zeros(total_pairs, dtype=np.int64)

    fold_metrics_list: List[CVFoldMetrics] = []
    fold_importances: List[pl.DataFrame] = []
    total_train_time = 0.0

    current_offset = 0

    for split in splits:
        fold_idx = split.fold_idx

        # Model instance with configuration parameters
        model = BaselineMatchingModel(
            n_estimators=lgbm_params.get("n_estimators", 150),
            learning_rate=lgbm_params.get("learning_rate", 0.05),
            num_leaves=lgbm_params.get("num_leaves", 31),
            max_depth=lgbm_params.get("max_depth", -1),
            min_child_samples=lgbm_params.get("min_child_samples", 20),
            subsample=lgbm_params.get("subsample", 0.8),
            colsample_bytree=lgbm_params.get("colsample_bytree", 0.8),
            random_state=lgbm_params.get("random_state", 42),
            importance_type="gain",
            feature_names=feature_names,
        )

        # Train model
        t0 = time.perf_counter()
        model.fit(split.train)
        fold_time = time.perf_counter() - t0
        total_train_time += fold_time

        # Validate on out-of-fold partition
        val_probs = model.predict_proba(split.validation)
        val_y = split.validation[label_col].to_numpy().astype(int)
        val_row_ids = split.validation["_row_id"].to_numpy().astype(np.int64)

        n_val = len(val_probs)
        oof_probs[current_offset:current_offset + n_val] = val_probs
        oof_y_true[current_offset:current_offset + n_val] = val_y
        oof_indices[current_offset:current_offset + n_val] = val_row_ids
        current_offset += n_val

        # Fold metrics
        rep = evaluate_matching_probabilities(val_y, val_probs)
        e50 = evaluate_predictions_at_threshold(val_y, val_probs, threshold=0.50)
        e90 = evaluate_predictions_at_threshold(val_y, val_probs, threshold=0.90)

        fold_metrics_list.append(
            CVFoldMetrics(
                fold_idx=fold_idx,
                pr_auc=rep.pr_auc,
                roc_auc=rep.roc_auc,
                f05_50=e50.f05,
                precision_50=e50.precision,
                recall_50=e50.recall,
                f1_50=e50.f1,
                f05_90=e90.f05,
                precision_90=e90.precision,
                recall_90=e90.recall,
                f1_90=e90.f1,
                tp_90=e90.true_positives,
                fp_90=e90.false_positives,
                tn_90=e90.true_negatives,
                fn_90=e90.false_negatives,
                train_time=fold_time,
            )
        )

        fold_importances.append(model.get_feature_importances())

    # Sort OOF predictions back to original row order
    sort_order = np.argsort(oof_indices)
    oof_probs_sorted = oof_probs[sort_order]
    oof_y_true_sorted = oof_y_true[sort_order]

    # Global Out-Of-Fold metrics
    oof_rep = evaluate_matching_probabilities(oof_y_true_sorted, oof_probs_sorted)
    oof_e50 = evaluate_predictions_at_threshold(oof_y_true_sorted, oof_probs_sorted, threshold=0.50)
    oof_e90 = evaluate_predictions_at_threshold(oof_y_true_sorted, oof_probs_sorted, threshold=0.90)

    # Find OOF-optimal threshold
    threshold_sweep = np.linspace(0.05, 0.99, 95)
    best_opt: Optional[ThresholdEvaluation] = None
    for t in threshold_sweep:
        eval_t = evaluate_predictions_at_threshold(oof_y_true_sorted, oof_probs_sorted, threshold=float(t))
        if best_opt is None or eval_t.f05 > best_opt.f05:
            best_opt = eval_t

    assert best_opt is not None

    # Fold statistics aggregation
    pr_aucs = [fm.pr_auc for fm in fold_metrics_list]
    roc_aucs = [fm.roc_auc for fm in fold_metrics_list]
    f05_90s = [fm.f05_90 for fm in fold_metrics_list]

    # Aggregate feature importances across folds
    feat_names = fold_importances[0]["feature_name"].to_list()
    avg_gains: Dict[str, float] = {f: 0.0 for f in feat_names}
    for fi in fold_importances:
        for row in fi.iter_rows(named=True):
            avg_gains[row["feature_name"]] += row["importance"] / n_splits

    total_gain = sum(avg_gains.values())
    avg_imp_df = (
        pl.DataFrame({
            "feature_name": list(avg_gains.keys()),
            "importance": list(avg_gains.values()),
            "relative_pct": [round((v / total_gain * 100.0), 2) if total_gain > 0 else 0.0 for v in avg_gains.values()],
        })
        .sort("importance", descending=True)
        .with_columns(pl.int_range(1, len(avg_gains) + 1).alias("rank"))
    )

    return CVResult(
        config_name=config_name,
        feature_names=feature_names,
        lgbm_params=lgbm_params,
        fold_metrics=fold_metrics_list,
        oof_probabilities=oof_probs_sorted,
        oof_y_true=oof_y_true_sorted,
        oof_indices=oof_indices[sort_order],
        mean_pr_auc=float(np.mean(pr_aucs)),
        std_pr_auc=float(np.std(pr_aucs)),
        mean_roc_auc=float(np.mean(roc_aucs)),
        std_roc_auc=float(np.std(roc_aucs)),
        mean_f05_90=float(np.mean(f05_90s)),
        std_f05_90=float(np.std(f05_90s)),
        oof_f05_50=oof_e50.f05,
        oof_prec_50=oof_e50.precision,
        oof_rec_50=oof_e50.recall,
        oof_f05_90=oof_e90.f05,
        oof_prec_90=oof_e90.precision,
        oof_rec_90=oof_e90.recall,
        oof_opt_f05=best_opt.f05,
        oof_opt_threshold=best_opt.threshold,
        oof_opt_prec=best_opt.precision,
        oof_opt_rec=best_opt.recall,
        oof_pr_auc=oof_rep.pr_auc,
        oof_roc_auc=oof_rep.roc_auc,
        total_train_time=total_train_time,
        avg_fold_train_time=total_train_time / n_splits,
        feature_importances=avg_imp_df,
    )
