"""
Threshold Optimization and Match Post-Processing for Business Entity Resolution.

Part 9:
- Leakage-safe threshold optimization for F0.5 using Out-Of-Fold (OOF) cross-validation
  strictly on the TRAINING set (never using validation labels).
- Evaluates the chosen threshold on the untouched validation set and compares against 0.50.
- Flexible match post-processing preserving:
  - S1 entities with zero matches (when no candidate exceeds threshold)
  - Multiple valid matches (when supported by model probabilities)
  - No global top-1 forcing
  - Per-source candidate separation (source2 and source3 handled independently)
- Rich diagnostics:
  - Predicted match count per S1 entity (zero-match, single-match, multi-match)
  - Probability distributions of accepted vs rejected pairs
  - Detailed error analysis of False Positives and False Negatives
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    LABEL_COLUMN,
    ALL_FEATURE_NAMES,
)
from src.models.baseline_model import BaselineMatchingModel
from src.models.metrics import (
    compute_f_beta,
    evaluate_predictions_at_threshold,
    ThresholdEvaluation,
    SupervisedEvaluationReport,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Threshold Optimization Data Structures
# =============================================================================

@dataclass
class ThresholdOptimizationResult:
    """Results from Out-Of-Fold threshold optimization on training data."""
    best_threshold: float
    best_f05: float
    best_precision: float
    best_recall: float
    best_f1: float
    oof_pr_auc: float
    oof_roc_auc: float
    threshold_sweep: List[ThresholdEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_threshold": self.best_threshold,
            "best_f05": self.best_f05,
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_f1": self.best_f1,
            "oof_pr_auc": self.oof_pr_auc,
            "oof_roc_auc": self.oof_roc_auc,
            "threshold_sweep": [te.to_dict() for te in self.threshold_sweep],
        }


# =============================================================================
# 2. Leakage-Safe Out-Of-Fold Threshold Optimizer
# =============================================================================

def optimize_threshold_oof(
    train_df: pl.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    threshold_candidates: Optional[Sequence[float]] = None,
    beta: float = 0.5,
    model_params: Optional[Dict[str, Any]] = None,
) -> ThresholdOptimizationResult:
    """
    Optimizes the decision threshold for F-beta (default F0.5) using Out-Of-Fold (OOF)
    cross-validation strictly on the TRAINING set.

    Guarantees:
    - Zero Leakage: Completely blind to validation labels. Only train_df is used.
    - Entity-Disjoint Folds: Unique source1_entity_ids are partitioned across folds,
      ensuring OOF evaluation measures generalization to unseen entities.
    - Deterministic: Reproducible with a fixed random seed.

    Parameters:
        train_df: Labeled training DataFrame with features and target 'label'.
        n_splits: Number of cross-validation folds on unique S1 entities (default 5).
        seed: Random seed for entity fold partitioning.
        threshold_candidates: Sequence of threshold values to test.
        beta: Metric weight (default 0.5 for F0.5).
        model_params: Hyperparameters for BaselineMatchingModel.

    Returns:
        ThresholdOptimizationResult with the optimal threshold and diagnostic sweep.
    """
    if LABEL_COLUMN not in train_df.columns:
        raise ValueError(f"Training DataFrame is missing target column '{LABEL_COLUMN}'")

    if threshold_candidates is None:
        threshold_candidates = [
            round(t, 2) for t in np.linspace(0.05, 0.95, 37)
        ]

    # Partition unique S1 entities across folds
    unique_s1 = sorted(train_df["source1_entity_id"].unique().to_list())
    if len(unique_s1) < n_splits:
        n_splits = max(2, len(unique_s1))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probs = np.zeros(train_df.height, dtype=np.float32)
    y_true = train_df[LABEL_COLUMN].to_numpy().astype(int)

    params = dict(model_params or {})
    params.pop("random_state", None)

    logger.info(
        f"Starting {n_splits}-fold OOF threshold optimization on {train_df.height:,} training pairs "
        f"({len(unique_s1):,} unique S1 entities, target metric=F{beta})..."
    )

    s1_array = train_df["source1_entity_id"].to_numpy()

    for fold_idx, (train_s1_idx, val_s1_idx) in enumerate(kf.split(unique_s1), 1):
        fold_train_s1 = set(unique_s1[i] for i in train_s1_idx)
        fold_val_s1 = set(unique_s1[i] for i in val_s1_idx)

        # Boolean mask over pairs
        is_train_pair = np.isin(s1_array, list(fold_train_s1))
        is_val_pair = np.isin(s1_array, list(fold_val_s1))

        fold_train_df = train_df.filter(pl.Series(is_train_pair))
        fold_val_df = train_df.filter(pl.Series(is_val_pair))

        if fold_val_df.height == 0:
            continue

        # Fit model on fold training pairs
        model = BaselineMatchingModel(
            random_state=seed + fold_idx,
            **params,
        )
        model.fit(fold_train_df)

        # Predict probabilities on out-of-fold pairs
        val_probs = model.predict_proba(fold_val_df)
        oof_probs[is_val_pair] = val_probs

        logger.debug(f"Fold {fold_idx}/{n_splits} completed: {fold_val_df.height:,} OOF pairs.")

    # Sweep thresholds on full OOF predictions
    best_thresh = 0.5
    best_f = -1.0
    best_eval: Optional[ThresholdEvaluation] = None
    sweep_evals: List[ThresholdEvaluation] = []

    for t in threshold_candidates:
        ev = evaluate_predictions_at_threshold(y_true, oof_probs, threshold=t)
        sweep_evals.append(ev)
        metric_val = ev.f05 if beta == 0.5 else compute_f_beta(y_true, oof_probs >= t, beta=beta)
        if metric_val > best_f:
            best_f = metric_val
            best_thresh = ev.threshold
            best_eval = ev

    assert best_eval is not None

    # Compute OOF PR-AUC and ROC-AUC
    from sklearn.metrics import average_precision_score, roc_auc_score
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    oof_pr_auc = float(average_precision_score(y_true, oof_probs)) if n_pos > 0 and n_neg > 0 else 0.0
    oof_roc_auc = float(roc_auc_score(y_true, oof_probs)) if n_pos > 0 and n_neg > 0 else 0.0

    logger.info(
        f"OOF Threshold Optimization complete: Selected Threshold = {best_thresh:.2f} "
        f"(OOF F{beta}={best_eval.f05:.4f}, Prec={best_eval.precision:.4f}, Rec={best_eval.recall:.4f}, "
        f"PR-AUC={oof_pr_auc:.4f})."
    )

    return ThresholdOptimizationResult(
        best_threshold=best_thresh,
        best_f05=best_eval.f05,
        best_precision=best_eval.precision,
        best_recall=best_eval.recall,
        best_f1=best_eval.f1,
        oof_pr_auc=oof_pr_auc,
        oof_roc_auc=oof_roc_auc,
        threshold_sweep=sweep_evals,
    )


# =============================================================================
# 3. Match Post-Processor
# =============================================================================

class MatchPostProcessor:
    """
    Flexible match post-processing utility for candidate pairs with probabilities.

    Guarantees:
    - Never forces top-1 globally: S1 entities with zero qualifying matches receive zero matches;
      S1 entities with multiple strong candidates retain all valid matches.
    - Handles Source2 and Source3 independently without collapsing composite identities.
    - Configurable optional per-source top-K filtering (e.g. max 1 or 2 per source).
    - Configurable optional score-margin filtering.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_matches_per_source: Optional[int] = None,
        score_margin: Optional[float] = None,
        prob_col: str = "probability",
    ):
        """
        Initialize post-processor.

        Parameters:
            threshold: Probability decision threshold.
            max_matches_per_source: Optional maximum matches allowed per source (e.g. 1 per S2, 1 per S3).
                If None (default), all candidates >= threshold are preserved.
            score_margin: Optional maximum difference from top candidate's probability for an S1 entity.
            prob_col: Name of probability column.
        """
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        self.threshold = threshold
        self.max_matches_per_source = max_matches_per_source
        self.score_margin = score_margin
        self.prob_col = prob_col

    def filter_candidate_pairs(
        self,
        df_with_probs: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Filters candidate pairs based on threshold, source separation, and optional ranking rules.

        Returns:
            Filtered Polars DataFrame containing only accepted pairs, sorted by S1 ID and probability descending.
        """
        for col in PAIR_ID_COLUMNS:
            if col not in df_with_probs.columns:
                raise ValueError(f"Missing required identity column: '{col}'")
        if self.prob_col not in df_with_probs.columns:
            raise ValueError(f"Missing probability column: '{self.prob_col}'")

        if df_with_probs.height == 0:
            return df_with_probs

        # 1. Base threshold filter
        filtered = df_with_probs.filter(pl.col(self.prob_col) >= self.threshold)

        if filtered.height == 0:
            return filtered

        # 2. Score margin filter (if specified)
        if self.score_margin is not None:
            max_probs = (
                filtered.group_by("source1_entity_id")
                .agg(pl.col(self.prob_col).max().alias("_max_prob"))
            )
            filtered = (
                filtered.join(max_probs, on="source1_entity_id", how="left")
                .filter(pl.col(self.prob_col) >= (pl.col("_max_prob") - self.score_margin))
                .drop("_max_prob")
            )

        # 3. Optional per-source top-K filter (without global top-1 forcing)
        if self.max_matches_per_source is not None and self.max_matches_per_source > 0:
            filtered = (
                filtered.sort(["source1_entity_id", "candidate_source", self.prob_col], descending=[False, False, True])
                .group_by(["source1_entity_id", "candidate_source"], maintain_order=True)
                .head(self.max_matches_per_source)
            )

        return filtered.sort(["source1_entity_id", self.prob_col], descending=[False, True])

    def format_submission_linkages(
        self,
        accepted_pairs_df: pl.DataFrame,
        all_s1_entities: Sequence[str],
    ) -> pl.DataFrame:
        """
        Formats accepted matches into competition submission format:
        [source1_entity_id, matched_entity_ids] (comma-separated string).

        Guarantees:
        - Every entity in all_s1_entities appears in the output table.
        - Zero-match entities receive an empty string ("").
        - Multi-match entities have their candidate IDs joined with commas (e.g. "S2-101,S3-202").
        """
        if accepted_pairs_df.height == 0:
            return pl.DataFrame({
                "source1_entity_id": list(all_s1_entities),
                "matched_entity_ids": ["" for _ in all_s1_entities],
            })

        # Group accepted candidate entity IDs per S1
        grouped = (
            accepted_pairs_df.sort(["source1_entity_id", self.prob_col], descending=[False, True])
            .group_by("source1_entity_id", maintain_order=True)
            .agg(pl.col("candidate_entity_id").str.join(","))
            .rename({"candidate_entity_id": "matched_entity_ids"})
        )

        all_df = pl.DataFrame({"source1_entity_id": list(all_s1_entities)})
        submission_df = (
            all_df.join(grouped, on="source1_entity_id", how="left")
            .with_columns(pl.col("matched_entity_ids").fill_null(""))
        )

        return submission_df


# =============================================================================
# 4. Diagnostics & Error Analysis
# =============================================================================

def compute_prediction_diagnostics(
    val_df: pl.DataFrame,
    probs: np.ndarray,
    threshold: float,
    label_col: str = LABEL_COLUMN,
    top_errors_to_inspect: int = 10,
) -> Dict[str, Any]:
    """
    Generates detailed diagnostics on validation predictions:
    - Match count distributions per S1 entity (zero-match, single-match, multi-match)
    - Source breakdown (source2 vs source3)
    - Accepted vs rejected probability distribution stats
    - False Positive and False Negative candidate pair inspection
    """
    if len(val_df) != len(probs):
        raise ValueError("val_df and probs must have identical lengths")

    df_eval = val_df.with_columns([
        pl.Series("probability", probs, dtype=pl.Float64),
        pl.Series("pred_label", (probs >= threshold).astype(int), dtype=pl.Int8),
    ])

    all_s1 = sorted(df_eval["source1_entity_id"].unique().to_list())
    total_s1 = len(all_s1)

    accepted_df = df_eval.filter(pl.col("pred_label") == 1)
    rejected_df = df_eval.filter(pl.col("pred_label") == 0)

    # 1. Match count distribution per S1 entity
    if accepted_df.height > 0:
        counts_per_s1 = (
            accepted_df.group_by("source1_entity_id")
            .agg([
                pl.len().alias("total_pred_matches"),
                (pl.col("candidate_source") == "source2").sum().alias("s2_matches"),
                (pl.col("candidate_source") == "source3").sum().alias("s3_matches"),
            ])
        )
        count_dict = dict(zip(counts_per_s1["source1_entity_id"].to_list(), counts_per_s1["total_pred_matches"].to_list()))
    else:
        count_dict = {}

    match_counts = [count_dict.get(s1, 0) for s1 in all_s1]
    zero_match_s1 = sum(1 for c in match_counts if c == 0)
    single_match_s1 = sum(1 for c in match_counts if c == 1)
    multi_match_s1 = sum(1 for c in match_counts if c > 1)
    max_matches_single_s1 = max(match_counts) if match_counts else 0

    # 2. Source breakdown
    s2_accepted = int((accepted_df["candidate_source"] == "source2").sum()) if accepted_df.height > 0 else 0
    s3_accepted = int((accepted_df["candidate_source"] == "source3").sum()) if accepted_df.height > 0 else 0

    # 3. Probability distribution statistics
    def _prob_stats(series: pl.Series) -> Dict[str, float]:
        if series.len() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
        arr = series.to_numpy()
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    accepted_prob_stats = _prob_stats(accepted_df["probability"])
    rejected_prob_stats = _prob_stats(rejected_df["probability"])

    # 4. Error Analysis: False Positives & False Negatives
    fp_df = df_eval.filter((pl.col("pred_label") == 1) & (pl.col(label_col) == 0)).sort("probability", descending=True)
    fn_df = df_eval.filter((pl.col("pred_label") == 0) & (pl.col(label_col) == 1)).sort("probability", descending=True)

    key_diagnostic_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "probability",
        label_col,
    ]
    # Add informative features if present
    for extra_col in ["name_char_3gram_jaccard", "address_char_3gram_similarity", "shared_address_number_count", "matched_key_count"]:
        if extra_col in df_eval.columns:
            key_diagnostic_cols.append(extra_col)

    fp_list = fp_df.select(key_diagnostic_cols).head(top_errors_to_inspect).to_dicts()
    fn_list = fn_df.select(key_diagnostic_cols).head(top_errors_to_inspect).to_dicts()

    return {
        "total_s1_entities": total_s1,
        "zero_match_s1_count": zero_match_s1,
        "zero_match_s1_pct": round(zero_match_s1 / total_s1 * 100.0, 2) if total_s1 > 0 else 0.0,
        "single_match_s1_count": single_match_s1,
        "single_match_s1_pct": round(single_match_s1 / total_s1 * 100.0, 2) if total_s1 > 0 else 0.0,
        "multi_match_s1_count": multi_match_s1,
        "multi_match_s1_pct": round(multi_match_s1 / total_s1 * 100.0, 2) if total_s1 > 0 else 0.0,
        "max_matches_single_s1": max_matches_single_s1,
        "source2_accepted": s2_accepted,
        "source3_accepted": s3_accepted,
        "accepted_prob_stats": accepted_prob_stats,
        "rejected_prob_stats": rejected_prob_stats,
        "false_positive_count": fp_df.height,
        "false_negative_count": fn_df.height,
        "false_positives": fp_list,
        "false_negatives": fn_list,
    }
