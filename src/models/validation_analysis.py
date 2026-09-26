"""
Phase 4E: Comprehensive Validation, Threshold & Error Analysis Utilities.

Provides reusable analytical functions for:
1. Fine threshold sweeping and tradeoff evaluation (Precision, Recall, F0.5, F1).
2. Entity-level resolution dynamics (top-1 accuracy, score margin, zero-GT behavior).
3. Rule-based error categorization for False Positives and False Negatives.
4. Subgroup breakdown across data sources (source2/source3) and countries (US/India).
5. Probability distribution and calibration binning across true and predicted classes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from src.features.feature_schema import (
    LABEL_COLUMN,
    PAIR_ID_COLUMNS,
)
from src.models.metrics import evaluate_predictions_at_threshold

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Fine Threshold Sweeping
# =============================================================================

def sweep_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> pl.DataFrame:
    """
    Sweeps decision thresholds and calculates full confusion and scoring metrics.

    Returns:
        Polars DataFrame sorted by threshold ascending.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.50, 0.99, 50)]

    rows: List[Dict[str, Any]] = []
    for t in thresholds:
        t_float = float(t)
        eval_t = evaluate_predictions_at_threshold(y_true, y_prob, threshold=t_float)
        rows.append({
            "threshold": round(t_float, 2),
            "tp": eval_t.true_positives,
            "fp": eval_t.false_positives,
            "fn": eval_t.false_negatives,
            "tn": eval_t.true_negatives,
            "precision": round(eval_t.precision, 4),
            "recall": round(eval_t.recall, 4),
            "f05": round(eval_t.f05, 4),
            "f1": round(eval_t.f1, 4),
            "predicted_matches": eval_t.predicted_positives,
        })

    return pl.DataFrame(rows)


# =============================================================================
# 2. Entity-Level Resolution Analysis
# =============================================================================

def compute_entity_level_analysis(
    df_with_preds: pl.DataFrame,
    label_col: str = LABEL_COLUMN,
    prob_col: str = "probability",
) -> pl.DataFrame:
    """
    Computes entity-resolution metrics grouped by source1_entity_id.

    Calculates:
    - Candidate count per S1
    - Ground-truth match count
    - Predicted match count at 0.82, 0.90, 0.95
    - Maximum prediction probability (top 1)
    - Second highest probability (top 2)
    - Probability margin (top 1 - top 2)
    - Whether top-1 candidate is a true GT match
    """
    required_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source", label_col, prob_col]
    for c in required_cols:
        if c not in df_with_preds.columns:
            raise ValueError(f"Missing required column '{c}' in input DataFrame.")

    # Sort candidates by probability descending within each S1
    sorted_df = df_with_preds.sort(
        by=["source1_entity_id", prob_col],
        descending=[False, True],
    )

    entity_records: List[Dict[str, Any]] = []

    # Process grouped by source1_entity_id
    # Using Polars partition_by or iterate over groups
    grouped = sorted_df.partition_by("source1_entity_id", as_dict=True)

    for s1_key, group_df in grouped.items():
        s1_id = s1_key[0] if isinstance(s1_key, tuple) else s1_key
        cand_count = group_df.height
        labels = group_df[label_col].to_numpy()
        probs = group_df[prob_col].to_numpy()
        cands = group_df["candidate_entity_id"].to_list()
        sources = group_df["candidate_source"].to_list()

        gt_count = int(np.sum(labels == 1))
        pred_82 = int(np.sum(probs >= 0.82))
        pred_90 = int(np.sum(probs >= 0.90))
        pred_95 = int(np.sum(probs >= 0.95))

        top1_prob = float(probs[0])
        top1_cand = str(cands[0])
        top1_src = str(sources[0])
        top1_correct = bool(labels[0] == 1)

        top2_prob = float(probs[1]) if cand_count > 1 else 0.0
        prob_margin = top1_prob - top2_prob

        entity_records.append({
            "source1_entity_id": s1_id,
            "cand_count": cand_count,
            "gt_matches": gt_count,
            "pred_matches_82": pred_82,
            "pred_matches_90": pred_90,
            "pred_matches_95": pred_95,
            "top1_prob": round(top1_prob, 4),
            "top2_prob": round(top2_prob, 4),
            "prob_margin": round(prob_margin, 4),
            "has_zero_gt": gt_count == 0,
            "top1_is_correct": top1_correct,
            "top1_candidate_id": top1_cand,
            "top1_source": top1_src,
        })

    return pl.DataFrame(entity_records)


# =============================================================================
# 3. Rule-Based Error Categorization
# =============================================================================

def categorize_false_positive(row: Dict[str, Any]) -> str:
    """Categorizes a False Positive pair based on feature values."""
    addr_sim = float(row.get("address_char_3gram_similarity", 0.0) or 0.0)
    name_sim = float(row.get("name_char_3gram_jaccard", 0.0) or 0.0)
    addr_missing_cand = int(row.get("address_missing_candidate", 0) or 0)
    shared_nums = int(row.get("shared_address_number_count", 0) or 0)

    # 1. Same street/mall address with different business
    if addr_sim >= 0.75 and name_sim < 0.65:
        return "SHARED_LOCATION_DIFFERENT_BIZ"

    # 2. High name similarity but candidate address is missing
    if name_sim >= 0.75 and addr_missing_cand == 1:
        return "NAME_SIMILAR_MISSING_CAND_ADDR"

    # 3. Address number collision with divergent address text
    if shared_nums > 0 and addr_sim < 0.50 and name_sim < 0.70:
        return "ADDRESS_NUMBER_COLLISION"

    # 4. Generic/Common business words with weak address
    if 0.50 <= name_sim < 0.75 and addr_sim < 0.50:
        return "GENERIC_NAME_WEAK_ADDRESS"

    # 5. Near identical names across different branches/locations
    if name_sim >= 0.85 and addr_sim < 0.40:
        return "SIMILAR_NAME_BRANCH_MISMATCH"

    return "OTHER_FALSE_POSITIVE"


def categorize_false_negative(row: Dict[str, Any]) -> str:
    """Categorizes a False Negative pair based on feature values."""
    addr_sim = float(row.get("address_char_3gram_similarity", 0.0) or 0.0)
    name_sim = float(row.get("name_char_3gram_jaccard", 0.0) or 0.0)
    addr_missing_cand = int(row.get("address_missing_candidate", 0) or 0)
    addr_missing_s1 = int(row.get("address_missing_s1", 0) or 0)
    exact_translit = int(row.get("name_exact_translit", 0) or 0)
    len_ratio = float(row.get("name_char_len_ratio", 1.0) or 1.0)
    acronym = int(row.get("name_acronym_match", 0) or 0)
    first_tok = int(row.get("name_first_token_exact", 0) or 0)

    # 1. Missing candidate address in ground truth match
    if addr_missing_cand == 1 or addr_missing_s1 == 1:
        return "MISSING_ADDRESS_EVIDENCE"

    # 2. Severe address divergence (different branch/city) despite same name
    if addr_sim < 0.30 and name_sim >= 0.60:
        return "SEVERE_ADDRESS_DIVERGENCE"

    # 3. Transliteration / script divergence
    if exact_translit == 1 or (len_ratio < 0.60 and name_sim < 0.60):
        return "TRANSLITERATION_OR_SCRIPT_VARIANT"

    # 4. Abbreviation or acronym variants
    if acronym == 1 or (first_tok == 1 and name_sim < 0.50):
        return "ABBREVIATION_OR_ACRONYM"

    # 5. Weak name similarity despite present address
    if name_sim < 0.40:
        return "NAME_DIVERGENCE"

    return "OTHER_FALSE_NEGATIVE"


def categorize_error_patterns(
    df_with_preds: pl.DataFrame,
    threshold: float = 0.90,
    label_col: str = LABEL_COLUMN,
    prob_col: str = "probability",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Extracts and categorizes all False Positives and False Negatives at a threshold.
    """
    preds = (df_with_preds[prob_col] >= threshold).cast(pl.Int8)
    augmented = df_with_preds.with_columns(preds.alias("_pred"))

    # False Positives: label == 0 and _pred == 1
    fp_raw = augmented.filter((pl.col(label_col) == 0) & (pl.col("_pred") == 1))
    fp_cats = [categorize_false_positive(r) for r in fp_raw.iter_rows(named=True)]
    fp_df = fp_raw.with_columns(pl.Series("error_category", fp_cats)).sort(prob_col, descending=True)

    # False Negatives: label == 1 and _pred == 0
    fn_raw = augmented.filter((pl.col(label_col) == 1) & (pl.col("_pred") == 0))
    fn_cats = [categorize_false_negative(r) for r in fn_raw.iter_rows(named=True)]
    fn_df = fn_raw.with_columns(pl.Series("error_category", fn_cats)).sort(prob_col, descending=False)

    return fp_df, fn_df


# =============================================================================
# 4. Subgroup Metric Breakdown
# =============================================================================

def compute_subgroup_metrics(
    df_with_preds: pl.DataFrame,
    group_col: str,
    thresholds: Sequence[float],
    label_col: str = LABEL_COLUMN,
    prob_col: str = "probability",
) -> pl.DataFrame:
    """
    Computes precision, recall, and F0.5 across distinct values of group_col
    for each threshold in thresholds.
    """
    if group_col not in df_with_preds.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    groups = sorted(df_with_preds[group_col].unique().to_list())
    rows: List[Dict[str, Any]] = []

    for grp in groups:
        sub_df = df_with_preds.filter(pl.col(group_col) == grp)
        cand_count = sub_df.height
        y_true = sub_df[label_col].to_numpy().astype(int)
        y_prob = sub_df[prob_col].to_numpy().astype(float)
        pos_count = int(np.sum(y_true == 1))

        for t in thresholds:
            t_float = float(t)
            eval_t = evaluate_predictions_at_threshold(y_true, y_prob, threshold=t_float)
            rows.append({
                "subgroup_dimension": group_col,
                "subgroup_value": str(grp),
                "threshold": round(t_float, 2),
                "candidates": cand_count,
                "positives": pos_count,
                "tp": eval_t.true_positives,
                "fp": eval_t.false_positives,
                "fn": eval_t.false_negatives,
                "precision": round(eval_t.precision, 4),
                "recall": round(eval_t.recall, 4),
                "f05": round(eval_t.f05, 4),
                "f1": round(eval_t.f1, 4),
            })

    return pl.DataFrame(rows)


# =============================================================================
# 5. Score Distribution & Calibration Binning
# =============================================================================

def compute_score_distributions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: Optional[Sequence[float]] = None,
) -> pl.DataFrame:
    """
    Computes histogram distributions and positive fractions across probability bins.
    """
    if bins is None:
        bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.90, 0.95, 1.0001]

    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    rows: List[Dict[str, Any]] = []
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i + 1]
        mask = (y_prob >= low) & (y_prob < high)

        bin_count = int(np.sum(mask))
        bin_pos = int(np.sum(mask & (y_true == 1)))
        bin_neg = int(np.sum(mask & (y_true == 0)))
        bin_precision = (bin_pos / bin_count) if bin_count > 0 else 0.0
        pct_of_all_pos = (bin_pos / total_pos * 100.0) if total_pos > 0 else 0.0
        pct_of_all_neg = (bin_neg / total_neg * 100.0) if total_neg > 0 else 0.0

        label_str = f"[{low:.2f}, {high if high <= 1.0 else 1.0:.2f})"
        if high > 1.0:
            label_str = f"[{low:.2f}, 1.00]"

        rows.append({
            "bin_range": label_str,
            "total_pairs": bin_count,
            "positives": bin_pos,
            "negatives": bin_neg,
            "empirical_precision": round(bin_precision, 4),
            "pct_of_all_positives": round(pct_of_all_pos, 2),
            "pct_of_all_negatives": round(pct_of_all_neg, 2),
        })

    return pl.DataFrame(rows)
