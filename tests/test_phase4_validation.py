"""
Unit tests for Phase 4E: Comprehensive Validation, Threshold & Error Analysis.

Tests:
1. sweep_threshold_metrics: metric monotonicity, edge cases, correct fields.
2. compute_entity_level_analysis: grouping, top1/top2 probs, margins, correct match tracking.
3. categorize_false_positive: rule-based error categories for FPs.
4. categorize_false_negative: rule-based error categories for FNs.
5. categorize_error_patterns: separation and categorization of FP/FN.
6. compute_subgroup_metrics: subgroup breakdown (source2/source3, US/India).
7. compute_score_distributions: calibration and binning distributions.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.validation_analysis import (
    categorize_error_patterns,
    categorize_false_negative,
    categorize_false_positive,
    compute_entity_level_analysis,
    compute_score_distributions,
    compute_subgroup_metrics,
    sweep_threshold_metrics,
)


# =============================================================================
# 1. Tests for sweep_threshold_metrics
# =============================================================================

def test_sweep_threshold_metrics_schema_and_monotonicity():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.85, 0.75, 0.90, 0.60, 0.40, 0.20, 0.10], dtype=float)

    thresholds = [0.50, 0.70, 0.80, 0.90]
    df = sweep_threshold_metrics(y_true, y_prob, thresholds=thresholds)

    assert isinstance(df, pl.DataFrame)
    assert df.height == 4
    expected_cols = [
        "threshold", "tp", "fp", "fn", "tn",
        "precision", "recall", "f05", "f1", "predicted_matches"
    ]
    for col in expected_cols:
        assert col in df.columns

    # As threshold increases, predicted matches should be non-increasing
    preds = df["predicted_matches"].to_list()
    assert preds == sorted(preds, reverse=True)

    # Check specific values at threshold 0.90:
    # y_prob >= 0.90: 0.95 (y=1, TP), 0.90 (y=0, FP) -> TP=1, FP=1, FN=2, TN=4
    row_90 = df.filter(pl.col("threshold") == 0.90).to_dicts()[0]
    assert row_90["tp"] == 1
    assert row_90["fp"] == 1
    assert row_90["fn"] == 2
    assert row_90["tn"] == 4
    assert row_90["precision"] == 0.50
    assert row_90["recall"] == pytest.approx(1 / 3, abs=1e-3)


# =============================================================================
# 2. Tests for compute_entity_level_analysis
# =============================================================================

def test_compute_entity_level_analysis():
    data = pl.DataFrame({
        "source1_entity_id": ["E1", "E1", "E1", "E2", "E2", "E3"],
        "candidate_entity_id": ["C1_A", "C1_B", "C1_C", "C2_A", "C2_B", "C3_A"],
        "candidate_source": ["source2", "source3", "source2", "source2", "source3", "source3"],
        "label": [1, 0, 0, 0, 0, 1],
        "probability": [0.96, 0.88, 0.40, 0.70, 0.65, 0.92],
    })

    result = compute_entity_level_analysis(data)

    assert result.height == 3
    assert "source1_entity_id" in result.columns
    assert "cand_count" in result.columns
    assert "gt_matches" in result.columns
    assert "top1_prob" in result.columns
    assert "top2_prob" in result.columns
    assert "prob_margin" in result.columns
    assert "has_zero_gt" in result.columns
    assert "top1_is_correct" in result.columns

    records = {r["source1_entity_id"]: r for r in result.to_dicts()}

    # E1: 3 candidates, top1=0.96 (label 1), top2=0.88 -> margin 0.08, top1_correct=True
    e1 = records["E1"]
    assert e1["cand_count"] == 3
    assert e1["gt_matches"] == 1
    assert e1["top1_prob"] == 0.96
    assert e1["top2_prob"] == 0.88
    assert e1["prob_margin"] == 0.08
    assert e1["has_zero_gt"] is False
    assert e1["top1_is_correct"] is True
    assert e1["top1_candidate_id"] == "C1_A"
    assert e1["pred_matches_82"] == 2
    assert e1["pred_matches_90"] == 1
    assert e1["pred_matches_95"] == 1

    # E2: 2 candidates, zero GT matches
    e2 = records["E2"]
    assert e2["cand_count"] == 2
    assert e2["gt_matches"] == 0
    assert e2["has_zero_gt"] is True
    assert e2["top1_is_correct"] is False
    assert e2["prob_margin"] == 0.05

    # E3: single candidate, top2_prob should be 0.0
    e3 = records["E3"]
    assert e3["cand_count"] == 1
    assert e3["top2_prob"] == 0.0
    assert e3["prob_margin"] == 0.92
    assert e3["top1_is_correct"] is True


# =============================================================================
# 3. Tests for Error Categorization
# =============================================================================

def test_categorize_false_positive():
    # 1. Shared location with different biz
    fp1 = {"address_char_3gram_similarity": 0.85, "name_char_3gram_jaccard": 0.40}
    assert categorize_false_positive(fp1) == "SHARED_LOCATION_DIFFERENT_BIZ"

    # 2. High name similarity but candidate address missing
    fp2 = {"address_missing_candidate": 1, "name_char_3gram_jaccard": 0.80, "address_char_3gram_similarity": 0.0}
    assert categorize_false_positive(fp2) == "NAME_SIMILAR_MISSING_CAND_ADDR"

    # 3. Address number collision
    fp3 = {
        "shared_address_number_count": 2,
        "address_char_3gram_similarity": 0.30,
        "name_char_3gram_jaccard": 0.50,
    }
    assert categorize_false_positive(fp3) == "ADDRESS_NUMBER_COLLISION"

    # 4. Generic name with weak address
    fp4 = {"name_char_3gram_jaccard": 0.60, "address_char_3gram_similarity": 0.20}
    assert categorize_false_positive(fp4) == "GENERIC_NAME_WEAK_ADDRESS"

    # 5. Similar name branch mismatch
    fp5 = {"name_char_3gram_jaccard": 0.90, "address_char_3gram_similarity": 0.25}
    assert categorize_false_positive(fp5) == "SIMILAR_NAME_BRANCH_MISMATCH"


def test_categorize_false_negative():
    # 1. Missing address evidence
    fn1 = {"address_missing_candidate": 1, "name_char_3gram_jaccard": 0.70}
    assert categorize_false_negative(fn1) == "MISSING_ADDRESS_EVIDENCE"

    # 2. Severe address divergence
    fn2 = {
        "address_missing_candidate": 0,
        "address_missing_s1": 0,
        "address_char_3gram_similarity": 0.20,
        "name_char_3gram_jaccard": 0.75,
    }
    assert categorize_false_negative(fn2) == "SEVERE_ADDRESS_DIVERGENCE"

    # 3. Transliteration / script variant
    fn3 = {
        "address_missing_candidate": 0,
        "address_missing_s1": 0,
        "name_exact_translit": 1,
        "address_char_3gram_similarity": 0.80,
    }
    assert categorize_false_negative(fn3) == "TRANSLITERATION_OR_SCRIPT_VARIANT"

    # 4. Abbreviation or acronym
    fn4 = {
        "address_missing_candidate": 0,
        "address_missing_s1": 0,
        "name_exact_translit": 0,
        "name_acronym_match": 1,
        "address_char_3gram_similarity": 0.80,
    }
    assert categorize_false_negative(fn4) == "ABBREVIATION_OR_ACRONYM"

    # 5. Name divergence
    fn5 = {
        "address_missing_candidate": 0,
        "address_missing_s1": 0,
        "name_exact_translit": 0,
        "name_acronym_match": 0,
        "name_char_3gram_jaccard": 0.25,
        "address_char_3gram_similarity": 0.80,
    }
    assert categorize_false_negative(fn5) == "NAME_DIVERGENCE"


def test_categorize_error_patterns():
    df = pl.DataFrame({
        "source1_entity_id": ["E1", "E2", "E3", "E4"],
        "candidate_entity_id": ["C1", "C2", "C3", "C4"],
        "candidate_source": ["source2", "source3", "source2", "source3"],
        "label": [1, 0, 1, 0],
        "probability": [0.95, 0.92, 0.40, 0.10],
        "name_char_3gram_jaccard": [0.95, 0.85, 0.30, 0.10],
        "address_char_3gram_similarity": [0.90, 0.20, 0.70, 0.10],
        "address_missing_candidate": [0, 0, 0, 0],
        "address_missing_s1": [0, 0, 0, 0],
        "shared_address_number_count": [1, 0, 0, 0],
        "name_exact_translit": [0, 0, 0, 0],
        "name_char_len_ratio": [1.0, 1.0, 1.0, 1.0],
        "name_acronym_match": [0, 0, 0, 0],
        "name_first_token_exact": [1, 1, 0, 0],
    })

    # At threshold 0.90:
    # E1: prob=0.95, label=1 -> TP
    # E2: prob=0.92, label=0 -> FP
    # E3: prob=0.40, label=1 -> FN
    # E4: prob=0.10, label=0 -> TN
    fp_df, fn_df = categorize_error_patterns(df, threshold=0.90)

    assert fp_df.height == 1
    assert fp_df["source1_entity_id"][0] == "E2"
    assert "error_category" in fp_df.columns

    assert fn_df.height == 1
    assert fn_df["source1_entity_id"][0] == "E3"
    assert "error_category" in fn_df.columns
    assert fn_df["error_category"][0] == "NAME_DIVERGENCE"


# =============================================================================
# 4. Tests for Subgroup Metrics
# =============================================================================

def test_compute_subgroup_metrics():
    df = pl.DataFrame({
        "source1_entity_id": ["E1", "E2", "E3", "E4"],
        "candidate_entity_id": ["C1", "C2", "C3", "C4"],
        "candidate_source": ["source2", "source2", "source3", "source3"],
        "label": [1, 0, 1, 0],
        "probability": [0.92, 0.85, 0.95, 0.10],
    })

    sub_df = compute_subgroup_metrics(df, group_col="candidate_source", thresholds=[0.90])

    assert sub_df.height == 2
    groups = sub_df["subgroup_value"].to_list()
    assert "source2" in groups
    assert "source3" in groups

    s2_row = sub_df.filter(pl.col("subgroup_value") == "source2").to_dicts()[0]
    # In source2 at 0.90: E1 (0.92, label 1) is TP, E2 (0.85, label 0) is TN
    assert s2_row["tp"] == 1
    assert s2_row["fp"] == 0
    assert s2_row["fn"] == 0
    assert s2_row["precision"] == 1.0
    assert s2_row["recall"] == 1.0


# =============================================================================
# 5. Tests for Score Distributions
# =============================================================================

def test_compute_score_distributions():
    y_true = np.array([1, 1, 0, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.85, 0.88, 0.40, 0.05], dtype=float)

    bins = [0.0, 0.5, 0.8, 1.0001]
    dist_df = compute_score_distributions(y_true, y_prob, bins=bins)

    assert dist_df.height == 3
    assert "bin_range" in dist_df.columns
    assert "total_pairs" in dist_df.columns
    assert "positives" in dist_df.columns
    assert "negatives" in dist_df.columns
    assert "empirical_precision" in dist_df.columns

    # Check sum of pairs
    assert dist_df["total_pairs"].sum() == 5
    assert dist_df["positives"].sum() == 2
    assert dist_df["negatives"].sum() == 3

    # Check top bin [0.80, 1.00]: contains 0.95 (pos), 0.85 (pos), 0.88 (neg) -> 2 pos, 1 neg -> precision = 2/3 = 0.6667
    top_bin = dist_df.filter(pl.col("bin_range") == "[0.80, 1.00]").to_dicts()[0]
    assert top_bin["positives"] == 2
    assert top_bin["negatives"] == 1
    assert top_bin["empirical_precision"] == pytest.approx(2 / 3, abs=1e-3)
