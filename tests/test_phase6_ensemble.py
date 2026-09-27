"""
Unit tests for Phase 6 Multi-Model Ensembling framework.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.ensemble import (
    EnsembleDefinition,
    STANDARD_ENSEMBLES,
    blend_rank_average,
    blend_weighted_probabilities,
    compute_ensemble_error_exchange,
    evaluate_ensemble,
    rank_transform,
)


def test_blend_weighted_probabilities():
    preds = {
        "m1": np.array([0.2, 0.8, 0.5]),
        "m2": np.array([0.4, 0.6, 0.9]),
    }
    # Equal blend
    blended = blend_weighted_probabilities(preds, {"m1": 0.5, "m2": 0.5})
    np.testing.assert_allclose(blended, [0.3, 0.7, 0.7])

    # Unnormalized weights (3:1 -> 0.75 : 0.25)
    blended_weighted = blend_weighted_probabilities(preds, {"m1": 3.0, "m2": 1.0})
    expected = 0.75 * preds["m1"] + 0.25 * preds["m2"]
    np.testing.assert_allclose(blended_weighted, expected)

    # Missing model error
    with pytest.raises(KeyError, match="Model 'm3' specified in weights not found"):
        blend_weighted_probabilities(preds, {"m1": 0.5, "m3": 0.5})

    # Non-positive weight error
    with pytest.raises(ValueError, match="Sum of blend weights must be > 0"):
        blend_weighted_probabilities(preds, {"m1": 0.0, "m2": 0.0})

    # Length mismatch error
    bad_preds = {"m1": np.array([0.1, 0.2]), "m2": np.array([0.1, 0.2, 0.3])}
    with pytest.raises(ValueError, match="Prediction length mismatch"):
        blend_weighted_probabilities(bad_preds, {"m1": 0.5, "m2": 0.5})


def test_rank_transform_and_blend_rank_average():
    probs = np.array([0.1, 0.9, 0.4, 0.7])
    ranks = rank_transform(probs)

    # Lowest (0.1) should be 0.0, highest (0.9) should be 1.0
    assert ranks[0] == 0.0
    assert ranks[1] == 1.0
    assert 0.0 < ranks[2] < ranks[3] < 1.0

    preds = {
        "m1": np.array([0.1, 0.9, 0.4]),
        "m2": np.array([0.2, 0.8, 0.5]),
    }
    blended_rank = blend_rank_average(preds)
    assert len(blended_rank) == 3
    assert 0.0 <= blended_rank.min() and blended_rank.max() <= 1.0
    # Both models rank index 1 highest and index 0 lowest
    assert blended_rank[1] > blended_rank[2] > blended_rank[0]


def test_evaluate_ensemble_smoke():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=int)
    fold_assignments = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)

    preds = {
        "lightgbm": np.array([0.95, 0.88, 0.12, 0.05, 0.91, 0.10, 0.02, 0.85, 0.30, 0.01]),
        "xgboost":  np.array([0.92, 0.85, 0.15, 0.04, 0.89, 0.08, 0.01, 0.88, 0.25, 0.02]),
    }

    definition = EnsembleDefinition(
        name="test_blend",
        description="Test 50/50 blend",
        method="probability_blend",
        weights={"lightgbm": 0.5, "xgboost": 0.5},
    )

    res = evaluate_ensemble(
        definition=definition,
        predictions_map=preds,
        y_true=y_true,
        fold_assignments=fold_assignments,
        thresholds=(0.80, 0.90),
    )

    assert res.definition.name == "test_blend"
    assert 0.0 <= res.oof_pr_auc <= 1.0
    assert 0.0 <= res.oof_roc_auc <= 1.0
    assert 0.80 in res.metrics_at_thresholds
    assert 0.90 in res.metrics_at_thresholds
    assert len(res.fold_metrics) == 2
    assert 0.0 <= res.mean_f05_90 <= 1.0
    assert res.optimal_threshold > 0.0


def test_compute_ensemble_error_exchange():
    # 5 candidate pairs:
    # 0: GT=1, LGB=0.95 (TP), Ens=0.95 (TP) -> correct both
    # 1: GT=1, LGB=0.40 (FN), Ens=0.92 (TP) -> corrected FN
    # 2: GT=1, LGB=0.92 (TP), Ens=0.30 (FN) -> introduced FN
    # 3: GT=0, LGB=0.95 (FP), Ens=0.20 (TN) -> corrected FP
    # 4: GT=0, LGB=0.10 (TN), Ens=0.95 (FP) -> introduced FP

    df = pl.DataFrame({
        "source1_entity_id": [f"S_{i}" for i in range(5)],
        "candidate_entity_id": [f"C_{i}" for i in range(5)],
        "candidate_source": ["source2"] * 5,
        "label": [1, 1, 1, 0, 0],
        "name_char_3gram_jaccard": [0.8] * 5,
        "address_char_3gram_similarity": [0.8] * 5,
        "address_missing_candidate": [0] * 5,
        "address_missing_s1": [0] * 5,
        "shared_address_number_count": [0] * 5,
        "name_exact_translit": [0] * 5,
        "name_char_len_ratio": [1.0] * 5,
        "name_acronym_match": [0] * 5,
        "name_first_token_exact": [0] * 5,
    })

    lgb_probs = np.array([0.95, 0.40, 0.92, 0.95, 0.10])
    ens_scores = np.array([0.95, 0.92, 0.30, 0.20, 0.95])

    exchange = compute_ensemble_error_exchange(
        df=df,
        lgb_probs=lgb_probs,
        ens_scores=ens_scores,
        ensemble_name="test_ens",
        threshold=0.90,
    )

    assert exchange.total_positives == 3
    assert exchange.total_negatives == 2
    assert exchange.lgb_fps == 1  # idx 3
    assert exchange.lgb_fns == 1  # idx 1
    assert exchange.ens_fps == 1  # idx 4
    assert exchange.ens_fns == 1  # idx 2

    assert exchange.corrected_fns == 1  # idx 1
    assert exchange.corrected_fps == 1  # idx 3
    assert exchange.introduced_fns == 1  # idx 2
    assert exchange.introduced_fps == 1  # idx 4
    assert exchange.net_error_change == 0

    assert exchange.error_overlap_df.height == 4  # idx 1, 2, 3, 4 are involved in errors
