"""
Unit and integration tests for Part 9: Threshold optimization and match post-processing.

Tests:
1. Leakage-safe threshold selection (strictly uses training data, never touches val)
2. Threshold optimization maximizes F0.5
3. Zero-match entities preservation (S1 with no candidates above threshold receives 0 matches)
4. Multi-match entities preservation (multiple candidates exceeding threshold are preserved)
5. Source separation (source2 and source3 handled independently)
6. Post-processing filtering (pure threshold, optional per-source top-K, score margin)
7. Submission linkage formatting (empty string for zero matches, comma-joined for multi matches)
8. Deterministic reproducibility
9. Prediction diagnostics and error analysis computation
"""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    LABEL_COLUMN,
    LABEL_DTYPE,
    ALL_FEATURE_NAMES,
)
from src.features.training_dataset import split_supervised_dataset
from src.models.baseline_model import BaselineMatchingModel
from src.models.post_processing import (
    ThresholdOptimizationResult,
    optimize_threshold_oof,
    MatchPostProcessor,
    compute_prediction_diagnostics,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_labeled_dataset() -> pl.DataFrame:
    """Creates a synthetic multi-entity dataset with 29 features."""
    np.random.seed(42)
    rows = []
    # 30 entities
    for i in range(30):
        s1_id = f"S1-{i:03d}"
        # Entity 0-10: positive in source2
        # Entity 11-15: positive in source2 AND positive in source3 (multi-match!)
        # Entity 16-29: 0 positives (zero-match!)
        if i <= 10:
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S2-{i:03d}",
                "candidate_source": "source2",
                "label": 1,
                "name_token_jaccard": 0.95,
                "name_char_3gram_jaccard": 0.90,
                "address_token_jaccard": 0.85,
                "address_char_3gram_similarity": 0.92,
                "shared_address_number_count": 2,
                "country_match": 1,
                "matched_key_count": 3,
            })
        elif 11 <= i <= 15:
            # Multi-match
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S2-{i:03d}",
                "candidate_source": "source2",
                "label": 1,
                "name_token_jaccard": 0.92,
                "name_char_3gram_jaccard": 0.88,
                "address_token_jaccard": 0.80,
                "address_char_3gram_similarity": 0.85,
                "shared_address_number_count": 1,
                "country_match": 1,
                "matched_key_count": 2,
            })
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S3-{i:03d}",
                "candidate_source": "source3",
                "label": 1,
                "name_token_jaccard": 0.90,
                "name_char_3gram_jaccard": 0.85,
                "address_token_jaccard": 0.78,
                "address_char_3gram_similarity": 0.82,
                "shared_address_number_count": 1,
                "country_match": 1,
                "matched_key_count": 2,
            })

        # Add 3 negative candidates per S1 entity
        for j in range(1, 4):
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S2-NEG-{i}-{j}",
                "candidate_source": "source2",
                "label": 0,
                "name_token_jaccard": float(np.random.uniform(0.0, 0.2)),
                "name_char_3gram_jaccard": float(np.random.uniform(0.0, 0.2)),
                "address_token_jaccard": float(np.random.uniform(0.0, 0.1)),
                "address_char_3gram_similarity": float(np.random.uniform(0.0, 0.1)),
                "shared_address_number_count": 0,
                "country_match": 1,
                "matched_key_count": 1,
            })

    df = pl.DataFrame(rows)
    for feat in ALL_FEATURE_NAMES:
        if feat not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(feat))
    return df.with_columns(pl.col("label").cast(LABEL_DTYPE))


# =============================================================================
# 1. Leakage-Safe Threshold Selection Tests
# =============================================================================

class TestThresholdOptimization:
    def test_threshold_selection_no_leakage(self, synthetic_labeled_dataset):
        """Threshold optimization runs purely on training set, never receiving validation data."""
        split = split_supervised_dataset(synthetic_labeled_dataset, val_fraction=0.25, seed=42)

        # Optimize threshold using training data only
        res = optimize_threshold_oof(
            train_df=split.train,
            n_splits=3,
            seed=42,
            model_params={"n_estimators": 25, "learning_rate": 0.1},
        )

        assert 0.05 <= res.best_threshold <= 0.95
        assert res.best_f05 > 0.0
        assert res.oof_pr_auc > 0.0
        assert len(res.threshold_sweep) > 0

        # Evaluate the chosen threshold once on untouched validation set
        model = BaselineMatchingModel(n_estimators=25, learning_rate=0.1, random_state=42)
        model.fit(split.train)
        val_probs = model.predict_proba(split.validation)

        from src.models.metrics import evaluate_predictions_at_threshold
        val_eval = evaluate_predictions_at_threshold(
            y_true=split.validation["label"].to_numpy(),
            y_prob=val_probs,
            threshold=res.best_threshold,
        )
        assert val_eval.threshold == res.best_threshold
        assert val_eval.f05 > 0.0

    def test_deterministic_threshold_optimization(self, synthetic_labeled_dataset):
        """Identical random seed produces identical selected threshold and metrics."""
        res1 = optimize_threshold_oof(
            train_df=synthetic_labeled_dataset,
            n_splits=3,
            seed=123,
            model_params={"n_estimators": 15},
        )
        res2 = optimize_threshold_oof(
            train_df=synthetic_labeled_dataset,
            n_splits=3,
            seed=123,
            model_params={"n_estimators": 15},
        )
        assert res1.best_threshold == res2.best_threshold
        assert abs(res1.best_f05 - res2.best_f05) < 1e-6


# =============================================================================
# 2. Match Post-Processing Tests
# =============================================================================

class TestMatchPostProcessor:
    @pytest.fixture
    def mock_pairs_with_probs(self) -> pl.DataFrame:
        """Sample candidate pairs with predicted probabilities."""
        return pl.DataFrame({
            "source1_entity_id": [
                "S1-ZERO", "S1-ZERO",
                "S1-SINGLE", "S1-SINGLE",
                "S1-MULTI", "S1-MULTI", "S1-MULTI",
            ],
            "candidate_entity_id": [
                "S2-Z1", "S2-Z2",
                "S2-S1", "S2-S2",
                "S2-M1", "S3-M2", "S2-M3",
            ],
            "candidate_source": [
                "source2", "source2",
                "source2", "source2",
                "source2", "source3", "source2",
            ],
            "probability": [
                0.10, 0.25,        # S1-ZERO has no candidate above 0.50
                0.92, 0.30,        # S1-SINGLE has 1 candidate above 0.50
                0.88, 0.85, 0.60,  # S1-MULTI has 3 candidates above 0.50 across S2 and S3
            ],
        })

    def test_zero_match_entities_preserved(self, mock_pairs_with_probs):
        """Entities with no candidate exceeding threshold receive zero predicted matches."""
        processor = MatchPostProcessor(threshold=0.50)
        filtered = processor.filter_candidate_pairs(mock_pairs_with_probs)

        # S1-ZERO must NOT be in accepted pairs
        assert "S1-ZERO" not in filtered["source1_entity_id"].to_list()

        # In submission formatting, S1-ZERO receives an empty string ""
        all_s1 = ["S1-ZERO", "S1-SINGLE", "S1-MULTI"]
        sub = processor.format_submission_linkages(filtered, all_s1)
        zero_row = sub.filter(pl.col("source1_entity_id") == "S1-ZERO").to_dicts()[0]
        assert zero_row["matched_entity_ids"] == ""

    def test_multi_match_entities_preserved_without_global_top1_forcing(self, mock_pairs_with_probs):
        """Multiple candidates exceeding threshold are preserved without global top-1 forcing."""
        processor = MatchPostProcessor(threshold=0.50)
        filtered = processor.filter_candidate_pairs(mock_pairs_with_probs)

        multi_rows = filtered.filter(pl.col("source1_entity_id") == "S1-MULTI")
        # All 3 candidates (0.88, 0.85, 0.60) exceed 0.50
        assert multi_rows.height == 3
        cand_ids = multi_rows["candidate_entity_id"].to_list()
        assert set(cand_ids) == {"S2-M1", "S3-M2", "S2-M3"}

        # In submission format, multi-matches are comma-separated
        all_s1 = ["S1-ZERO", "S1-SINGLE", "S1-MULTI"]
        sub = processor.format_submission_linkages(filtered, all_s1)
        multi_sub = sub.filter(pl.col("source1_entity_id") == "S1-MULTI").to_dicts()[0]
        assert "S2-M1" in multi_sub["matched_entity_ids"]
        assert "S3-M2" in multi_sub["matched_entity_ids"]
        assert "S2-M3" in multi_sub["matched_entity_ids"]

    def test_per_source_candidate_separation(self, mock_pairs_with_probs):
        """Optional per-source top-K allows at most K per source without confusing sources."""
        # max_matches_per_source = 1 allows at most 1 from source2 and 1 from source3
        processor = MatchPostProcessor(threshold=0.50, max_matches_per_source=1)
        filtered = processor.filter_candidate_pairs(mock_pairs_with_probs)

        multi_rows = filtered.filter(pl.col("source1_entity_id") == "S1-MULTI")
        # In S2, S2-M1 (0.88) wins over S2-M3 (0.60). In S3, S3-M2 (0.85) is retained.
        assert multi_rows.height == 2
        sources = dict(zip(multi_rows["candidate_source"].to_list(), multi_rows["candidate_entity_id"].to_list()))
        assert sources["source2"] == "S2-M1"
        assert sources["source3"] == "S3-M2"

    def test_score_margin_filtering(self, mock_pairs_with_probs):
        """Score margin rejects candidates that trail the top candidate by more than margin."""
        # Top prob for S1-MULTI is 0.88. Margin 0.15 rejects 0.60 (diff=0.28) but keeps 0.85 (diff=0.03).
        processor = MatchPostProcessor(threshold=0.50, score_margin=0.15)
        filtered = processor.filter_candidate_pairs(mock_pairs_with_probs)

        multi_rows = filtered.filter(pl.col("source1_entity_id") == "S1-MULTI")
        assert multi_rows.height == 2
        assert set(multi_rows["candidate_entity_id"].to_list()) == {"S2-M1", "S3-M2"}


# =============================================================================
# 3. Prediction Diagnostics Tests
# =============================================================================

class TestPredictionDiagnostics:
    def test_diagnostics_calculation(self):
        """Diagnostics correctly tally zero/multi-match counts, probability distributions, and errors."""
        val_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S3-1", "S2-2", "S2-3"],
            "candidate_source": ["source2", "source3", "source2", "source2"],
            "label": [1, 0, 1, 0],
        }).with_columns(pl.col("label").cast(LABEL_DTYPE))

        # S1-1: S2-1 (0.90, true=1 -> TP), S3-1 (0.80, true=0 -> FP) -> S1-1 has 2 matches (multi-match)
        # S1-2: S2-2 (0.30, true=1 -> FN) -> S1-2 has 0 matches (zero-match)
        # S1-3: S2-3 (0.05, true=0 -> TN) -> S1-3 has 0 matches (zero-match)
        probs = np.array([0.90, 0.80, 0.30, 0.05])

        diag = compute_prediction_diagnostics(val_df, probs, threshold=0.50)

        assert diag["total_s1_entities"] == 3
        assert diag["zero_match_s1_count"] == 2  # S1-2 and S1-3
        assert diag["multi_match_s1_count"] == 1  # S1-1
        assert diag["single_match_s1_count"] == 0

        # Errors
        assert diag["false_positive_count"] == 1  # S3-1
        assert diag["false_negative_count"] == 1  # S2-2
        assert diag["false_positives"][0]["candidate_entity_id"] == "S3-1"
        assert diag["false_negatives"][0]["candidate_entity_id"] == "S2-2"

        # Prob distributions
        assert diag["accepted_prob_stats"]["min"] == 0.80
        assert diag["accepted_prob_stats"]["max"] == 0.90
        assert diag["rejected_prob_stats"]["min"] == 0.05
        assert diag["rejected_prob_stats"]["max"] == 0.30
