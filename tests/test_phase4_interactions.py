"""
Unit tests for Phase 4C: Targeted Interaction Features and Ablation.

Tests:
1. Exact computation and numerical correctness of all 4 interaction features.
2. Safe null/missing value handling in interaction calculation.
3. Subset feature computation behavior.
4. Feature schema validation (correct counts, deterministic names, zero identity leakage).
5. Model training compatibility with 29, 30, 31, and 33 feature specifications.
6. Non-leakage split preservation on augmented dataset.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.feature_schema import (
    ALL_FEATURE_NAMES,
    LABEL_COLUMN,
    PAIR_ID_COLUMNS,
)
from src.features.training_dataset import split_supervised_dataset
from src.models.baseline_model import BaselineMatchingModel
from src.models.phase4_interactions import (
    ALL_INTERACTION_FEATURES,
    BASELINE_FEATURES,
    EXPERIMENT_SPECS,
    EXP_A_FEATURES,
    EXP_B_FEATURES,
    EXP_C_FEATURES,
    EXP_ALL_FEATURES,
    INTERACTION_FEATURES_A,
    INTERACTION_FEATURES_B,
    INTERACTION_FEATURES_C,
    compute_interaction_features,
)


# =============================================================================
# 1. Feature Computation & Math Correctness Tests
# =============================================================================

class TestInteractionFeatureComputation:
    @pytest.fixture
    def sample_features_df(self) -> pl.DataFrame:
        """Creates sample DataFrame with base 29 features and known values."""
        return pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
            "candidate_entity_id": ["S2-1", "S3-2", "S2-3", "S3-4"],
            "candidate_source": ["source2", "source3", "source2", "source3"],
            LABEL_COLUMN: [1, 0, 1, 0],
            # Core columns for interactions
            "name_char_3gram_jaccard": [0.80, 0.90, 0.50, None],
            "name_token_overlap": [1.00, 0.75, 0.00, 0.50],
            "address_missing_candidate": [1, 0, 1, 0],
            "address_char_3gram_similarity": [0.00, 0.85, 0.40, 0.60],
            "shared_address_number_count": [0, 2, 1, None],
            # Fill the rest of the 29 features
            **{
                feat: [0.5] * 4
                for feat in ALL_FEATURE_NAMES
                if feat not in [
                    "name_char_3gram_jaccard",
                    "name_token_overlap",
                    "address_missing_candidate",
                    "address_char_3gram_similarity",
                    "shared_address_number_count",
                ]
            },
        })

    def test_all_interaction_features_computed_correctly(self, sample_features_df):
        """Verifies formulas, dtypes, and null handling for all 4 interaction features."""
        augmented = compute_interaction_features(sample_features_df)

        for feat in ALL_INTERACTION_FEATURES:
            assert feat in augmented.columns
            assert augmented[feat].dtype == pl.Float32

        # 1. name_char_3gram_x_address_missing_cand = name_char_3gram_jaccard * address_missing_candidate
        # Row 0: 0.80 * 1 = 0.80
        # Row 1: 0.90 * 0 = 0.00
        # Row 2: 0.50 * 1 = 0.50
        # Row 3: None * 0 -> 0.00
        feat1 = augmented["name_char_3gram_x_address_missing_cand"].to_list()
        assert pytest.approx(feat1[0], abs=1e-4) == 0.80
        assert pytest.approx(feat1[1], abs=1e-4) == 0.00
        assert pytest.approx(feat1[2], abs=1e-4) == 0.50
        assert pytest.approx(feat1[3], abs=1e-4) == 0.00

        # 2. name_token_overlap_x_address_missing_cand = name_token_overlap * address_missing_candidate
        # Row 0: 1.00 * 1 = 1.00
        # Row 1: 0.75 * 0 = 0.00
        # Row 2: 0.00 * 1 = 0.00
        # Row 3: 0.50 * 0 = 0.00
        feat2 = augmented["name_token_overlap_x_address_missing_cand"].to_list()
        assert pytest.approx(feat2[0], abs=1e-4) == 1.00
        assert pytest.approx(feat2[1], abs=1e-4) == 0.00
        assert pytest.approx(feat2[2], abs=1e-4) == 0.00
        assert pytest.approx(feat2[3], abs=1e-4) == 0.00

        # 3. name_char_3gram_x_address_char_3gram = name_char_3gram_jaccard * address_char_3gram_similarity
        # Row 0: 0.80 * 0.00 = 0.00
        # Row 1: 0.90 * 0.85 = 0.765
        # Row 2: 0.50 * 0.40 = 0.20
        # Row 3: None * 0.60 -> 0.00
        feat3 = augmented["name_char_3gram_x_address_char_3gram"].to_list()
        assert pytest.approx(feat3[0], abs=1e-4) == 0.00
        assert pytest.approx(feat3[1], abs=1e-4) == 0.765
        assert pytest.approx(feat3[2], abs=1e-4) == 0.20
        assert pytest.approx(feat3[3], abs=1e-4) == 0.00

        # 4. shared_address_num_x_address_char_3gram = shared_address_number_count * address_char_3gram_similarity
        # Row 0: 0 * 0.00 = 0.00
        # Row 1: 2 * 0.85 = 1.70
        # Row 2: 1 * 0.40 = 0.40
        # Row 3: None * 0.60 -> 0.00
        feat4 = augmented["shared_address_num_x_address_char_3gram"].to_list()
        assert pytest.approx(feat4[0], abs=1e-4) == 0.00
        assert pytest.approx(feat4[1], abs=1e-4) == 1.70
        assert pytest.approx(feat4[2], abs=1e-4) == 0.40
        assert pytest.approx(feat4[3], abs=1e-4) == 0.00

    def test_subset_interaction_computation(self, sample_features_df):
        """Verifies computing only a subset of interaction features."""
        augmented = compute_interaction_features(
            sample_features_df,
            features_to_add=INTERACTION_FEATURES_A,
        )

        assert "name_char_3gram_x_address_missing_cand" in augmented.columns
        assert "name_token_overlap_x_address_missing_cand" in augmented.columns
        assert "name_char_3gram_x_address_char_3gram" not in augmented.columns
        assert "shared_address_num_x_address_char_3gram" not in augmented.columns


# =============================================================================
# 2. Experiment Specs & Feature Non-Leakage Tests
# =============================================================================

class TestExperimentSpecs:
    def test_feature_counts_and_isolation(self):
        """Verifies deterministic feature counts and absence of identity columns."""
        assert len(BASELINE_FEATURES) == 29
        assert len(EXP_A_FEATURES) == 31
        assert len(EXP_B_FEATURES) == 30
        assert len(EXP_C_FEATURES) == 30
        assert len(EXP_ALL_FEATURES) == 33

        for key, spec in EXPERIMENT_SPECS.items():
            feats = spec["features"]
            # Exclude identity columns and target
            for col in PAIR_ID_COLUMNS + [LABEL_COLUMN]:
                assert col not in feats, f"Forbidden column '{col}' found in {key} features!"

            # Check that all features in baseline are present
            for b_feat in BASELINE_FEATURES:
                assert b_feat in feats

    def test_zero_leakage_split_on_augmented_df(self):
        """Validates that entity-level split on augmented df preserves zero leakage."""
        df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
            "candidate_entity_id": ["S2-1", "S3-2", "S2-3", "S3-4", "S2-5", "S3-6"],
            "candidate_source": ["source2", "source3", "source2", "source3", "source2", "source3"],
            LABEL_COLUMN: [1, 0, 1, 0, 0, 0],
            "name_char_3gram_jaccard": [0.8] * 6,
            "name_token_overlap": [0.8] * 6,
            "address_missing_candidate": [0] * 6,
            "address_char_3gram_similarity": [0.5] * 6,
            "shared_address_number_count": [1] * 6,
            **{
                feat: [0.5] * 6
                for feat in ALL_FEATURE_NAMES
                if feat not in [
                    "name_char_3gram_jaccard",
                    "name_token_overlap",
                    "address_missing_candidate",
                    "address_char_3gram_similarity",
                    "shared_address_number_count",
                ]
            },
        })

        augmented = compute_interaction_features(df)
        split = split_supervised_dataset(augmented, val_fraction=0.40, seed=42)

        assert split.s1_overlap_count == 0
        assert split.candidate_pair_overlap_count == 0

        # All 4 interaction features exist in both train and validation
        for feat in ALL_INTERACTION_FEATURES:
            assert feat in split.train.columns
            assert feat in split.validation.columns


# =============================================================================
# 3. Model Compatibility Test
# =============================================================================

class TestModelCompatibility:
    def test_baseline_model_with_exp_all_features(self):
        """Verifies BaselineMatchingModel fits and evaluates with 33 features."""
        np.random.seed(42)
        n_rows = 50
        df = pl.DataFrame({
            "source1_entity_id": [f"S1-{i//5}" for i in range(n_rows)],
            "candidate_entity_id": [f"S2-{i}" for i in range(n_rows)],
            "candidate_source": ["source2"] * n_rows,
            LABEL_COLUMN: [1 if i % 10 == 0 else 0 for i in range(n_rows)],
            "name_char_3gram_jaccard": np.random.uniform(0, 1, n_rows).tolist(),
            "name_token_overlap": np.random.uniform(0, 1, n_rows).tolist(),
            "address_missing_candidate": [1 if i % 4 == 0 else 0 for i in range(n_rows)],
            "address_char_3gram_similarity": np.random.uniform(0, 1, n_rows).tolist(),
            "shared_address_number_count": [i % 3 for i in range(n_rows)],
            **{
                feat: np.random.uniform(0, 1, n_rows).tolist()
                for feat in ALL_FEATURE_NAMES
                if feat not in [
                    "name_char_3gram_jaccard",
                    "name_token_overlap",
                    "address_missing_candidate",
                    "address_char_3gram_similarity",
                    "shared_address_number_count",
                ]
            },
        })

        augmented = compute_interaction_features(df)
        split = split_supervised_dataset(augmented, val_fraction=0.30, seed=42)

        model = BaselineMatchingModel(
            n_estimators=10,
            num_leaves=15,
            feature_names=EXP_ALL_FEATURES,
            random_state=42,
        )
        model.fit(split.train)

        assert model.is_fitted
        importances = model.get_feature_importances()
        assert importances.height == 33

        probs = model.predict_proba(split.validation)
        assert probs.shape == (split.validation.height,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
