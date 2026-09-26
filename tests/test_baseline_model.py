"""
Unit and integration tests for Part 8: Baseline supervised matching model.

Tests:
1. Model input feature selection (exactly 29 engineered features)
2. No identity columns passed to model (raises ValueError if attempted)
3. Train/validation entity-level separation in model training
4. Deterministic training across identical random seeds
5. Probability output shape and [0.0, 1.0] range
6. F0.5 calculation accuracy and edge-case behavior
7. Handling highly imbalanced labels (0.2% positive rate)
8. Evaluation across thresholds and PR-AUC calculation
9. Feature importance extraction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
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
from src.models.metrics import (
    compute_f_beta,
    evaluate_predictions_at_threshold,
    evaluate_matching_probabilities,
)
from src.models.baseline_model import (
    BaselineMatchingModel,
    NON_FEATURE_COLUMNS,
)


# =============================================================================
# 1. F0.5 Metric Calculation Tests
# =============================================================================

class TestF05Calculation:
    def test_f05_mathematical_accuracy(self):
        """F0.5 = 1.25 * P * R / (0.25 * P + R) weights precision twice as much as recall."""
        # TP=8, FP=2, FN=2, TN=8 -> Precision=0.8, Recall=0.8 -> F0.5=0.8
        y_true = [1] * 10 + [0] * 10
        y_pred = [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8

        f05 = compute_f_beta(y_true, y_pred, beta=0.5)
        # 1.25 * 0.8 * 0.8 / (0.25 * 0.8 + 0.8) = 0.8 / 1.0 = 0.8
        assert abs(f05 - 0.8) < 1e-4

        # High precision (P=1.0, R=0.5):
        # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
        y_pred_high_prec = [1] * 5 + [0] * 5 + [0] * 10
        f05_high_prec = compute_f_beta(y_true, y_pred_high_prec, beta=0.5)
        assert abs(f05_high_prec - (0.625 / 0.75)) < 1e-4

        # High recall (P=0.5, R=1.0):
        # F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555
        # Confirms F0.5 strongly penalizes lower precision
        y_pred_high_rec = [1] * 10 + [1] * 10
        f05_high_rec = compute_f_beta(y_true, y_pred_high_rec, beta=0.5)
        assert abs(f05_high_rec - (0.625 / 1.125)) < 1e-4
        assert f05_high_prec > f05_high_rec

    def test_f05_edge_cases_zero_division(self):
        """Zero predicted positives or zero true positives safely return 0.0."""
        assert compute_f_beta([0, 0, 0], [0, 0, 0], beta=0.5) == 0.0
        assert compute_f_beta([1, 1, 1], [0, 0, 0], beta=0.5) == 0.0
        assert compute_f_beta([0, 0, 0], [1, 1, 1], beta=0.5) == 0.0


# =============================================================================
# 2. Model Input Feature Selection & Identity Column Exclusion Tests
# =============================================================================

class TestFeatureSelection:
    @pytest.fixture
    def mock_labeled_data(self) -> pl.DataFrame:
        """Creates a mock DataFrame containing all 32 columns (3 ID + 29 features) + label."""
        n_samples = 50
        data = {
            "source1_entity_id": [f"S1-{i % 10:03d}" for i in range(n_samples)],
            "candidate_entity_id": [f"S2-{i:03d}" for i in range(n_samples)],
            "candidate_source": ["source2" if i % 2 == 0 else "source3" for i in range(n_samples)],
        }
        for feat in ALL_FEATURE_NAMES:
            if "exact" in feat or "match" in feat or "missing" in feat or feat.startswith("matched_key"):
                data[feat] = [int(i % 3 == 0) for i in range(n_samples)]
            else:
                data[feat] = [float(i / n_samples) for i in range(n_samples)]

        data[LABEL_COLUMN] = [1 if i % 5 == 0 else 0 for i in range(n_samples)]
        return pl.DataFrame(data).with_columns(pl.col(LABEL_COLUMN).cast(LABEL_DTYPE))

    def test_exact_29_features_selected(self, mock_labeled_data):
        """Model extracts exactly the 29 engineered features in deterministic order."""
        model = BaselineMatchingModel(n_estimators=10)
        X, y = model.prepare_features(mock_labeled_data)

        assert X.shape == (mock_labeled_data.height, 29)
        assert y is not None and len(y) == mock_labeled_data.height
        assert model.feature_names == ALL_FEATURE_NAMES

    def test_no_identity_columns_passed_to_model(self):
        """Attempting to include identity columns in feature_names raises ValueError."""
        for forbidden in ["source1_entity_id", "candidate_entity_id", "candidate_source", "label"]:
            with pytest.raises(ValueError, match="cannot be included in model features"):
                BaselineMatchingModel(feature_names=[forbidden, "name_token_jaccard"])

    def test_missing_features_raises_error(self):
        """Passing DataFrame missing a required engineered feature raises ValueError."""
        model = BaselineMatchingModel(feature_names=["name_token_jaccard", "address_exact"])
        incomplete_df = pl.DataFrame({"name_token_jaccard": [0.5]})
        with pytest.raises(ValueError, match="missing required feature columns"):
            model.prepare_features(incomplete_df)


# =============================================================================
# 3. Model Training, Determinism & Output Probability Tests
# =============================================================================

class TestModelTrainingAndOutputs:
    @pytest.fixture
    def synthetic_train_val(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Creates entity-separated train and validation datasets with 29 features."""
        np.random.seed(42)
        rows = []
        for i in range(60):
            s1_id = f"S1-{i:03d}"
            # 1 positive and 4 negatives per entity
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"C-POS-{i}",
                "candidate_source": "source2",
                "label": 1,
                "name_token_jaccard": float(np.random.uniform(0.7, 1.0)),
                "name_char_3gram_jaccard": float(np.random.uniform(0.6, 1.0)),
                "address_token_jaccard": float(np.random.uniform(0.5, 1.0)),
                "country_match": 1,
                "matched_key_A": 1,
                "matched_key_count": 3,
            })
            for j in range(4):
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": f"C-NEG-{i}-{j}",
                    "candidate_source": "source2",
                    "label": 0,
                    "name_token_jaccard": float(np.random.uniform(0.0, 0.4)),
                    "name_char_3gram_jaccard": float(np.random.uniform(0.0, 0.3)),
                    "address_token_jaccard": float(np.random.uniform(0.0, 0.2)),
                    "country_match": 1,
                    "matched_key_A": 0,
                    "matched_key_count": 1,
                })

        # Add remaining features filled with 0
        df = pl.DataFrame(rows)
        for feat in ALL_FEATURE_NAMES:
            if feat not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(feat))

        df = df.with_columns(pl.col("label").cast(LABEL_DTYPE))

        # Split at entity level
        split = split_supervised_dataset(df, val_fraction=0.25, seed=42)
        return split.train, split.validation

    def test_train_validation_entity_separation(self, synthetic_train_val):
        """Train and validation S1 entities are completely disjoint."""
        train_df, val_df = synthetic_train_val
        train_s1 = set(train_df["source1_entity_id"].unique().to_list())
        val_s1 = set(val_df["source1_entity_id"].unique().to_list())
        assert len(train_s1 & val_s1) == 0

    def test_deterministic_training(self, synthetic_train_val):
        """Same random seed produces identical probabilities on validation data."""
        train_df, val_df = synthetic_train_val

        model1 = BaselineMatchingModel(n_estimators=30, learning_rate=0.1, random_state=42)
        model1.fit(train_df)
        probs1 = model1.predict_proba(val_df)

        model2 = BaselineMatchingModel(n_estimators=30, learning_rate=0.1, random_state=42)
        model2.fit(train_df)
        probs2 = model2.predict_proba(val_df)

        np.testing.assert_allclose(probs1, probs2, rtol=1e-6, atol=1e-6)

    def test_probability_output_shape_and_range(self, synthetic_train_val):
        """Probability outputs strictly match row count and lie within [0.0, 1.0]."""
        train_df, val_df = synthetic_train_val

        model = BaselineMatchingModel(n_estimators=30, learning_rate=0.1, random_state=42)
        model.fit(train_df)
        probs = model.predict_proba(val_df)

        assert len(probs) == val_df.height
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        # Separable data should have high probability for positives
        val_pos_idx = np.where(val_df["label"].to_numpy() == 1)[0]
        val_neg_idx = np.where(val_df["label"].to_numpy() == 0)[0]
        assert np.mean(probs[val_pos_idx]) > np.mean(probs[val_neg_idx])

    def test_handles_extreme_class_imbalance(self):
        """Model trains smoothly on 1:500 extreme class imbalance without crashing."""
        n_pos = 10
        n_neg = 5000  # 1:500 ratio
        rows = []
        for i in range(n_pos):
            rows.append({
                "source1_entity_id": f"S1-POS-{i}",
                "candidate_entity_id": f"C-POS-{i}",
                "candidate_source": "source2",
                "label": 1,
                "name_token_jaccard": 0.9,
                "name_char_3gram_jaccard": 0.85,
            })
        for i in range(n_neg):
            rows.append({
                "source1_entity_id": f"S1-NEG-{i % 500}",
                "candidate_entity_id": f"C-NEG-{i}",
                "candidate_source": "source2",
                "label": 0,
                "name_token_jaccard": 0.1,
                "name_char_3gram_jaccard": 0.05,
            })

        df = pl.DataFrame(rows)
        for feat in ALL_FEATURE_NAMES:
            if feat not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(feat))
        df = df.with_columns(pl.col("label").cast(LABEL_DTYPE))

        model = BaselineMatchingModel(n_estimators=20, random_state=42)
        model.fit(df)

        probs = model.predict_proba(df)
        assert len(probs) == n_pos + n_neg
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_feature_importance_extraction(self, synthetic_train_val):
        """Feature importance returns all 29 features sorted descending by gain."""
        train_df, _ = synthetic_train_val
        model = BaselineMatchingModel(n_estimators=30, random_state=42)
        model.fit(train_df)

        fi_df = model.get_feature_importances()
        assert fi_df.height == 29
        assert fi_df.columns == ["feature_name", "importance", "relative_pct"]
        # Must be sorted descending
        imps = fi_df["importance"].to_list()
        assert imps == sorted(imps, reverse=True)


# =============================================================================
# 4. Evaluation Across Thresholds Tests
# =============================================================================

class TestThresholdEvaluation:
    def test_threshold_sweep_report(self):
        """Threshold sweep computes metrics without premature threshold locking."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.95, 0.85, 0.45, 0.60, 0.30, 0.10, 0.05, 0.02])

        report = evaluate_matching_probabilities(
            y_true=y_true,
            y_prob=y_prob,
            thresholds=[0.3, 0.5, 0.8],
        )

        assert report.total_samples == 8
        assert report.positive_samples == 3
        assert report.negative_samples == 5
        assert report.pr_auc > 0.0
        assert len(report.threshold_evaluations) == 3

        # At high threshold (0.8), precision should be 1.0
        te_08 = [te for te in report.threshold_evaluations if te.threshold == 0.8][0]
        assert te_08.precision == 1.0
        assert te_08.true_positives == 2
        assert te_08.false_positives == 0
        assert te_08.f05 > 0.0
