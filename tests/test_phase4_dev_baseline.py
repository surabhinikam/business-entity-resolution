"""
Unit tests for Phase 4 Part 1: Full Development-Scale Baseline Model Training.

Tests:
1. Dataset and GT intersection validation (positive_candidate_pairs == GT pairs ∩ candidate_pairs).
2. Entity-disjoint train/validation split guarantees (zero S1 overlap, zero pair leakage).
3. Feature preparation excludes composite IDs and targets (zero leakage).
4. Threshold selection on continuous probabilities maximizing F0.5.
5. End-to-end baseline runner execution on mock feature set.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import polars as pl
import pytest

from scripts.run_phase4_dev_baseline import (
    find_optimal_threshold,
    generate_markdown_report,
    validate_dataset_and_ground_truth,
)
from src.features.feature_schema import (
    ALL_FEATURE_NAMES,
    LABEL_COLUMN,
    PAIR_ID_COLUMNS,
)
from src.features.training_dataset import (
    normalize_ground_truth,
    split_supervised_dataset,
)
from src.models.baseline_model import BaselineMatchingModel
from src.models.metrics import evaluate_predictions_at_threshold


# =============================================================================
# 1. Dataset & Ground-Truth Intersection Tests
# =============================================================================

class TestDatasetGTIntersection:
    def test_gt_intersection_verification(self):
        """Verifies that positive_candidate_pairs == GT pairs ∩ candidate_pairs."""
        cand_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-10", "S3-20", "S2-30", "S3-40"],
            "candidate_source": ["source2", "source3", "source2", "source3"],
            **{feat: [0.5, 0.6, 0.7, 0.8] for feat in ALL_FEATURE_NAMES},
        })

        # Ground truth has (S1-1, S3-20, source3) and (S1-2, S2-30, source2), plus an un-retrieved pair
        gt_norm = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-99"],
            "candidate_entity_id": ["S3-20", "S2-30", "S2-999"],
            "candidate_source": ["source3", "source2", "source2"],
        })

        info = validate_dataset_and_ground_truth(cand_df, gt_norm)

        assert info["total_pairs"] == 4
        assert info["num_pos"] == 2
        assert info["num_neg"] == 2
        assert info["is_verified"] is True
        assert info["gt_intersection_count"] == 2
        assert info["dup_cand_pairs"] == 0
        assert info["dup_pos_pairs"] == 0
        assert info["s2_pos"] == 1
        assert info["s3_pos"] == 1


# =============================================================================
# 2. Entity-Disjoint Split Guarantees
# =============================================================================

class TestEntityDisjointSplit:
    def test_zero_leakage_guarantees(self):
        """Validates that entity-level split guarantees zero S1 overlap and zero pair overlap."""
        cand_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
            "candidate_entity_id": ["S2-1", "S3-2", "S2-3", "S3-4", "S2-5", "S3-6"],
            "candidate_source": ["source2", "source3", "source2", "source3", "source2", "source3"],
            LABEL_COLUMN: [1, 0, 1, 0, 0, 0],
            **{feat: [0.5] * 6 for feat in ALL_FEATURE_NAMES},
        })

        split = split_supervised_dataset(
            cand_df,
            val_fraction=0.40,
            stratify_by_positive=True,
            seed=42,
        )

        assert split.s1_overlap_count == 0
        assert split.candidate_pair_overlap_count == 0

        train_s1 = set(split.train["source1_entity_id"].to_list())
        val_s1 = set(split.validation["source1_entity_id"].to_list())
        assert train_s1.isdisjoint(val_s1)


# =============================================================================
# 3. Model Non-Leakage & Threshold Selection
# =============================================================================

class TestModelAndThresholding:
    def test_feature_preparation_excludes_identity_columns(self):
        """Ensures non-feature columns are strictly excluded from prepared feature matrix."""
        df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "candidate_source": ["source2", "source2"],
            LABEL_COLUMN: [1, 0],
            **{feat: [0.8, 0.2] for feat in ALL_FEATURE_NAMES},
        })

        model = BaselineMatchingModel()
        X, y = model.prepare_features(df)

        assert X.shape == (2, len(ALL_FEATURE_NAMES))
        assert len(y) == 2

    def test_find_optimal_threshold(self):
        """Validates that find_optimal_threshold finds the threshold maximizing F0.5."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        # Probabilities where high threshold gives high precision
        y_prob = np.array([0.95, 0.90, 0.60, 0.85, 0.20, 0.10])

        best_eval = find_optimal_threshold(y_true, y_prob)

        assert best_eval.threshold >= 0.86
        assert best_eval.precision == 1.0  # At >=0.86, only true positives accepted
        assert best_eval.f05 > 0.0


# =============================================================================
# 4. Report Generation Smoke Test
# =============================================================================

class TestReportGeneration:
    def test_markdown_report_generation(self):
        """Smoke test verifying that generate_markdown_report generates valid markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test_report.md")

            cand_df = pl.DataFrame({
                "source1_entity_id": ["S1-1", "S1-2"],
                "candidate_entity_id": ["S2-1", "S2-2"],
                "candidate_source": ["source2", "source2"],
                LABEL_COLUMN: [1, 0],
                **{feat: [0.8, 0.2] for feat in ALL_FEATURE_NAMES},
            })
            split = split_supervised_dataset(cand_df, val_fraction=0.5, seed=42)

            ds_info = {
                "total_pairs": 2,
                "dup_cand_pairs": 0,
                "num_pos": 1,
                "num_neg": 1,
                "pos_rate": 50.0,
                "gt_intersection_count": 1,
                "is_verified": True,
                "dup_pos_pairs": 0,
                "unique_s1": 2,
                "s1_with_pos": 1,
                "s1_zero_pos": 1,
                "s2_pos": 1,
                "s3_pos": 0,
                "s2_total": 2,
                "s3_total": 0,
            }

            model = BaselineMatchingModel(n_estimators=10, random_state=42)
            model.fit(split.train)
            importances_df = model.get_feature_importances()

            eval_ref = evaluate_predictions_at_threshold(np.array([1]), np.array([0.8]), threshold=0.5)
            eval_p3 = evaluate_predictions_at_threshold(np.array([1]), np.array([0.8]), threshold=0.9)
            eval_opt = eval_ref

            class MockReport:
                pr_auc = 1.0
                roc_auc = 1.0
                threshold_evaluations = [eval_ref]

            generate_markdown_report(
                dataset_info=ds_info,
                split=split,
                train_time=0.1,
                pred_time=0.01,
                peak_mem_mb=1.5,
                report_eval=MockReport(),
                eval_ref=eval_ref,
                eval_p3=eval_p3,
                eval_opt=eval_opt,
                fp_samples=[],
                fn_samples=[],
                fp_count=0,
                fn_count=0,
                importances_df=importances_df,
                output_path=out_file,
            )

            assert os.path.exists(out_file)
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "# Phase 4 Development Baseline Model Report" in content
            assert "0.80" in content or "0.50" in content
