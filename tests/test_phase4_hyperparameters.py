"""
Unit tests for Phase 4D: Entity-Disjoint K-Fold Cross-Validation and Hyperparameter Tuning.

Tests:
1. Entity-grouped folds guarantee zero S1 overlap for all folds.
2. Candidate pairs are partitioned disjointly across validation folds (zero pair overlap).
3. Every fold contains positive ground-truth examples.
4. Determinism: Seed 42 produces identical fold entity assignments across runs.
5. Out-of-fold probability vector length, bounds, and indexing alignment.
6. Hyperparameter configuration reproducibility and feature importance aggregation.
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
from src.models.cross_validation import (
    create_entity_disjoint_kfold_splits,
    run_cv_experiment,
)


@pytest.fixture
def mock_labeled_dataset() -> pl.DataFrame:
    """Creates a realistic small candidate dataset with 20 S1 entities and mixed labels."""
    np.random.seed(42)
    rows = []
    # 20 S1 entities, each having 3-5 candidates
    for s1_idx in range(20):
        s1_id = f"S1-{s1_idx:03d}"
        n_cands = 4
        for c_idx in range(n_cands):
            c_source = "source2" if c_idx % 2 == 0 else "source3"
            cand_id = f"C-{s1_idx}-{c_idx}"
            # Positives for even S1 entities, candidate 0
            label = 1 if (s1_idx % 2 == 0 and c_idx == 0) else 0
            row = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": c_source,
                LABEL_COLUMN: label,
            }
            # Add 29 base features
            for feat in ALL_FEATURE_NAMES:
                row[feat] = float(np.random.uniform(0.0, 1.0))
            rows.append(row)

    return pl.DataFrame(rows)


class TestEntityDisjointKFoldSplits:
    def test_kfold_guarantees(self, mock_labeled_dataset):
        """Validates zero S1 overlap, zero pair overlap, positive presence, and exhaustiveness."""
        n_splits = 5
        splits = create_entity_disjoint_kfold_splits(
            mock_labeled_dataset,
            n_splits=n_splits,
            seed=42,
        )

        assert len(splits) == n_splits

        total_val_rows = 0
        all_val_s1: List[str] = []

        for split in splits:
            # 1. Leakage guarantees
            assert split.s1_overlap_count == 0
            assert split.candidate_pair_overlap_count == 0

            train_s1 = set(split.train["source1_entity_id"].to_list())
            val_s1 = set(split.validation["source1_entity_id"].to_list())
            assert train_s1.isdisjoint(val_s1)

            # 2. Every fold contains positive examples
            assert split.validation_stats.positive_pairs > 0
            assert split.train_stats.positive_pairs > 0

            total_val_rows += split.validation.height
            all_val_s1.extend(list(val_s1))

        # 3. Exhaustiveness: every row is in validation exactly once
        assert total_val_rows == mock_labeled_dataset.height

        # Every S1 entity appears in exactly one validation fold
        unique_s1 = mock_labeled_dataset["source1_entity_id"].n_unique()
        assert len(all_val_s1) == unique_s1
        assert len(set(all_val_s1)) == unique_s1

    def test_kfold_reproducibility(self, mock_labeled_dataset):
        """Verifies that seed=42 produces completely deterministic fold assignments."""
        splits_1 = create_entity_disjoint_kfold_splits(mock_labeled_dataset, n_splits=5, seed=42)
        splits_2 = create_entity_disjoint_kfold_splits(mock_labeled_dataset, n_splits=5, seed=42)

        for s1, s2 in zip(splits_1, splits_2):
            assert s1.validation["source1_entity_id"].to_list() == s2.validation["source1_entity_id"].to_list()
            assert s1.validation.height == s2.validation.height
            assert s1.validation_stats.positive_pairs == s2.validation_stats.positive_pairs


class TestCVExperimentRunner:
    def test_run_cv_experiment(self, mock_labeled_dataset):
        """Tests that run_cv_experiment executes K-fold CV and aggregates OOF metrics."""
        splits = create_entity_disjoint_kfold_splits(mock_labeled_dataset, n_splits=3, seed=42)

        lgbm_params = {
            "n_estimators": 5,
            "num_leaves": 15,
            "min_child_samples": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        }

        res = run_cv_experiment(
            splits=splits,
            config_name="test_config",
            feature_names=list(ALL_FEATURE_NAMES),
            lgbm_params=lgbm_params,
        )

        assert len(res.fold_metrics) == 3
        assert len(res.oof_probabilities) == mock_labeled_dataset.height
        assert len(res.oof_y_true) == mock_labeled_dataset.height
        assert np.all((res.oof_probabilities >= 0.0) & (res.oof_probabilities <= 1.0))

        # Metrics are computed
        assert 0.0 <= res.mean_pr_auc <= 1.0
        assert 0.0 <= res.mean_roc_auc <= 1.0
        assert 0.0 <= res.oof_pr_auc <= 1.0
        assert 0.0 <= res.oof_opt_f05 <= 1.0
        assert 0.05 <= res.oof_opt_threshold <= 0.99

        # Feature importances are aggregated
        assert res.feature_importances.height == len(ALL_FEATURE_NAMES)
        assert "rank" in res.feature_importances.columns
        assert res.feature_importances["rank"][0] == 1
