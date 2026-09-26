"""
Unit and integration tests for Part 7: Supervised training dataset construction.

Tests:
1. Positive labeling
2. Negative labeling (no arbitrary negatives)
3. Multiple matches per S1
4. S1 entities with zero matches
5. Source2 / Source3 identity disambiguation
6. Entity-level train/validation separation (zero S1 overlap)
7. No candidate-pair leakage (zero pair overlap)
8. Reproducibility across runs with identical seed
9. Label distribution and statistics computation
10. Explicit configurable training negative downsampling
11. Blocker retention verification for validation positives
12. SupervisedDatasetBuilder high-level workflow
"""

import pytest
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    LABEL_COLUMN,
    LABEL_DTYPE,
    SUPERVISED_DATASET_COLUMNS,
    SUPERVISED_FEATURE_SCHEMA,
)
from src.features.training_dataset import (
    DatasetSplitStats,
    SupervisedDatasetSplit,
    normalize_ground_truth,
    label_candidate_pairs,
    compute_split_stats,
    split_supervised_dataset,
    SupervisedDatasetBuilder,
)


# =============================================================================
# 1. Ground Truth Normalization Tests
# =============================================================================

class TestGroundTruthNormalization:
    def test_raw_tsv_format_explosion(self):
        """Raw TSV format with comma-separated matched_entity_ids should explode correctly."""
        raw_gt = pl.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002", "S1-003"],
            "matched_entity_ids": ["S2-101,S3-201", "", "S2-102"],
        })
        norm_gt = normalize_ground_truth(raw_gt)

        assert norm_gt.columns == PAIR_ID_COLUMNS
        assert norm_gt.height == 3  # S1-002 has 0 matches, filtered out

        # S1-001 has two matches
        s1_matches = norm_gt.filter(pl.col("source1_entity_id") == "S1-001")
        assert s1_matches.height == 2
        sources = set(s1_matches["candidate_source"].to_list())
        assert sources == {"source2", "source3"}

    def test_tuple_inputs_normalization(self):
        """Tuples (s1, cand_id, source) and (s1, cand_id) should be normalized correctly."""
        tuples_3 = [
            ("S1-001", "S2-101", "source2"),
            ("S1-001", "S3-201", "source3"),
        ]
        df3 = normalize_ground_truth(tuples_3)
        assert df3.height == 2
        assert df3["candidate_source"].to_list() == ["source2", "source3"]

        tuples_2 = [
            ("S1-001", "S2-101"),
            ("S1-001", "S3-201"),
        ]
        df2 = normalize_ground_truth(tuples_2)
        assert df2.height == 2
        assert df2["candidate_source"].to_list() == ["source2", "source3"]

    def test_empty_ground_truth(self):
        """Empty ground truth should return empty DataFrame with PAIR_ID_COLUMNS."""
        empty_gt = pl.DataFrame({"source1_entity_id": [], "matched_entity_ids": []})
        norm = normalize_ground_truth(empty_gt)
        assert norm.columns == PAIR_ID_COLUMNS
        assert norm.height == 0


# =============================================================================
# 2. Positive & Negative Labeling Tests
# =============================================================================

class TestCandidatePairLabeling:
    def test_positive_and_negative_labeling(self):
        """Candidate pairs in GT become 1; other candidate pairs become 0."""
        gt = [
            ("S1-001", "S2-101", "source2"),
            ("S1-002", "S3-202", "source3"),
        ]
        candidate_pairs = pl.DataFrame({
            "source1_entity_id": ["S1-001", "S1-001", "S1-002", "S1-002"],
            "candidate_entity_id": ["S2-101", "S2-999", "S3-202", "S3-888"],
            "candidate_source": ["source2", "source2", "source3", "source3"],
        })

        labeled = label_candidate_pairs(candidate_pairs, gt)

        assert labeled.height == 4
        assert labeled.columns == PAIR_ID_COLUMNS + [LABEL_COLUMN]
        assert labeled[LABEL_COLUMN].dtype == LABEL_DTYPE
        assert labeled[LABEL_COLUMN].to_list() == [1, 0, 1, 0]

    def test_preserves_input_candidate_pairs_without_arbitrary_negatives(self):
        """Never creates Cartesian or arbitrary negative pairs not in candidate input."""
        gt = [
            ("S1-001", "S2-101", "source2"),
            ("S1-002", "S3-202", "source3"),
            ("S1-999", "S2-999", "source2"),  # Not in candidate pairs!
        ]
        candidate_pairs = pl.DataFrame({
            "source1_entity_id": ["S1-001"],
            "candidate_entity_id": ["S2-101"],
            "candidate_source": ["source2"],
        })

        labeled = label_candidate_pairs(candidate_pairs, gt)

        # Height must remain strictly 1: S1-999 is NOT added
        assert labeled.height == 1
        assert labeled["source1_entity_id"].to_list() == ["S1-001"]
        assert labeled["label"].to_list() == [1]

    def test_preserves_precomputed_features(self):
        """Feature columns already attached to candidate pairs are preserved unmodified."""
        gt = [("S1-001", "S2-101", "source2")]
        candidate_pairs = pl.DataFrame({
            "source1_entity_id": ["S1-001", "S1-001"],
            "candidate_entity_id": ["S2-101", "S2-999"],
            "candidate_source": ["source2", "source2"],
            "name_token_jaccard": [0.95, 0.10],
            "matched_key_A": [1, 0],
        })

        labeled = label_candidate_pairs(candidate_pairs, gt)

        assert labeled.columns == [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "name_token_jaccard",
            "matched_key_A",
            "label",
        ]
        assert labeled["name_token_jaccard"].to_list() == [0.95, 0.10]
        assert labeled["matched_key_A"].to_list() == [1, 0]
        assert labeled["label"].to_list() == [1, 0]


# =============================================================================
# 3. Multiple Matches & Zero Matches Tests
# =============================================================================

class TestMultipleAndZeroMatches:
    def test_multiple_matches_per_s1(self):
        """S1 entity with multiple true matches correctly labels all true matches as 1."""
        gt = [
            ("S1-001", "S2-101", "source2"),
            ("S1-001", "S2-102", "source2"),
            ("S1-001", "S3-201", "source3"),
        ]
        candidate_pairs = pl.DataFrame({
            "source1_entity_id": ["S1-001"] * 5,
            "candidate_entity_id": ["S2-101", "S2-102", "S3-201", "S2-999", "S3-888"],
            "candidate_source": ["source2", "source2", "source3", "source2", "source3"],
        })

        labeled = label_candidate_pairs(candidate_pairs, gt)

        assert labeled.height == 5
        assert labeled["label"].to_list() == [1, 1, 1, 0, 0]

    def test_s1_with_zero_matches(self):
        """S1 entity with zero matches in GT has all generated candidate pairs labeled as 0."""
        # S1-002 has 0 matches (omitted or empty in GT)
        gt = pl.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "matched_entity_ids": ["S2-101", ""],
        })
        candidate_pairs = pl.DataFrame({
            "source1_entity_id": ["S1-002"] * 4,
            "candidate_entity_id": ["S2-201", "S2-202", "S3-301", "S3-302"],
            "candidate_source": ["source2", "source2", "source3", "source3"],
        })

        labeled = label_candidate_pairs(candidate_pairs, gt)

        assert labeled.height == 4
        assert labeled["label"].to_list() == [0, 0, 0, 0]


# =============================================================================
# 4. Source2 vs Source3 Identity Disambiguation Tests
# =============================================================================

class TestSourceIdentityDisambiguation:
    def test_source2_source3_id_collision_handled_correctly(self):
        """
        If entity ID string is identical in source2 and source3 (e.g. 'ID-777'),
        source2 match does NOT match source3 candidate.
        """
        # S1-001 matches ID-777 ONLY in source2, NOT in source3
        gt = [
            ("S1-001", "ID-777", "source2"),
        ]
        candidate_pairs = pl.DataFrame({
            "source1_entity_id": ["S1-001", "S1-001"],
            "candidate_entity_id": ["ID-777", "ID-777"],
            "candidate_source": ["source2", "source3"],  # Colliding candidate ID
        })

        labeled = label_candidate_pairs(candidate_pairs, gt)

        assert labeled.height == 2
        labels = dict(zip(labeled["candidate_source"].to_list(), labeled["label"].to_list()))
        assert labels["source2"] == 1
        assert labels["source3"] == 0

    def test_invalid_missing_identity_column_raises(self):
        """Missing any column of PAIR_ID_COLUMNS raises ValueError."""
        with pytest.raises(ValueError, match="missing required composite identity column"):
            label_candidate_pairs(
                pl.DataFrame({"source1_entity_id": ["S1-001"], "candidate_entity_id": ["S2-001"]}),
                [],
            )


# =============================================================================
# 5. Entity-Level Train / Validation Separation Tests
# =============================================================================

class TestEntityLevelSplit:
    @pytest.fixture
    def synthetic_labeled_dataset(self) -> pl.DataFrame:
        """Creates a synthetic dataset with 20 S1 entities and realistic candidate counts."""
        rows = []
        for i in range(20):
            s1_id = f"S1-{i:03d}"
            # Entities 0-7 have a positive in source2
            # Entities 8-11 have a positive in source3
            # Entities 12-19 have 0 positives
            if i < 8:
                rows.append({"source1_entity_id": s1_id, "candidate_entity_id": f"S2-{i:03d}", "candidate_source": "source2", "label": 1})
            elif i < 12:
                rows.append({"source1_entity_id": s1_id, "candidate_entity_id": f"S3-{i:03d}", "candidate_source": "source3", "label": 1})

            # Add 3 negative candidates per S1 entity
            for j in range(1, 4):
                rows.append({"source1_entity_id": s1_id, "candidate_entity_id": f"S2-NEG-{i}-{j}", "candidate_source": "source2", "label": 0})

        return pl.DataFrame(rows).with_columns(pl.col("label").cast(LABEL_DTYPE))

    def test_zero_s1_leakage(self, synthetic_labeled_dataset):
        """No source1_entity_id appears in both train and validation."""
        split = split_supervised_dataset(
            synthetic_labeled_dataset,
            val_fraction=0.25,
            stratify_by_positive=True,
            seed=42,
        )

        train_s1 = set(split.train["source1_entity_id"].unique().to_list())
        val_s1 = set(split.validation["source1_entity_id"].unique().to_list())

        assert len(train_s1 & val_s1) == 0
        assert split.s1_overlap_count == 0
        assert len(train_s1 | val_s1) == synthetic_labeled_dataset["source1_entity_id"].n_unique()

    def test_zero_candidate_pair_leakage(self, synthetic_labeled_dataset):
        """No candidate pair appears in both train and validation."""
        split = split_supervised_dataset(
            synthetic_labeled_dataset,
            val_fraction=0.3,
            stratify_by_positive=True,
            seed=42,
        )

        train_pairs = set(zip(
            split.train["source1_entity_id"].to_list(),
            split.train["candidate_entity_id"].to_list(),
            split.train["candidate_source"].to_list(),
        ))
        val_pairs = set(zip(
            split.validation["source1_entity_id"].to_list(),
            split.validation["candidate_entity_id"].to_list(),
            split.validation["candidate_source"].to_list(),
        ))

        assert len(train_pairs & val_pairs) == 0
        assert split.candidate_pair_overlap_count == 0
        assert len(train_pairs) + len(val_pairs) == synthetic_labeled_dataset.height

    def test_reproducibility(self, synthetic_labeled_dataset):
        """Splitting twice with the same seed produces identical datasets."""
        split1 = split_supervised_dataset(synthetic_labeled_dataset, val_fraction=0.2, seed=123)
        split2 = split_supervised_dataset(synthetic_labeled_dataset, val_fraction=0.2, seed=123)

        assert split1.train.equals(split2.train)
        assert split1.validation.equals(split2.validation)

    def test_different_seeds_produce_different_splits(self, synthetic_labeled_dataset):
        """Different seeds partition different S1 entities."""
        split1 = split_supervised_dataset(synthetic_labeled_dataset, val_fraction=0.2, seed=42)
        split2 = split_supervised_dataset(synthetic_labeled_dataset, val_fraction=0.2, seed=999)

        val_s1_1 = set(split1.validation["source1_entity_id"].unique().to_list())
        val_s1_2 = set(split2.validation["source1_entity_id"].unique().to_list())

        assert val_s1_1 != val_s1_2

    def test_validation_positives_retained_by_blocker(self, synthetic_labeled_dataset):
        """All validation positives are candidate pairs retained by the blocker."""
        split = split_supervised_dataset(synthetic_labeled_dataset, val_fraction=0.25, seed=42)
        assert split.all_validation_positives_retained_by_blocker is True

        val_pos = split.validation.filter(pl.col("label") == 1)
        assert val_pos.height > 0
        # All validation positives must have originated from the input candidate set
        orig_pairs = set(zip(
            synthetic_labeled_dataset["source1_entity_id"].to_list(),
            synthetic_labeled_dataset["candidate_entity_id"].to_list(),
            synthetic_labeled_dataset["candidate_source"].to_list(),
        ))
        for row in val_pos.iter_rows(named=True):
            assert (row["source1_entity_id"], row["candidate_entity_id"], row["candidate_source"]) in orig_pairs


# =============================================================================
# 6. Statistics and Imbalance Reporting Tests
# =============================================================================

class TestDatasetSplitStats:
    def test_split_stats_accuracy(self):
        """DatasetSplitStats accurately counts positives, negatives, imbalance, and entities."""
        df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S2-2", "S3-1", "S2-3"],
            "candidate_source": ["source2", "source2", "source3", "source2"],
            "label": [1, 0, 1, 0],
        }).with_columns(pl.col("label").cast(LABEL_DTYPE))

        stats = compute_split_stats(df)

        assert stats.total_pairs == 4
        assert stats.positive_pairs == 2
        assert stats.negative_pairs == 2
        assert stats.positive_rate_pct == 50.0
        assert stats.imbalance_ratio == 1.0
        assert stats.unique_s1_entities == 3
        assert stats.unique_candidate_entities == 4
        assert stats.source2_candidate_count == 3
        assert stats.source3_candidate_count == 1
        assert stats.s1_entities_with_positives == 2
        assert stats.s1_entities_zero_positives == 1

    def test_empty_split_stats(self):
        """Empty DataFrame statistics handle division by zero safely."""
        empty_df = pl.DataFrame({
            "source1_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_source": pl.Series([], dtype=pl.Utf8),
            "label": pl.Series([], dtype=LABEL_DTYPE),
        })
        stats = compute_split_stats(empty_df)
        assert stats.total_pairs == 0
        assert stats.positive_rate_pct == 0.0
        assert stats.imbalance_ratio == 0.0


# =============================================================================
# 7. Explicit Negative Downsampling Tests
# =============================================================================

class TestExplicitDownsampling:
    def test_training_negative_downsampling(self):
        """
        Explicit negative downsampling reduces training negatives while
        keeping 100% of training positives and leaving validation untouched.
        """
        # 10 positives, 100 negatives
        rows = []
        for i in range(10):
            rows.append({"source1_entity_id": f"S1-{i}", "candidate_entity_id": f"S2-POS-{i}", "candidate_source": "source2", "label": 1})
            for j in range(10):
                rows.append({"source1_entity_id": f"S1-{i}", "candidate_entity_id": f"S2-NEG-{i}-{j}", "candidate_source": "source2", "label": 0})

        df = pl.DataFrame(rows).with_columns(pl.col("label").cast(LABEL_DTYPE))

        split = split_supervised_dataset(
            df,
            val_fraction=0.2,
            negative_downsample_ratio=2.0,  # Max 2 negatives per positive
            seed=42,
        )

        # Validation must NOT be downsampled
        val_pos = split.validation.filter(pl.col("label") == 1).height
        val_neg = split.validation.filter(pl.col("label") == 0).height
        assert val_neg == val_pos * 10  # Natural 1:10 ratio maintained

        # Training negatives must be capped at 2.0 * train_pos
        train_pos = split.train.filter(pl.col("label") == 1).height
        train_neg = split.train.filter(pl.col("label") == 0).height
        assert train_neg == train_pos * 2
        assert split.train_stats.imbalance_ratio == 2.0


# =============================================================================
# 8. High-Level SupervisedDatasetBuilder Tests
# =============================================================================

class TestSupervisedDatasetBuilder:
    def test_end_to_end_builder(self):
        """SupervisedDatasetBuilder executes labeling and splitting end-to-end."""
        gt_tuples = [
            ("S1-A", "S2-1", "source2"),
            ("S1-B", "S3-2", "source3"),
        ]
        builder = SupervisedDatasetBuilder(gt_tuples)

        pairs = pl.DataFrame({
            "source1_entity_id": ["S1-A", "S1-A", "S1-B", "S1-C"],
            "candidate_entity_id": ["S2-1", "S2-9", "S3-2", "S2-8"],
            "candidate_source": ["source2", "source2", "source3", "source2"],
        })

        split = builder.create_splits(pairs, val_fraction=0.33, seed=42)

        assert split.s1_overlap_count == 0
        assert split.candidate_pair_overlap_count == 0
        assert split.train.height + split.validation.height == 4
        assert LABEL_COLUMN in split.train.columns
        assert LABEL_COLUMN in split.validation.columns
