"""
Unit tests for Part 4: Blocking Provenance Features.

Covers:
1. Single-pair feature calculation (compute_blocking_features, compute_blocking_pair_features)
   - Each key individually: A, C, D, E, F
   - Multiple keys combinations (e.g. A+C, C+D+F, all 5 keys)
   - No matched keys (empty set, None, unknown keys)
   - Formatting and casing robustness ("a", "KEY_A", "A||country||name")
   - Value types and ranges (flags in {0, 1}, count in [0, 5])
   - Record representation / dict comparison
2. Vectorized batch calculation from provenance mapping (extract_blocking_features_from_provenance_dict)
   - Integration with BlockIndex.generate_pairs()
   - Unmatched pairs in candidate list receiving zeroes
   - Empty candidate pairs DataFrame handling
   - Empty provenance dict handling
   - Verification of composite pair identity preservation
   - Data types check (strictly pl.Int8)
3. Vectorized batch calculation from precomputed keys (extract_blocking_features_from_keys)
   - Dictionaries of BlockingRecordRepresentation
   - Dictionaries of plain dicts
   - Polars DataFrames
   - Handling of nulls, empty strings, and missing keys (no spurious matches)
   - List/tuple of candidate sources (S2 + S3)
   - Verification against real sample parquet files on disk
4. Unified batch extraction (extract_blocking_features_batch)
   - Routing to provenance vs keys
   - Fallback when neither is provided
   - Missing identity column validation
5. Consistency between single-pair and batch methods
"""

from __future__ import annotations

import os
import pytest
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    BLOCKING_FEATURE_NAMES,
    PERSON2_FEATURE_SCHEMA,
    BlockingRecordRepresentation,
)
from src.features.blocking_features import (
    VALID_KEYS,
    compute_blocking_features,
    compute_blocking_pair_features,
    extract_blocking_features_from_provenance_dict,
    extract_blocking_features_from_keys,
    extract_blocking_features_batch,
)
from src.candidate_generation.block_index import BlockIndex


# =============================================================================
# 1. Single-Pair Tests
# =============================================================================

class TestSinglePairBlockingFeatures:
    """Test suite for single-pair blocking feature calculations."""

    @pytest.mark.parametrize("key_name", ["A", "C", "D", "E", "F"])
    def test_individual_key_match(self, key_name: str):
        """Each valid key should produce 1 for its matched feature and 1 for count."""
        res = compute_blocking_features({key_name})
        for k in VALID_KEYS:
            expected = 1 if k == key_name else 0
            assert res[f"matched_key_{k}"] == expected
        assert res["matched_key_count"] == 1

    def test_multiple_keys_match(self):
        """Multiple matched keys should reflect their individual flags and exact sum."""
        res = compute_blocking_features({"A", "C", "D"})
        assert res["matched_key_A"] == 1
        assert res["matched_key_C"] == 1
        assert res["matched_key_D"] == 1
        assert res["matched_key_E"] == 0
        assert res["matched_key_F"] == 0
        assert res["matched_key_count"] == 3

    def test_all_five_keys_match(self):
        """All five active keys matching should yield count = 5."""
        res = compute_blocking_features({"A", "C", "D", "E", "F"})
        for k in VALID_KEYS:
            assert res[f"matched_key_{k}"] == 1
        assert res["matched_key_count"] == 5

    def test_no_keys_match(self):
        """Empty or None input should yield 0 for all flags and count = 0."""
        res_empty = compute_blocking_features(set())
        assert res_empty["matched_key_count"] == 0
        for k in VALID_KEYS:
            assert res_empty[f"matched_key_{k}"] == 0

        res_none = compute_blocking_features(None)
        assert res_none["matched_key_count"] == 0
        for k in VALID_KEYS:
            assert res_none[f"matched_key_{k}"] == 0

    def test_excluded_or_unknown_keys_ignored(self):
        """Key B (permanently excluded) and random keys should not be counted."""
        res = compute_blocking_features({"B", "Z", "X"})
        assert res["matched_key_count"] == 0
        for k in VALID_KEYS:
            assert res[f"matched_key_{k}"] == 0

        # Mixed valid and invalid keys
        res_mixed = compute_blocking_features({"A", "B", "Z"})
        assert res_mixed["matched_key_A"] == 1
        assert res_mixed["matched_key_count"] == 1

    def test_case_and_prefix_robustness(self):
        """Should accept lowercase, 'key_' prefix, or full key strings ('A||...')."""
        assert compute_blocking_features({"a"})["matched_key_A"] == 1
        assert compute_blocking_features({"key_c"})["matched_key_C"] == 1
        assert compute_blocking_features({"D||india||tata||100"})["matched_key_D"] == 1
        assert compute_blocking_features({"KEY_E"})["matched_key_E"] == 1
        assert compute_blocking_features({"F||united states||108 main st"})["matched_key_F"] == 1

    def test_compute_blocking_pair_features_representations(self):
        """Comparing two BlockingRecordRepresentation objects."""
        s1 = BlockingRecordRepresentation(
            entity_id="S1-1",
            key_A="A||us||apple inc",
            key_C="C||us||apple inc",
            key_D="D||us||apple||108",
            key_E=None,
            key_F="F||us||108 main st",
        )
        cand = BlockingRecordRepresentation(
            entity_id="S2-1",
            key_A="A||us||apple inc",      # matches
            key_C="C||us||apple retail",   # differs
            key_D="D||us||apple||108",     # matches
            key_E="E||us||apple",          # s1 has None -> no match
            key_F=None,                    # cand has None -> no match
        )
        res = compute_blocking_pair_features(s1, cand)
        assert res["matched_key_A"] == 1
        assert res["matched_key_C"] == 0
        assert res["matched_key_D"] == 1
        assert res["matched_key_E"] == 0
        assert res["matched_key_F"] == 0
        assert res["matched_key_count"] == 2

    def test_compute_blocking_pair_features_dicts(self):
        """Comparing two plain record dictionaries."""
        s1 = {"key_A": "A||in||shree ram", "key_C": "C||in||shree ram", "key_D": None}
        c = {"key_A": "A||in||shree ram", "key_C": "C||in||different", "key_D": None}
        res = compute_blocking_pair_features(s1, c)
        assert res["matched_key_A"] == 1
        assert res["matched_key_C"] == 0
        assert res["matched_key_D"] == 0
        assert res["matched_key_count"] == 1


# =============================================================================
# 2. Vectorized Batch Provenance Tests
# =============================================================================

class TestBatchProvenanceFeatures:
    """Test suite for extract_blocking_features_from_provenance_dict."""

    def test_batch_provenance_extraction_basic(self):
        """Extract features from provenance dictionary."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S2-2", "S3-1"],
            "candidate_source": ["source2", "source2", "source3"],
        })
        prov = {
            ("S1-1", "S2-1"): {"A", "C"},
            ("S1-2", "S2-2"): {"D", "F"},
            ("S1-3", "S3-1"): {"E"},
        }
        res = extract_blocking_features_from_provenance_dict(pairs_df, prov)

        assert res.height == 3
        assert res.columns == PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES
        for col in BLOCKING_FEATURE_NAMES:
            assert res[col].dtype == pl.Int8

        # S1-1 vs S2-1: A + C
        assert res["matched_key_A"][0] == 1
        assert res["matched_key_C"][0] == 1
        assert res["matched_key_D"][0] == 0
        assert res["matched_key_count"][0] == 2

        # S1-2 vs S2-2: D + F
        assert res["matched_key_D"][1] == 1
        assert res["matched_key_F"][1] == 1
        assert res["matched_key_count"][1] == 2

        # S1-3 vs S3-1: E
        assert res["matched_key_E"][2] == 1
        assert res["matched_key_count"][2] == 1

    def test_integration_with_block_index_generate_pairs(self):
        """Integration test with BlockIndex.generate_pairs output."""
        idx = BlockIndex()
        s1_rec = {
            "entity_id": "S1-100",
            "business_name_normalized": "alpha beta",
            "business_name_transliterated": "alpha beta",
            "business_address_normalized": "100 main road",
            "country_normalized": "india",
        }
        cand_rec = {
            "entity_id": "S2-200",
            "business_name_normalized": "alpha beta",
            "business_name_transliterated": "alpha beta",
            "business_address_normalized": "100 main road",
            "country_normalized": "india",
        }
        idx.add_s1_record("S1-100", s1_rec, active_keys=VALID_KEYS)
        idx.add_candidate_record("S2-200", cand_rec, active_keys=VALID_KEYS)

        pairs, prov = idx.generate_pairs()
        assert ("S1-100", "S2-200") in pairs
        assert len(prov[("S1-100", "S2-200")]) > 0

        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-100"],
            "candidate_entity_id": ["S2-200"],
            "candidate_source": ["source2"],
        })
        res = extract_blocking_features_from_provenance_dict(pairs_df, prov)
        assert res["matched_key_count"][0] == len(prov[("S1-100", "S2-200")] & set(VALID_KEYS))
        assert res["matched_key_A"][0] == (1 if "A" in prov[("S1-100", "S2-200")] else 0)

    def test_unmatched_pairs_get_zeroes(self):
        """Pairs present in candidate_pairs_df but not in provenance dict receive zeroes."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-999"],
            "candidate_entity_id": ["S2-1", "S2-999"],
            "candidate_source": ["source2", "source2"],
        })
        prov = {("S1-1", "S2-1"): {"A"}}
        res = extract_blocking_features_from_provenance_dict(pairs_df, prov)

        # S1-999 vs S2-999 has no provenance
        assert res["matched_key_A"][1] == 0
        assert res["matched_key_count"][1] == 0

    def test_empty_candidate_pairs(self):
        """Empty candidate pairs DataFrame returns empty DataFrame with correct schema and dtypes."""
        empty_df = pl.DataFrame({
            "source1_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_source": pl.Series([], dtype=pl.Utf8),
        })
        res = extract_blocking_features_from_provenance_dict(empty_df, {})
        assert res.height == 0
        assert res.columns == PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES
        for col in BLOCKING_FEATURE_NAMES:
            assert res[col].dtype == pl.Int8

    def test_empty_provenance_dict(self):
        """Non-empty candidate pairs with empty provenance dict gives all zeroes."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "candidate_source": ["source2", "source2"],
        })
        res = extract_blocking_features_from_provenance_dict(pairs_df, {})
        assert res.height == 2
        assert res["matched_key_count"].to_list() == [0, 0]


# =============================================================================
# 3. Vectorized Batch Keys Tests
# =============================================================================

class TestBatchKeyComparisonFeatures:
    """Test suite for extract_blocking_features_from_keys."""

    def test_from_blocking_record_representations(self):
        """Extract features by comparing dictionaries of BlockingRecordRepresentation."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "candidate_source": ["source2", "source2"],
        })
        s1_keys = {
            "S1-1": BlockingRecordRepresentation(
                entity_id="S1-1",
                key_A="A||us||starbucks",
                key_C="C||us||starbucks coffee",
                key_D="D||us||starbucks||120",
                key_E=None,
                key_F="F||us||120 pike st",
            ),
            "S1-2": BlockingRecordRepresentation(
                entity_id="S1-2",
                key_A="A||in||infy",
                key_C="C||in||infosys tech",
                key_D=None,
                key_E="E||in||infosys",
                key_F=None,
            ),
        }
        cand_keys = {
            "S2-1": BlockingRecordRepresentation(
                entity_id="S2-1",
                key_A="A||us||starbucks",          # Match
                key_C="C||us||starbucks drive",   # Mismatch
                key_D="D||us||starbucks||120",     # Match
                key_E=None,
                key_F="F||us||120 pike st",        # Match
            ),
            "S2-2": BlockingRecordRepresentation(
                entity_id="S2-2",
                key_A="A||in||wipro",
                key_C="C||in||wipro ltd",
                key_D=None,
                key_E="E||in||infosys",            # Match
                key_F=None,
            ),
        }

        res = extract_blocking_features_from_keys(pairs_df, s1_keys, cand_keys)
        assert res.height == 2
        assert res.columns == PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES

        # S1-1 vs S2-1: A, D, F match -> count = 3
        assert res["matched_key_A"][0] == 1
        assert res["matched_key_C"][0] == 0
        assert res["matched_key_D"][0] == 1
        assert res["matched_key_E"][0] == 0
        assert res["matched_key_F"][0] == 1
        assert res["matched_key_count"][0] == 3

        # S1-2 vs S2-2: E matches -> count = 1
        assert res["matched_key_A"][1] == 0
        assert res["matched_key_E"][1] == 1
        assert res["matched_key_count"][1] == 1

    def test_from_polars_dataframes(self):
        """Extract features directly from Polars DataFrames."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-2"],
            "candidate_source": ["source2", "source3"],
        })
        s1_df = pl.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "key_A": ["A||us||alpha", "A||us||beta"],
            "key_C": ["C||us||alpha corp", None],
            "key_D": [None, "D||us||beta||42"],
            "key_E": ["E||us||alpha", None],
            "key_F": ["F||us||alpha lane", "F||us||42 wallaby"],
        })
        cand_df = pl.DataFrame({
            "entity_id": ["S2-1", "S3-2"],
            "key_A": ["A||us||alpha", "A||us||beta"],
            "key_C": ["C||us||alpha corp", None],
            "key_D": [None, "D||us||beta||99"],     # differs
            "key_E": ["E||us||alpha", None],
            "key_F": ["F||us||diff lane", "F||us||42 wallaby"],
        })

        res = extract_blocking_features_from_keys(pairs_df, s1_df, cand_df)
        assert res.height == 2

        # S1-1: A, C, E match
        assert res["matched_key_A"][0] == 1
        assert res["matched_key_C"][0] == 1
        assert res["matched_key_E"][0] == 1
        assert res["matched_key_F"][0] == 0
        assert res["matched_key_count"][0] == 3

        # S1-2: A, F match
        assert res["matched_key_A"][1] == 1
        assert res["matched_key_D"][1] == 0
        assert res["matched_key_F"][1] == 1
        assert res["matched_key_count"][1] == 2

    def test_no_false_matches_on_null_or_empty(self):
        """Null or empty keys must never match each other."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "candidate_source": ["source2"],
        })
        s1_keys = {"S1-1": {"key_A": None, "key_C": "", "key_D": "   ", "key_E": None, "key_F": None}}
        cand_keys = {"S2-1": {"key_A": None, "key_C": "", "key_D": "   ", "key_E": None, "key_F": None}}

        res = extract_blocking_features_from_keys(pairs_df, s1_keys, cand_keys)
        assert res["matched_key_count"][0] == 0
        for k in VALID_KEYS:
            assert res[f"matched_key_{k}"][0] == 0

    def test_missing_entities_in_lookup(self):
        """Entities missing from lookup tables get zeroes."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-UNKNOWN"],
            "candidate_entity_id": ["S2-UNKNOWN"],
            "candidate_source": ["source2"],
        })
        s1_keys = {"S1-OTHER": {"key_A": "A||us||foo"}}
        cand_keys = {"S2-OTHER": {"key_A": "A||us||foo"}}

        res = extract_blocking_features_from_keys(pairs_df, s1_keys, cand_keys)
        assert res["matched_key_count"][0] == 0

    def test_real_parquet_sampling(self):
        """Verify functionality on real precomputed blocking keys parquets if present."""
        s1_path = "data/processed/train/blocking_keys/s1_keys.parquet"
        s2_path = "data/processed/train/blocking_keys/s2_keys.parquet"
        if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
            pytest.skip("Precomputed blocking parquets not found.")

        # Read small sample
        s1_sample = pl.read_parquet(s1_path, n_rows=5)
        s2_sample = pl.read_parquet(s2_path, n_rows=5)

        pairs_df = pl.DataFrame({
            "source1_entity_id": s1_sample["entity_id"][:3].to_list(),
            "candidate_entity_id": s2_sample["entity_id"][:3].to_list(),
            "candidate_source": ["source2", "source2", "source2"],
        })

        res = extract_blocking_features_from_keys(pairs_df, s1_sample, s2_sample)
        assert res.height == 3
        for col in BLOCKING_FEATURE_NAMES:
            assert col in res.columns
            assert res[col].dtype == pl.Int8
            assert res[col].is_not_null().all()


# =============================================================================
# 4. Unified Batch & Consistency Tests
# =============================================================================

class TestUnifiedBatchAndConsistency:
    """Test suite for unified extract_blocking_features_batch and consistency."""

    def test_unified_batch_with_provenance(self):
        """Unified extractor delegates correctly to provenance dict."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "candidate_source": ["source2"],
        })
        prov = {("S1-1", "S2-1"): {"A", "F"}}
        res = extract_blocking_features_batch(pairs_df, pair_provenance=prov)
        assert res["matched_key_A"][0] == 1
        assert res["matched_key_F"][0] == 1
        assert res["matched_key_count"][0] == 2

    def test_unified_batch_with_keys(self):
        """Unified extractor delegates correctly to keys lookup."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "candidate_source": ["source2"],
        })
        s1_keys = {"S1-1": {"key_C": "C||us||target"}}
        cand_keys = {"S2-1": {"key_C": "C||us||target"}}
        res = extract_blocking_features_batch(pairs_df, s1_keys=s1_keys, cand_keys=cand_keys)
        assert res["matched_key_C"][0] == 1
        assert res["matched_key_count"][0] == 1

    def test_unified_batch_fallback_zeroes(self):
        """Unified extractor returns zeroes when neither provenance nor keys are provided."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "candidate_source": ["source2"],
        })
        res = extract_blocking_features_batch(pairs_df)
        assert res["matched_key_count"][0] == 0
        for k in VALID_KEYS:
            assert res[f"matched_key_{k}"][0] == 0

    def test_missing_identity_column_raises(self):
        """Missing any composite identity column raises ValueError."""
        bad_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            # candidate_source missing
        })
        with pytest.raises(ValueError, match="Required identity column"):
            extract_blocking_features_batch(bad_df)

    def test_batch_single_consistency(self):
        """Batch extraction yields exact same values as single-pair evaluation."""
        pairs = [
            ("S1-1", "S2-1", {"A"}),
            ("S1-2", "S2-2", {"C", "D"}),
            ("S1-3", "S2-3", set()),
            ("S1-4", "S2-4", {"A", "C", "D", "E", "F"}),
        ]
        pairs_df = pl.DataFrame({
            "source1_entity_id": [p[0] for p in pairs],
            "candidate_entity_id": [p[1] for p in pairs],
            "candidate_source": ["source2"] * len(pairs),
        })
        prov = {(p[0], p[1]): p[2] for p in pairs}

        batch_res = extract_blocking_features_from_provenance_dict(pairs_df, prov)

        for i, (s1, cand, expected_keys) in enumerate(pairs):
            single_res = compute_blocking_features(expected_keys)
            for k in VALID_KEYS:
                feat_name = f"matched_key_{k}"
                assert batch_res[feat_name][i] == single_res[feat_name], f"Mismatch on {feat_name} row {i}"
            assert batch_res["matched_key_count"][i] == single_res["matched_key_count"]
