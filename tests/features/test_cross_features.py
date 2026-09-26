"""
Unit tests for Part 3: Cross-Field Features (country_match).

Covers:
- matching countries
- different countries
- missing S1 country
- missing candidate country
- both missing
- whitespace and case insensitivity
- sentinel values ('', 'none', 'nan', 'null')
- missing entity IDs from lookup tables
- representation objects as input
- batch vs single-pair consistency
- correct dtype and value range
- empty candidate pairs DataFrame
"""

from __future__ import annotations

import pytest
import polars as pl

from src.features.cross_features import (
    compute_cross_features,
    extract_cross_features_batch,
)
from src.features.record_representation import (
    build_address_representation,
)
from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    CROSS_FEATURE_NAMES,
    PERSON2_FEATURE_SCHEMA,
)


class TestCrossFeatures:
    """Test suite for country_match feature."""

    def test_matching_countries(self):
        """Matching normalized countries should return 1."""
        assert compute_cross_features("united states", "united states")["country_match"] == 1
        assert compute_cross_features("india", "india")["country_match"] == 1

    def test_case_and_whitespace_insensitivity(self):
        """Case variations and leading/trailing whitespace should still match."""
        assert compute_cross_features("  United States  ", "united states")["country_match"] == 1
        assert compute_cross_features("INDIA", "india")["country_match"] == 1
        assert compute_cross_features("  India", "India  ")["country_match"] == 1

    def test_different_countries(self):
        """Different valid countries should return 0."""
        assert compute_cross_features("united states", "india")["country_match"] == 0
        assert compute_cross_features("india", "france")["country_match"] == 0

    def test_missing_s1_country(self):
        """Missing S1 country should return -1."""
        assert compute_cross_features(None, "india")["country_match"] == -1
        assert compute_cross_features("", "india")["country_match"] == -1
        assert compute_cross_features("   ", "india")["country_match"] == -1
        assert compute_cross_features("none", "india")["country_match"] == -1
        assert compute_cross_features("nan", "india")["country_match"] == -1
        assert compute_cross_features("null", "india")["country_match"] == -1

    def test_missing_candidate_country(self):
        """Missing candidate country should return -1."""
        assert compute_cross_features("united states", None)["country_match"] == -1
        assert compute_cross_features("united states", "")["country_match"] == -1
        assert compute_cross_features("united states", "   ")["country_match"] == -1
        assert compute_cross_features("united states", "None")["country_match"] == -1
        assert compute_cross_features("united states", "NAN")["country_match"] == -1

    def test_both_missing_countries(self):
        """Both countries missing should return -1."""
        assert compute_cross_features(None, None)["country_match"] == -1
        assert compute_cross_features("", "")["country_match"] == -1
        assert compute_cross_features("None", "none")["country_match"] == -1
        assert compute_cross_features(None, "")["country_match"] == -1

    def test_with_representation_objects(self):
        """Accepts AddressRepresentation or dictionaries with country_normalized."""
        rep_a = build_address_representation("108 main st", country="united states")
        rep_b = build_address_representation("570 mg rd", country="united states")
        rep_c = build_address_representation("570 mg rd", country="india")
        rep_missing = build_address_representation("108 main st", country=None)

        assert compute_cross_features(rep_a, rep_b)["country_match"] == 1
        assert compute_cross_features(rep_a, rep_c)["country_match"] == 0
        assert compute_cross_features(rep_a, rep_missing)["country_match"] == -1


class TestBatchCrossFeatures:
    """Test suite for batch/DataFrame cross feature extraction."""

    def test_batch_vs_single_pair_consistency(self):
        """Batch extraction produces exact same values as single-pair evaluation."""
        test_cases = [
            ("united states", "united states"),
            ("India", "india"),
            ("united states", "india"),
            ("india", "united states"),
            (None, "india"),
            ("united states", None),
            (None, None),
            ("   ", "india"),
            ("united states", "none"),
            ("france", "france"),
        ]

        pair_rows = []
        s1_countries = {}
        cand_countries = {}

        for idx, (c1, c2) in enumerate(test_cases):
            s1_id = f"S1-{idx}"
            cand_id = f"S2-{idx}"
            s1_countries[s1_id] = c1
            cand_countries[cand_id] = c2
            pair_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": "source2",
            })

        # Add one pair where entities are absent from lookup tables
        pair_rows.append({
            "source1_entity_id": "S1-missing",
            "candidate_entity_id": "S2-missing",
            "candidate_source": "source2",
        })

        pairs_df = pl.DataFrame(pair_rows)

        # Vectorized batch calculation
        batch_res = extract_cross_features_batch(pairs_df, s1_countries, cand_countries)

        assert batch_res.columns == PAIR_ID_COLUMNS + CROSS_FEATURE_NAMES
        assert batch_res["country_match"].dtype == PERSON2_FEATURE_SCHEMA["country_match"]
        assert batch_res.height == len(pair_rows)

        # Validate values against single-pair
        for row in batch_res.iter_rows(named=True):
            s1_id = row["source1_entity_id"]
            c_id = row["candidate_entity_id"]
            c1 = s1_countries.get(s1_id)
            c2 = cand_countries.get(c_id)
            single_val = compute_cross_features(c1, c2)["country_match"]
            assert row["country_match"] == single_val, f"Mismatch for pair ({s1_id}, {c_id})"

    def test_schema_dtype_and_value_range(self):
        """Verify column types, bounds in {-1, 0, 1}, and identity column preservation."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S3-2", "S2-3"],
            "candidate_source": ["source2", "source3", "source2"],
        })
        s1_map = {"S1-1": "united states", "S1-2": "india", "S1-3": None}
        cand_map = {"S2-1": "united states", "S3-2": "united states", "S2-3": "india"}

        res = extract_cross_features_batch(pairs_df, s1_map, cand_map)

        assert res.columns == ["source1_entity_id", "candidate_entity_id", "candidate_source", "country_match"]
        assert res["country_match"].dtype == pl.Int8
        assert res["country_match"].is_in([-1, 0, 1]).all()
        assert res["country_match"].to_list() == [1, 0, -1]

        # Verify composite identity unchanged
        assert res["source1_entity_id"].to_list() == ["S1-1", "S1-2", "S1-3"]
        assert res["candidate_entity_id"].to_list() == ["S2-1", "S3-2", "S2-3"]
        assert res["candidate_source"].to_list() == ["source2", "source3", "source2"]

    def test_empty_candidate_pairs(self):
        """Empty input DataFrame produces empty typed DataFrame."""
        empty_pairs = pl.DataFrame({
            "source1_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_source": pl.Series([], dtype=pl.Utf8),
        })
        res = extract_cross_features_batch(empty_pairs, {}, {})
        assert res.height == 0
        assert res.columns == PAIR_ID_COLUMNS + CROSS_FEATURE_NAMES
        assert res["country_match"].dtype == pl.Int8

    def test_missing_identity_column_raises_error(self):
        """Missing any column of composite identity raises ValueError."""
        invalid_pairs = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            # candidate_source missing
        })
        with pytest.raises(ValueError, match="Required identity column 'candidate_source' missing"):
            extract_cross_features_batch(invalid_pairs, {}, {})
