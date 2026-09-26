"""
Unit test suite for Part 2: Address Pair Features.

Covers all 20 required validation scenarios:
1. identical addresses
2. slightly different addresses
3. completely different addresses
4. one missing address
5. both missing addresses
6. identical token sets with different order
7. partially overlapping token sets
8. no token overlap
9. shared address numbers
10. different address numbers
11. no extractable address number
12. postal codes matching
13. postal codes differing
14. missing postal code
15. US 5-digit ZIP
16. Indian 6-digit PIN
17. address length difference
18. empty/whitespace addresses
19. single-token address
20. batch calculation produces the same values as the single-pair implementation
"""

from __future__ import annotations

import pytest
import polars as pl
import numpy as np

from src.features.record_representation import (
    AddressRepresentation,
    build_address_representation,
    extract_postal_code,
)
from src.features.address_features import (
    compute_address_features,
    compute_address_pair_features,
    extract_address_features_batch,
)
from src.features.feature_schema import (
    ADDRESS_FEATURE_NAMES,
    PAIR_ID_COLUMNS,
    PERSON2_FEATURE_SCHEMA,
)


class TestAddressFeatures:
    """Test suite for pairwise address features."""

    def test_1_identical_addresses(self):
        """1. Identical addresses should have exact=1, jaccard=1.0, overlap=1.0, 3gram=1.0, len_diff=0."""
        addr = "108 norle street college twp pa 16801"
        rep_a = build_address_representation(addr, country="united states")
        rep_b = build_address_representation(addr, country="united states")

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_exact"] == 1
        assert pytest.approx(feat["address_token_jaccard"]) == 1.0
        assert pytest.approx(feat["address_token_overlap"]) == 1.0
        assert pytest.approx(feat["address_char_3gram_similarity"]) == 1.0
        assert feat["address_length_difference"] == 0
        assert feat["address_missing_s1"] == 0
        assert feat["address_missing_candidate"] == 0
        assert feat["address_number_overlap"] == 1
        assert feat["postal_match"] == 1
        assert feat["shared_address_number_count"] >= 1

    def test_2_slightly_different_addresses(self):
        """2. Slightly different addresses (e.g. abbreviation or minor typo)."""
        addr_a = "108 norle street college twp pa 16801"
        addr_b = "108 norle st college twp pa 16801"
        rep_a = build_address_representation(addr_a, country="united states")
        rep_b = build_address_representation(addr_b, country="united states")

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_exact"] == 0
        assert 0.6 < feat["address_token_jaccard"] < 1.0
        assert 0.7 < feat["address_token_overlap"] <= 1.0
        assert 0.7 < feat["address_char_3gram_similarity"] < 1.0
        assert feat["address_length_difference"] > 0
        assert feat["address_number_overlap"] == 1  # '108' matches
        assert feat["postal_match"] == 1  # '16801' matches

    def test_3_completely_different_addresses(self):
        """3. Completely different addresses."""
        addr_a = "108 norle street college twp pa 16801"
        addr_b = "570 mg road bangalore 560001"
        rep_a = build_address_representation(addr_a, country="united states")
        rep_b = build_address_representation(addr_b, country="india")

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_exact"] == 0
        assert feat["address_token_jaccard"] == 0.0
        assert feat["address_token_overlap"] == 0.0
        assert feat["address_char_3gram_similarity"] < 0.1
        assert feat["shared_address_number_count"] == 0
        assert feat["address_number_overlap"] == 0  # 108 vs 570
        assert feat["postal_match"] == 0  # 16801 vs 560001

    def test_4_one_missing_address(self):
        """4. One missing address (S1 present, Candidate missing, and vice versa)."""
        rep_a = build_address_representation("108 norle street pa", country="united states")
        rep_b = build_address_representation(None)

        feat1 = compute_address_features(rep_a, rep_b)
        assert feat1["address_missing_s1"] == 0
        assert feat1["address_missing_candidate"] == 1
        assert feat1["address_exact"] == -1
        assert feat1["address_token_jaccard"] == 0.0
        assert feat1["address_token_overlap"] == 0.0
        assert feat1["address_char_3gram_similarity"] == 0.0
        assert feat1["shared_address_number_count"] == 0
        assert feat1["address_number_overlap"] == -1
        assert feat1["postal_match"] == -1
        assert feat1["address_length_difference"] == -1

        feat2 = compute_address_features(rep_b, rep_a)
        assert feat2["address_missing_s1"] == 1
        assert feat2["address_missing_candidate"] == 0
        assert feat2["address_exact"] == -1
        assert feat2["address_length_difference"] == -1

    def test_5_both_missing_addresses(self):
        """5. Both addresses missing."""
        rep_a = build_address_representation(None)
        rep_b = build_address_representation("")

        feat = compute_address_features(rep_a, rep_b)
        assert feat["address_missing_s1"] == 1
        assert feat["address_missing_candidate"] == 1
        assert feat["address_exact"] == -1
        assert feat["address_token_jaccard"] == 0.0
        assert feat["address_token_overlap"] == 0.0
        assert feat["address_char_3gram_similarity"] == 0.0
        assert feat["shared_address_number_count"] == 0
        assert feat["address_number_overlap"] == -1
        assert feat["postal_match"] == -1
        assert feat["address_length_difference"] == -1

    def test_6_identical_token_sets_different_order(self):
        """6. Identical token sets with different word order."""
        addr_a = "middle river 2701 eastern boulevard"
        addr_b = "2701 eastern boulevard middle river"
        rep_a = build_address_representation(addr_a)
        rep_b = build_address_representation(addr_b)

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_exact"] == 0  # String differs
        assert pytest.approx(feat["address_token_jaccard"]) == 1.0  # Sets identical
        assert pytest.approx(feat["address_token_overlap"]) == 1.0
        assert feat["address_length_difference"] == 0
        assert feat["address_number_overlap"] == 1

    def test_7_partially_overlapping_token_sets(self):
        """7. Partially overlapping token sets."""
        addr_a = "108 main st suite 400"
        addr_b = "108 main st 2nd floor"
        rep_a = build_address_representation(addr_a)
        rep_b = build_address_representation(addr_b)

        feat = compute_address_features(rep_a, rep_b)

        # Intersection: {'108', 'main', 'st'} (3 tokens)
        # Union: {'108', 'main', 'st', 'suite', '400', '2nd', 'floor'} (7 tokens)
        assert pytest.approx(feat["address_token_jaccard"]) == 3.0 / 7.0
        assert pytest.approx(feat["address_token_overlap"]) == 3.0 / 5.0
        assert feat["address_number_overlap"] == 1  # 108 == 108

    def test_8_no_token_overlap(self):
        """8. No token overlap."""
        addr_a = "oak plaza"
        addr_b = "elm street"
        rep_a = build_address_representation(addr_a)
        rep_b = build_address_representation(addr_b)

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_token_jaccard"] == 0.0
        assert feat["address_token_overlap"] == 0.0
        assert feat["shared_address_number_count"] == 0

    def test_9_shared_address_numbers(self):
        """9. Shared address numbers."""
        addr_a = "12 mg road 402"
        addr_b = "12 m g road 402"
        rep_a = build_address_representation(addr_a)
        rep_b = build_address_representation(addr_b)

        feat = compute_address_features(rep_a, rep_b)

        assert feat["shared_address_number_count"] == 2
        assert feat["address_number_overlap"] == 1  # '12' == '12'

    def test_10_different_address_numbers(self):
        """10. Different primary address numbers."""
        addr_a = "108 norle street"
        addr_b = "110 norle street"
        rep_a = build_address_representation(addr_a)
        rep_b = build_address_representation(addr_b)

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_number_overlap"] == 0  # 108 != 110
        assert feat["shared_address_number_count"] == 0

    def test_11_no_extractable_address_number(self):
        """11. No extractable address number."""
        addr_a = "suite b oak plaza"
        addr_b = "oak plaza"
        rep_a = build_address_representation(addr_a)
        rep_b = build_address_representation(addr_b)

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_number_overlap"] == -1
        assert feat["shared_address_number_count"] == 0

    def test_12_postal_codes_matching(self):
        """12. Postal codes matching."""
        addr_a = "108 norle street pa 16801"
        addr_b = "norle st pa 16801"
        rep_a = build_address_representation(addr_a, country="united states")
        rep_b = build_address_representation(addr_b, country="united states")

        feat = compute_address_features(rep_a, rep_b)

        assert feat["postal_match"] == 1

    def test_13_postal_codes_differing(self):
        """13. Postal codes differing."""
        addr_a = "108 norle street pa 16801"
        addr_b = "108 norle street pa 16802"
        rep_a = build_address_representation(addr_a, country="united states")
        rep_b = build_address_representation(addr_b, country="united states")

        feat = compute_address_features(rep_a, rep_b)

        assert feat["postal_match"] == 0

    def test_14_missing_postal_code(self):
        """14. Missing postal code on one or both sides."""
        addr_a = "108 norle street pa 16801"
        addr_b = "108 norle street pa"
        rep_a = build_address_representation(addr_a, country="united states")
        rep_b = build_address_representation(addr_b, country="united states")

        feat = compute_address_features(rep_a, rep_b)

        assert feat["postal_match"] == -1

    def test_15_us_5_digit_zip(self):
        """15. US 5-digit ZIP extraction and distinction from house number."""
        # 108 is house number, 77301 is zip
        addr_a = "108 friendship ln conroe tx 77301"
        addr_b = "friendship ln conroe tx 77301"
        rep_a = build_address_representation(addr_a, country="united states")
        rep_b = build_address_representation(addr_b, country="united states")

        assert rep_a.postal_code == "77301"
        assert rep_b.postal_code == "77301"
        assert rep_a.primary_number == "108"
        assert rep_b.primary_number is None  # No street number in addr_b
        feat = compute_address_features(rep_a, rep_b)
        assert feat["postal_match"] == 1
        assert feat["address_number_overlap"] == -1  # One side has no street number

    def test_16_indian_6_digit_pin(self):
        """16. Indian 6-digit PIN extraction."""
        addr_a = "797 lake town block a kolkata 700089 west bengal"
        addr_b = "block a lake town kolkata 700089"
        rep_a = build_address_representation(addr_a, country="india")
        rep_b = build_address_representation(addr_b, country="india")

        assert rep_a.postal_code == "700089"
        assert rep_b.postal_code == "700089"
        feat = compute_address_features(rep_a, rep_b)
        assert feat["postal_match"] == 1

    def test_17_address_length_difference(self):
        """17. Address length difference calculation."""
        addr_a = "short st"  # len 8
        addr_b = "very long avenue drive"  # len 22
        rep_a = build_address_representation(addr_a)
        rep_b = build_address_representation(addr_b)

        feat = compute_address_features(rep_a, rep_b)

        assert feat["address_length_difference"] == 14

    def test_18_empty_whitespace_addresses(self):
        """18. Empty and whitespace-only addresses treated as missing."""
        rep_a = build_address_representation("   ")
        rep_b = build_address_representation("none")

        assert rep_a.is_missing
        assert rep_b.is_missing
        feat = compute_address_features(rep_a, rep_b)
        assert feat["address_missing_s1"] == 1
        assert feat["address_missing_candidate"] == 1
        assert feat["address_exact"] == -1
        assert feat["address_length_difference"] == -1

    def test_19_single_token_address(self):
        """19. Single token address handled safely without division by zero."""
        rep_a = build_address_representation("pune")
        rep_b = build_address_representation("pune")
        feat = compute_address_features(rep_a, rep_b)
        assert feat["address_exact"] == 1
        assert feat["address_token_jaccard"] == 1.0

        rep_c = build_address_representation("mumbai")
        feat2 = compute_address_features(rep_a, rep_c)
        assert feat2["address_exact"] == 0
        assert feat2["address_token_jaccard"] == 0.0

    def test_20_batch_matches_single_pair_values(self):
        """20. Batch calculation produces exact same values as single-pair implementation."""
        test_pairs = [
            ("108 norle street pa 16801", "108 norle street pa 16801", "united states"),
            ("108 norle street pa 16801", "108 norle st pa 16801", "united states"),
            ("108 norle street", "570 mg road bangalore", "india"),
            ("108 main st", None, "united states"),
            (None, "570 mg road", "india"),
            (None, None, "united states"),
            ("12 mg road 402", "12 m g road 402", "india"),
            ("suite b oak plaza", "oak plaza", "united states"),
            ("797 lake town 700089", "797 lake town 700088", "india"),
            ("short", "very long avenue drive", "united states"),
        ]

        s1_reps = {}
        cand_reps = {}
        pair_rows = []

        for idx, (a1, a2, country) in enumerate(test_pairs):
            s1_id = f"S1-{idx}"
            cand_id = f"S2-{idx}"
            rep1 = build_address_representation(a1, country=country, entity_id=s1_id, source="source1")
            rep2 = build_address_representation(a2, country=country, entity_id=cand_id, source="source2")
            s1_reps[s1_id] = rep1
            cand_reps[cand_id] = rep2
            pair_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": "source2",
            })

        pairs_df = pl.DataFrame(pair_rows)

        # Batch evaluation
        batch_res = extract_address_features_batch(pairs_df, s1_reps, cand_reps)

        # Compare row-by-row
        for row in batch_res.iter_rows(named=True):
            s1_id = row["source1_entity_id"]
            c_id = row["candidate_entity_id"]
            single_feat = compute_address_features(s1_reps[s1_id], cand_reps[c_id])

            for col in ADDRESS_FEATURE_NAMES:
                val_batch = row[col]
                val_single = single_feat[col]
                if isinstance(val_batch, float):
                    assert pytest.approx(val_batch, abs=1e-5) == val_single, (
                        f"Mismatch in {col} for pair ({s1_id}, {c_id}): batch={val_batch}, single={val_single}"
                    )
                else:
                    assert val_batch == val_single, (
                        f"Mismatch in {col} for pair ({s1_id}, {c_id}): batch={val_batch}, single={val_single}"
                    )


class TestBatchEdgeCasesAndValidation:
    """Validation of batch types, empty frames, and bounds."""

    def test_schema_types_and_bounds(self):
        """Verify column types, similarity in [0, 1], flags in {-1, 0, 1}."""
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S2-2", "S2-3"],
            "candidate_source": ["source2", "source2", "source2"],
        })
        s1_reps = {
            "S1-1": build_address_representation("108 main st pa 16801", country="united states"),
            "S1-2": build_address_representation(None),
            "S1-3": build_address_representation("suite 400"),
        }
        cand_reps = {
            "S2-1": build_address_representation("108 main st pa 16801", country="united states"),
            "S2-2": build_address_representation("570 mg rd 560001", country="india"),
            "S2-3": build_address_representation(None),
        }

        res = extract_address_features_batch(pairs_df, s1_reps, cand_reps)

        assert res.columns == PAIR_ID_COLUMNS + ADDRESS_FEATURE_NAMES
        for col in ADDRESS_FEATURE_NAMES:
            assert res[col].dtype == PERSON2_FEATURE_SCHEMA[col]

        for row in res.iter_rows(named=True):
            assert row["address_exact"] in (-1, 0, 1)
            assert 0.0 <= row["address_token_jaccard"] <= 1.0
            assert 0.0 <= row["address_token_overlap"] <= 1.0
            assert 0.0 <= row["address_char_3gram_similarity"] <= 1.0
            assert row["shared_address_number_count"] >= 0
            assert row["address_number_overlap"] in (-1, 0, 1)
            assert row["postal_match"] in (-1, 0, 1)
            assert row["address_missing_s1"] in (0, 1)
            assert row["address_missing_candidate"] in (0, 1)

    def test_empty_candidate_pairs(self):
        """Empty input DataFrame produces empty typed DataFrame."""
        empty_pairs = pl.DataFrame({
            "source1_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_source": pl.Series([], dtype=pl.Utf8),
        })
        res = extract_address_features_batch(empty_pairs, {}, {})
        assert res.height == 0
        assert res.columns == PAIR_ID_COLUMNS + ADDRESS_FEATURE_NAMES

    def test_missing_pair_id_column_raises(self):
        """Missing PAIR_ID_COLUMNS in input raises ValueError."""
        bad_df = pl.DataFrame({"source1_entity_id": ["S1-1"]})
        with pytest.raises(ValueError, match="candidate_entity_id"):
            extract_address_features_batch(bad_df, {}, {})


class TestAddressRepresentationBuilder:
    """Unit tests for build_address_representation and extract_postal_code helpers."""

    def test_none_input_is_missing(self):
        rep = build_address_representation(None)
        assert rep.is_missing
        assert rep.clean_address == ""
        assert rep.token_set == frozenset()
        assert rep.primary_number is None
        assert rep.postal_code is None

    def test_empty_string_is_missing(self):
        rep = build_address_representation("")
        assert rep.is_missing

    def test_sentinel_strings_are_missing(self):
        for s in ("none", "nan", "null", "None", "NaN"):
            rep = build_address_representation(s)
            assert rep.is_missing, f"Expected '{s}' to be treated as missing"

    def test_normal_address_fields(self):
        rep = build_address_representation("108 main street", country="united states")
        assert not rep.is_missing
        assert rep.clean_address == "108 main street"
        assert rep.char_length == 15
        assert "main" in rep.token_set
        assert "108" in rep.token_set
        assert rep.primary_number == "108"
        assert len(rep.char_3grams) > 0

    def test_dict_record_input(self):
        record = {
            "business_address_normalized": "570 mg road bangalore",
            "business_address_tokens": ["570", "mg", "road", "bangalore"],
            "country_normalized": "india",
            "entity_id": "E1",
            "source": "source2",
        }
        rep = build_address_representation(record)
        assert not rep.is_missing
        assert rep.clean_address == "570 mg road bangalore"
        assert rep.entity_id == "E1"
        assert rep.source == "source2"
        assert rep.country_normalized == "india"
        assert "570" in rep.token_set

    def test_immutability(self):
        from dataclasses import FrozenInstanceError
        rep = build_address_representation("108 main st")
        with pytest.raises(FrozenInstanceError):
            rep.clean_address = "other"  # type: ignore

    def test_determinism(self):
        rep1 = build_address_representation("108 main st pa 16801", country="united states")
        rep2 = build_address_representation("108 main st pa 16801", country="united states")
        assert rep1 == rep2
        assert rep1.postal_code == rep2.postal_code
        assert rep1.primary_number == rep2.primary_number

    def test_extract_postal_code_india(self):
        assert extract_postal_code("108 lake town 700089", country="india") == "700089"
        assert extract_postal_code("700089", country="india") == "700089"
        assert extract_postal_code("mg road bangalore", country="india") is None

    def test_extract_postal_code_us(self):
        assert extract_postal_code("108 main st pa 16801", country="united states") == "16801"
        assert extract_postal_code("friendship ln conroe tx 77301", country="united states") == "77301"

    def test_extract_postal_code_missing(self):
        assert extract_postal_code(None) is None
        assert extract_postal_code("") is None
        assert extract_postal_code("suite b oak plaza") is None

    def test_backwards_compat_properties(self):
        rep = build_address_representation("108 main street", country="united states")
        assert rep.is_address_missing == rep.is_missing
        assert rep.normalized_address == rep.clean_address
        assert rep.primary_address_number == rep.primary_number
        assert set(rep.all_numeric_tokens) == set(rep.numeric_tokens)
        assert isinstance(rep.address_tokens, list)

    def test_compute_address_pair_features_alias(self):
        """compute_address_pair_features is a stable alias for compute_address_features."""
        rep = build_address_representation("108 main st")
        f1 = compute_address_features(rep, rep)
        f2 = compute_address_pair_features(rep, rep)
        assert f1 == f2
