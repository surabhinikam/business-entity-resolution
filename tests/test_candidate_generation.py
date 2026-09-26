"""
Unit tests for the candidate_generation module.

Tests stopwords, name tokens, address parser, blocking keys,
and block index functionality.
"""

import pytest

from src.candidate_generation.stopwords import BLOCKING_STOPWORDS, is_stopword
from src.candidate_generation.name_tokens import extract_meaningful_tokens
from src.candidate_generation.address_parser import (
    extract_address_number,
    has_address_number,
)
from src.candidate_generation.block_keys import (
    key_a_exact_norm_name,
    key_b_exact_translit_name,
    key_c_first_two_meaningful_tokens,
    key_d_first_token_addr_num,
    key_e_first_token_fallback,
    key_f_exact_norm_address,
    generate_all_keys,
    BLOCKING_KEY_NAMES,
)
from src.candidate_generation.block_index import BlockIndex


# ============================================================================
# Stopwords tests
# ============================================================================

class TestStopwords:
    def test_legal_suffixes_are_stopwords(self):
        """Legal suffix tokens should be in the stopword list."""
        for token in ["pvt", "ltd", "llc", "llp", "inc", "corp", "co", "plc"]:
            assert is_stopword(token), f"{token} should be a stopword"

    def test_generic_business_terms_are_stopwords(self):
        """Common business terms should be stopwords."""
        for token in ["services", "solutions", "enterprises", "group", "trading"]:
            assert is_stopword(token), f"{token} should be a stopword"

    def test_meaningful_tokens_not_stopwords(self):
        """Actual business names should not be stopwords."""
        for token in ["walmart", "google", "tata", "reliance", "amazon"]:
            assert not is_stopword(token), f"{token} should NOT be a stopword"

    def test_honorifics_are_stopwords(self):
        """Honorifics should be stopwords."""
        for token in ["mr", "mrs", "dr", "shri"]:
            assert is_stopword(token), f"{token} should be a stopword"

    def test_stopword_set_is_frozen(self):
        """BLOCKING_STOPWORDS should be a frozenset."""
        assert isinstance(BLOCKING_STOPWORDS, frozenset)


# ============================================================================
# Name tokens tests
# ============================================================================

class TestNameTokens:
    def test_basic_extraction(self):
        """Should extract meaningful tokens after filtering stopwords."""
        tokens = extract_meaningful_tokens("walmart inc")
        assert tokens == ["walmart"]

    def test_multiple_meaningful_tokens(self):
        """Should extract multiple meaningful tokens."""
        tokens = extract_meaningful_tokens("tata motors pvt ltd")
        # "motors" is a stopword, "pvt" and "ltd" are stopwords
        assert tokens == ["tata"]

    def test_all_stopwords_returns_empty(self):
        """If all tokens are stopwords, return empty list."""
        tokens = extract_meaningful_tokens("pvt ltd co")
        assert tokens == []

    def test_empty_input(self):
        """Empty or None input should return empty list."""
        assert extract_meaningful_tokens("") == []
        assert extract_meaningful_tokens(None) == []

    def test_preserves_order(self):
        """Tokens should be returned in original order."""
        tokens = extract_meaningful_tokens("acme technologies widget corp")
        # "technologies" and "corp" are stopwords
        assert tokens == ["acme", "widget"]

    def test_min_token_length_filter(self):
        """Tokens shorter than min_token_length should be filtered."""
        tokens = extract_meaningful_tokens("ab xyz tech", min_token_length=3)
        assert "ab" not in tokens
        assert "xyz" in tokens

    def test_preserves_business_specific_tokens(self):
        """Real business names should pass through."""
        tokens = extract_meaningful_tokens("reliance industries pvt ltd")
        assert tokens == ["reliance"]

    def test_us_business_name(self):
        """US business names should work correctly."""
        tokens = extract_meaningful_tokens("primary care group")
        # "care" and "group" are stopwords
        assert tokens == ["primary"]


# ============================================================================
# Address parser tests
# ============================================================================

class TestAddressParser:
    def test_simple_street_number(self):
        """Should extract a simple street number."""
        num, has = extract_address_number("108 norle street college twp pa")
        assert has is True
        assert num == "108"

    def test_compound_number(self):
        """Should extract compound numbers like 570/13."""
        num, has = extract_address_number("570/13 mg road bangalore ka")
        assert has is True
        assert num == "570/13"

    def test_no_number_in_address(self):
        """Address with no number should return (None, False)."""
        num, has = extract_address_number("suite b oak plaza")
        assert has is False
        assert num is None

    def test_none_address(self):
        """None address should return (None, False)."""
        num, has = extract_address_number(None)
        assert has is False
        assert num is None

    def test_empty_address(self):
        """Empty address should return (None, False)."""
        num, has = extract_address_number("")
        assert has is False
        assert num is None

    def test_hash_number(self):
        """Should extract numbers preceded by #."""
        num, has = extract_address_number("#402 main street")
        assert has is True
        assert num == "402"

    def test_has_address_number_convenience(self):
        """has_address_number should be a convenience wrapper."""
        assert has_address_number("108 main st") is True
        assert has_address_number("suite b") is False
        assert has_address_number(None) is False

    def test_zip_code_not_treated_as_address_number(self):
        """Zip codes (5-6 digits) should not be the extracted address number."""
        # If address only has a zip code, we should skip it
        num, has = extract_address_number("main street 560001")
        # The first number match should be checked against zip code pattern
        if has:
            # If a number is extracted, it shouldn't be the zip code
            assert len(num) < 5 or "/" in num or "-" in num


# ============================================================================
# Blocking keys tests
# ============================================================================

class TestBlockingKeys:
    @pytest.fixture
    def sample_record(self):
        return {
            "entity_id": "S1-001",
            "source": "source1",
            "business_name_normalized": "acme widget corp",
            "business_name_transliterated": "Acme Widget Corp",
            "business_address_normalized": "108 main street springfield il",
            "country_normalized": "united states",
        }

    def test_key_a(self, sample_record):
        """Key A should generate country + exact normalized name."""
        keys = key_a_exact_norm_name(sample_record)
        assert len(keys) == 1
        key = list(keys)[0]
        assert key.startswith("A||")
        assert "united states" in key
        assert "acme widget corp" in key

    def test_key_a_empty_name(self):
        """Key A should return empty set for empty name."""
        rec = {"country_normalized": "india", "business_name_normalized": ""}
        assert key_a_exact_norm_name(rec) == set()

    def test_key_b(self, sample_record):
        """Key B should generate country + exact transliterated name."""
        keys = key_b_exact_translit_name(sample_record)
        assert len(keys) == 1
        key = list(keys)[0]
        assert key.startswith("B||")
        assert "united states" in key

    def test_key_c(self, sample_record):
        """Key C should use first two meaningful tokens."""
        keys = key_c_first_two_meaningful_tokens(sample_record)
        assert len(keys) == 1
        key = list(keys)[0]
        assert key.startswith("C||")
        # "acme" and "widget" should be meaningful (corp is stopword)
        assert "acme" in key
        assert "widget" in key

    def test_key_c_insufficient_tokens(self):
        """Key C should return empty set if < 2 meaningful tokens."""
        rec = {
            "country_normalized": "india",
            "business_name_normalized": "pvt ltd",  # All stopwords
        }
        assert key_c_first_two_meaningful_tokens(rec) == set()

    def test_key_d(self, sample_record):
        """Key D should use first meaningful token + address number."""
        keys = key_d_first_token_addr_num(sample_record)
        assert len(keys) == 1
        key = list(keys)[0]
        assert key.startswith("D||")
        assert "acme" in key
        assert "108" in key

    def test_key_d_no_address_number(self):
        """Key D should return empty set if no address number."""
        rec = {
            "country_normalized": "india",
            "business_name_normalized": "acme enterprises pvt ltd",
            "business_address_normalized": "suite b oak plaza",
        }
        assert key_d_first_token_addr_num(rec) == set()

    def test_key_e_fallback(self):
        """Key E should activate only when no address number."""
        rec = {
            "country_normalized": "india",
            "business_name_normalized": "acme enterprises pvt ltd",
            "business_address_normalized": "suite b oak plaza",
        }
        keys = key_e_first_token_fallback(rec)
        assert len(keys) == 1
        key = list(keys)[0]
        assert key.startswith("E||")
        assert "acme" in key

    def test_key_e_not_activated_when_addr_num_exists(self, sample_record):
        """Key E should NOT activate when address number exists."""
        keys = key_e_first_token_fallback(sample_record)
        assert keys == set()

    def test_key_e_null_address(self):
        """Key E should activate when address is null."""
        rec = {
            "country_normalized": "india",
            "business_name_normalized": "acme enterprises pvt ltd",
            "business_address_normalized": None,
        }
        keys = key_e_first_token_fallback(rec)
        assert len(keys) == 1

    def test_key_f(self, sample_record):
        """Key F should generate country + exact normalized address."""
        keys = key_f_exact_norm_address(sample_record)
        assert len(keys) == 1
        key = list(keys)[0]
        assert key.startswith("F||")
        assert "108 main street springfield il" in key

    def test_no_country(self):
        """All keys should return empty set if country is missing."""
        rec = {"country_normalized": "", "business_name_normalized": "test"}
        assert key_a_exact_norm_name(rec) == set()

    def test_generate_all_keys(self, sample_record):
        """generate_all_keys should return keys for all labels plus ALL."""
        result = generate_all_keys(sample_record)
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" in result
        assert "E" in result  # Might be empty since addr num exists
        assert "F" in result
        assert "ALL" in result
        # ALL should be union of all individual keys
        individual_union = set()
        for label in ["A", "B", "C", "D", "E", "F"]:
            individual_union |= result[label]
        assert result["ALL"] == individual_union

    def test_key_d_and_e_mutual_exclusion(self, sample_record):
        """Key D and E should be mutually exclusive for a given record."""
        result = generate_all_keys(sample_record)
        # sample_record has address number 108
        assert len(result["D"]) > 0  # D should fire
        assert len(result["E"]) == 0  # E should NOT fire

    def test_blocking_key_names(self):
        """All key labels should have human-readable names."""
        for label in ["A", "B", "C", "D", "E", "F"]:
            assert label in BLOCKING_KEY_NAMES


# ============================================================================
# Block index tests
# ============================================================================

class TestBlockIndex:
    def test_basic_indexing(self):
        """Should index records and find matching pairs."""
        idx = BlockIndex()

        s1_rec = {
            "entity_id": "S1-001",
            "business_name_normalized": "acme widget corp",
            "business_name_transliterated": "acme widget corp",
            "business_address_normalized": "108 main st",
            "country_normalized": "united states",
        }
        cand_rec = {
            "entity_id": "S2-001",
            "business_name_normalized": "acme widget corp",
            "business_name_transliterated": "acme widget corp",
            "business_address_normalized": "108 main st",
            "country_normalized": "united states",
        }

        idx.add_s1_record("S1-001", s1_rec)
        idx.add_candidate_record("S2-001", cand_rec)

        pairs, prov = idx.generate_pairs()
        assert ("S1-001", "S2-001") in pairs
        assert len(prov[("S1-001", "S2-001")]) > 0

    def test_no_cross_country_pairs(self):
        """Records from different countries should not generate pairs."""
        idx = BlockIndex()

        s1_rec = {
            "entity_id": "S1-001",
            "business_name_normalized": "acme corp",
            "business_name_transliterated": "acme corp",
            "business_address_normalized": "108 main st",
            "country_normalized": "united states",
        }
        cand_rec = {
            "entity_id": "S2-001",
            "business_name_normalized": "acme corp",
            "business_name_transliterated": "acme corp",
            "business_address_normalized": "108 main st",
            "country_normalized": "india",
        }

        idx.add_s1_record("S1-001", s1_rec)
        idx.add_candidate_record("S2-001", cand_rec)

        pairs, _ = idx.generate_pairs()
        assert ("S1-001", "S2-001") not in pairs

    def test_oversized_block_detection(self):
        """Should detect oversized blocks without silently discarding."""
        idx = BlockIndex(max_block_size=2)

        # Add multiple S1 and cand records with same name to create large block
        for i in range(3):
            rec = {
                "entity_id": f"S1-{i:03d}",
                "business_name_normalized": "common name",
                "business_name_transliterated": "common name",
                "business_address_normalized": "",
                "country_normalized": "india",
            }
            idx.add_s1_record(rec["entity_id"], rec)

        for i in range(3):
            rec = {
                "entity_id": f"S2-{i:03d}",
                "business_name_normalized": "common name",
                "business_name_transliterated": "common name",
                "business_address_normalized": "",
                "country_normalized": "india",
            }
            idx.add_candidate_record(rec["entity_id"], rec)

        # Without capping, all pairs should be generated
        pairs, _ = idx.generate_pairs(cap_blocks=False)
        assert len(pairs) > 0

        # Oversized blocks should be detected
        assert len(idx.oversized_blocks) > 0

    def test_pair_deduplication(self):
        """Same pair from multiple keys should appear only once."""
        idx = BlockIndex()

        rec = {
            "entity_id": "S1-001",
            "business_name_normalized": "acme widget",
            "business_name_transliterated": "acme widget",
            "business_address_normalized": "108 main st",
            "country_normalized": "united states",
        }
        cand_rec = {
            "entity_id": "S2-001",
            "business_name_normalized": "acme widget",
            "business_name_transliterated": "acme widget",
            "business_address_normalized": "108 main st",
            "country_normalized": "united states",
        }

        idx.add_s1_record("S1-001", rec)
        idx.add_candidate_record("S2-001", cand_rec)

        pairs, prov = idx.generate_pairs()
        # Same pair should only appear once
        assert len([p for p in pairs if p == ("S1-001", "S2-001")]) == 1
        # But provenance should show multiple keys
        assert len(prov[("S1-001", "S2-001")]) > 1

    def test_recall_evaluation(self):
        """Should correctly compute recall against GT pairs."""
        idx = BlockIndex()

        s1_rec = {
            "entity_id": "S1-001",
            "business_name_normalized": "acme corp",
            "business_name_transliterated": "acme corp",
            "business_address_normalized": "108 main st",
            "country_normalized": "united states",
        }
        cand_rec = {
            "entity_id": "S2-001",
            "business_name_normalized": "acme corp",
            "business_name_transliterated": "acme corp",
            "business_address_normalized": "108 main st",
            "country_normalized": "united states",
        }

        idx.add_s1_record("S1-001", s1_rec)
        idx.add_candidate_record("S2-001", cand_rec)

        gt = {("S1-001", "S2-001")}
        result = idx.evaluate_recall(gt)
        assert result["blocking_recall"] == 1.0
        assert result["ground_truth_recovered"] == 1

    def test_get_stats(self):
        """get_stats should return expected statistics."""
        idx = BlockIndex(max_block_size=1000)

        rec = {
            "entity_id": "S1-001",
            "business_name_normalized": "test",
            "business_name_transliterated": "test",
            "business_address_normalized": "1 main st",
            "country_normalized": "india",
        }
        idx.add_s1_record("S1-001", rec)

        stats = idx.get_stats()
        assert stats["s1_entities"] == 1
        assert stats["max_block_size_setting"] == 1000
