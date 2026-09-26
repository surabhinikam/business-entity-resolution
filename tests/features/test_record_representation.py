"""Unit tests for the record-level business name representation layer."""

from __future__ import annotations

import math
import pytest
from dataclasses import FrozenInstanceError

from src.features.record_representation import (
    NameRepresentation,
    EMPTY_NAME_REPRESENTATION,
    build_name_representation,
)
from src.analysis.text_similarity import character_ngrams


class TestRecordRepresentation:
    """Test suite verifying NameRepresentation construction and behavior."""

    def test_normal_business_name(self):
        """1. Verify standard single/multi token business name."""
        rep = build_name_representation("google", ["google"])
        assert rep.clean_name == "google"
        assert rep.token_set == frozenset({"google"})
        assert rep.char_length == 6
        assert rep.token_count == 1
        assert rep.acronym == "g"
        assert len(rep.char_3grams) > 0

    def test_multiple_tokens(self):
        """2. Verify representation with multiple tokens."""
        rep = build_name_representation(
            "international business machines",
            ["international", "business", "machines"],
        )
        assert rep.clean_name == "international business machines"
        assert rep.token_set == frozenset({"international", "business", "machines"})
        assert rep.token_count == 3
        assert rep.acronym == "ibm"
        assert rep.char_length == 31

    def test_empty_string(self):
        """3. Verify empty string input returns empty representation."""
        rep1 = build_name_representation("", [])
        rep2 = build_name_representation("   ", None)
        assert rep1 == EMPTY_NAME_REPRESENTATION
        assert rep2 == EMPTY_NAME_REPRESENTATION
        assert rep1.clean_name == ""
        assert rep1.token_set == frozenset()
        assert rep1.char_3grams == frozenset()
        assert rep1.char_length == 0
        assert rep1.token_count == 0
        assert rep1.acronym == ""

    def test_none_input(self):
        """4. Verify None input returns empty representation without error."""
        rep = build_name_representation(None, None)
        assert rep == EMPTY_NAME_REPRESENTATION
        assert rep.clean_name == ""
        assert rep.token_count == 0

    def test_empty_token_list(self):
        """5. Verify passing explicit empty token list is respected."""
        rep = build_name_representation("acme corp", [])
        assert rep.clean_name == "acme corp"
        assert rep.token_set == frozenset()
        assert rep.token_count == 0
        assert rep.acronym == ""
        assert rep.char_length == 9
        # Char n-grams are still generated from clean_name
        assert len(rep.char_3grams) > 0

    def test_tokens_containing_empty_or_none_values(self):
        """6. Verify filtering of None, empty strings, and sentinel values in tokens."""
        rep = build_name_representation(
            "acme enterprises pvt ltd",
            ["acme", None, "", "  ", "enterprises", "None", "nan", "pvt", "ltd"],
        )
        assert rep.clean_name == "acme enterprises pvt ltd"
        assert rep.token_set == frozenset({"acme", "enterprises", "pvt", "ltd"})
        assert rep.token_count == 4
        assert rep.acronym == "aepl"

    def test_character_3gram_generation(self):
        """7. Verify character 3-grams match character_ngrams from text_similarity."""
        name = "tata consultancy"
        rep = build_name_representation(name, ["tata", "consultancy"])
        expected_3grams = frozenset(character_ngrams(name, n=3))
        assert rep.char_3grams == expected_3grams
        assert " ta" in rep.char_3grams
        assert "tat" in rep.char_3grams

    def test_correct_token_count(self):
        """8. Verify token count counts all valid tokens, including duplicates."""
        rep = build_name_representation("tata tata motors", ["tata", "tata", "motors"])
        assert rep.token_count == 3
        # token_set contains distinct tokens
        assert rep.token_set == frozenset({"tata", "motors"})
        assert len(rep.token_set) == 2

    def test_correct_character_length(self):
        """9. Verify char_length matches len(clean_name)."""
        name = "shree ram traders"
        rep = build_name_representation(name, ["shree", "ram", "traders"])
        assert rep.char_length == len(name)
        assert rep.char_length == 17

    def test_acronym_generation(self):
        """10. Verify acronym formation from ordered tokens."""
        cases = [
            (["state", "bank", "of", "india"], "sboi"),
            (["bharat", "heavy", "electricals", "limited"], "bhel"),
            (["3m", "company"], "3c"),
            (["a"], "a"),
        ]
        for tokens, expected_acronym in cases:
            rep = build_name_representation(" ".join(tokens), tokens)
            assert rep.acronym == expected_acronym

    def test_deterministic_output(self):
        """11. Verify representation is strictly deterministic across repeated invocations."""
        rep1 = build_name_representation("reliance retail", ["reliance", "retail"])
        rep2 = build_name_representation("reliance retail", ["reliance", "retail"])
        assert rep1 == rep2
        assert hash(rep1) == hash(rep2)
        assert rep1.token_set == rep2.token_set
        assert rep1.char_3grams == rep2.char_3grams

    def test_missing_name_does_not_create_fake_tokens_or_ngrams(self):
        """12. Verify sentinel string values ('None', 'nan', 'null') do not create fake tokens/ngrams."""
        sentinels = ["None", "none", "nan", "NaN", "null", "NULL", float("nan")]
        for val in sentinels:
            rep = build_name_representation(val, ["None", "nan"])
            assert rep == EMPTY_NAME_REPRESENTATION
            assert rep.clean_name == ""
            assert rep.token_set == frozenset()
            assert rep.char_3grams == frozenset()
            assert rep.char_length == 0
            assert rep.token_count == 0
            assert rep.acronym == ""

    def test_immutability(self):
        """Verify NameRepresentation instances are frozen."""
        rep = build_name_representation("infosys", ["infosys"])
        with pytest.raises(FrozenInstanceError):
            rep.clean_name = "other"  # type: ignore

    def test_token_fallback_when_tokens_none(self):
        """Verify fallback to splitting clean_name when tokens argument is None."""
        rep = build_name_representation("apple computer inc", None)
        assert rep.clean_name == "apple computer inc"
        assert rep.token_set == frozenset({"apple", "computer", "inc"})
        assert rep.token_count == 3
        assert rep.acronym == "aci"
