"""Unit tests for the pairwise business-name feature extraction layer."""

from __future__ import annotations

import pytest

from src.features.record_representation import (
    build_name_representation,
    EMPTY_NAME_REPRESENTATION,
)
from src.features.name_features import (
    compute_name_features,
    extract_legal_suffix,
    CANONICAL_LEGAL_SUFFIXES,
)
from src.normalization.business_vocabulary import LEGAL_SUFFIX_RULES


EXPECTED_FEATURE_KEYS = (
    "name_exact_norm",
    "name_exact_translit",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_dice",
    "name_token_count_diff",
    "name_char_3gram_jaccard",
    "name_char_len_diff",
    "name_char_len_ratio",
    "name_first_token_exact",
    "name_acronym_match",
    "name_legal_suffix_match",
)


class TestNameFeatures:
    """Test suite verifying pairwise business name similarity feature computation."""

    def test_feature_keys_and_types_stable(self):
        """Verify all 12 expected keys are returned with exact numeric types."""
        rep = build_name_representation("google inc", ["google", "inc"])
        feats = compute_name_features(rep, rep, translit_a="google inc", translit_b="google inc")

        assert tuple(feats.keys()) == EXPECTED_FEATURE_KEYS
        for key in (
            "name_exact_norm",
            "name_exact_translit",
            "name_token_count_diff",
            "name_char_len_diff",
            "name_first_token_exact",
            "name_acronym_match",
            "name_legal_suffix_match",
        ):
            assert isinstance(feats[key], int), f"{key} must be int"

        for key in (
            "name_token_jaccard",
            "name_token_overlap",
            "name_token_dice",
            "name_char_3gram_jaccard",
            "name_char_len_ratio",
        ):
            assert isinstance(feats[key], float), f"{key} must be float"

    def test_case_a_identical_names(self):
        """A. Identical names should have maximal similarities."""
        rep_a = build_name_representation("acme industries pvt ltd", ["acme", "industries", "pvt", "ltd"])
        rep_b = build_name_representation("acme industries pvt ltd", ["acme", "industries", "pvt", "ltd"])
        feats = compute_name_features(rep_a, rep_b, translit_a="acme industries pvt ltd", translit_b="acme industries pvt ltd")

        assert feats["name_exact_norm"] == 1
        assert feats["name_exact_translit"] == 1
        assert feats["name_token_jaccard"] == 1.0
        assert feats["name_token_overlap"] == 1.0
        assert feats["name_token_dice"] == 1.0
        assert feats["name_token_count_diff"] == 0
        assert feats["name_char_3gram_jaccard"] == 1.0
        assert feats["name_char_len_diff"] == 0
        assert feats["name_char_len_ratio"] == 1.0
        assert feats["name_first_token_exact"] == 1
        assert feats["name_acronym_match"] == 1
        assert feats["name_legal_suffix_match"] == 1

    def test_case_b_completely_different_names(self):
        """B. Completely different names should have zero similarity."""
        rep_a = build_name_representation("tata steel", ["tata", "steel"])
        rep_b = build_name_representation("wipro technology", ["wipro", "technology"])
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_exact_norm"] == 0
        assert feats["name_token_jaccard"] == 0.0
        assert feats["name_token_overlap"] == 0.0
        assert feats["name_token_dice"] == 0.0
        assert feats["name_char_3gram_jaccard"] == 0.0
        assert feats["name_first_token_exact"] == 0
        assert feats["name_acronym_match"] == 0
        assert feats["name_legal_suffix_match"] == 0

    def test_case_c_partial_token_overlap(self):
        """C. Partial token overlap (subset vs superset)."""
        rep_a = build_name_representation("shree ram traders", ["shree", "ram", "traders"])
        rep_b = build_name_representation("shree ram", ["shree", "ram"])
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_exact_norm"] == 0
        # 2 shared tokens out of 3 total -> Jaccard = 2/3
        assert pytest.approx(feats["name_token_jaccard"], 0.001) == 2 / 3
        # Subet containment -> Overlap = 2 / min(3, 2) = 1.0
        assert feats["name_token_overlap"] == 1.0
        # Dice = 2 * 2 / (3 + 2) = 4/5 = 0.8
        assert pytest.approx(feats["name_token_dice"], 0.001) == 0.8
        assert feats["name_first_token_exact"] == 1
        assert feats["name_token_count_diff"] == 1

    def test_case_d_empty_names(self):
        """D. Empty names should not produce false positive matches."""
        rep_a = EMPTY_NAME_REPRESENTATION
        rep_b = EMPTY_NAME_REPRESENTATION
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_exact_norm"] == 0
        assert feats["name_token_jaccard"] == 0.0
        assert feats["name_token_overlap"] == 0.0
        assert feats["name_token_dice"] == 0.0
        assert feats["name_char_3gram_jaccard"] == 0.0
        assert feats["name_char_len_ratio"] == 0.0
        assert feats["name_first_token_exact"] == 0
        assert feats["name_acronym_match"] == 0
        assert feats["name_legal_suffix_match"] == 0

    def test_case_e_none_transliteration_both_sides(self):
        """E. None transliteration on both sides must NOT count as a match."""
        rep_a = build_name_representation("acme", ["acme"])
        rep_b = build_name_representation("acme", ["acme"])
        feats = compute_name_features(rep_a, rep_b, translit_a=None, translit_b=None)

        assert feats["name_exact_translit"] == 0

    def test_case_f_one_side_missing_transliteration(self):
        """F. Only one side missing transliteration should not match."""
        rep_a = build_name_representation("apple", ["apple"])
        rep_b = build_name_representation("apple", ["apple"])
        feats1 = compute_name_features(rep_a, rep_b, translit_a="apple", translit_b=None)
        feats2 = compute_name_features(rep_a, rep_b, translit_a=None, translit_b="apple")

        assert feats1["name_exact_translit"] == 0
        assert feats2["name_exact_translit"] == 0

    def test_case_g_one_token_vs_multi_token(self):
        """G. One-token vs multi-token names."""
        rep_a = build_name_representation("target", ["target"])
        rep_b = build_name_representation("target stores corp", ["target", "stores", "corp"])
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_first_token_exact"] == 1
        assert feats["name_token_count_diff"] == 2
        assert feats["name_token_overlap"] == 1.0
        assert feats["name_acronym_match"] == 0

    def test_case_h_identical_acronyms(self):
        """H. Identical non-empty acronyms."""
        rep_a = build_name_representation("international business machines", ["international", "business", "machines"])
        rep_b = build_name_representation("india bakery mart", ["india", "bakery", "mart"])
        feats = compute_name_features(rep_a, rep_b)

        assert rep_a.acronym == "ibm"
        assert rep_b.acronym == "ibm"
        assert feats["name_acronym_match"] == 1
        assert feats["name_first_token_exact"] == 0

    def test_case_i_different_acronyms(self):
        """I. Different acronyms."""
        rep_a = build_name_representation("state bank india", ["state", "bank", "india"])
        rep_b = build_name_representation("punjab national bank", ["punjab", "national", "bank"])
        feats = compute_name_features(rep_a, rep_b)

        assert rep_a.acronym == "sbi"
        assert rep_b.acronym == "pnb"
        assert feats["name_acronym_match"] == 0

    def test_case_j_identical_3gram_sets(self):
        """J. Identical 3-gram sets."""
        rep_a = build_name_representation("infosys", ["infosys"])
        rep_b = build_name_representation("infosys", ["infosys"])
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_char_3gram_jaccard"] == 1.0

    def test_case_k_disjoint_3gram_sets(self):
        """K. Completely disjoint 3-gram sets."""
        rep_a = build_name_representation("aaa", ["aaa"])
        rep_b = build_name_representation("zzz", ["zzz"])
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_char_3gram_jaccard"] == 0.0

    def test_case_l_different_token_counts(self):
        """L. Different token counts check."""
        rep_a = build_name_representation("a b c d", ["a", "b", "c", "d"])
        rep_b = build_name_representation("a b", ["a", "b"])
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_token_count_diff"] == 2

    def test_case_m_zero_length_names(self):
        """M. Zero-length names."""
        rep_a = build_name_representation("", [])
        rep_b = build_name_representation("test", ["test"])
        feats = compute_name_features(rep_a, rep_b)

        assert feats["name_char_len_diff"] == 4
        assert feats["name_char_len_ratio"] == 0.0
        assert feats["name_token_count_diff"] == 1
        assert feats["name_first_token_exact"] == 0

    def test_case_n_legal_suffix_agreement(self):
        """N. Legal suffix agreement across canonical forms."""
        cases = [
            ("acme solutions pvt ltd", "apex global pvt ltd", "pvt ltd"),
            ("amazon inc", "apple inc", "inc"),
            ("tata steel ltd", "jsw steel ltd", "ltd"),
            ("deloitte llp", "kpmg llp", "llp"),
            ("google llc", "meta llc", "llc"),
            ("ford corp", "sony corp", "corp"),
            ("shell plc", "bp plc", "plc"),
            ("disney co", "ford co", "co"),
        ]
        for name_a, name_b, expected_suffix in cases:
            rep_a = build_name_representation(name_a, name_a.split())
            rep_b = build_name_representation(name_b, name_b.split())
            assert extract_legal_suffix(name_a) == expected_suffix
            assert extract_legal_suffix(name_b) == expected_suffix
            feats = compute_name_features(rep_a, rep_b)
            assert feats["name_legal_suffix_match"] == 1

    def test_all_canonical_targets_from_legal_suffix_rules_recognized(self):
        """Verify every canonical target produced by LEGAL_SUFFIX_RULES is recognized."""
        for _, canonical in LEGAL_SUFFIX_RULES:
            assert canonical in CANONICAL_LEGAL_SUFFIXES
            name = f"test enterprise {canonical}"
            assert extract_legal_suffix(name) == canonical

    def test_pvt_ltd_detected_before_ltd(self):
        """Verify compound 'pvt ltd' is detected before single-token 'ltd'."""
        name = "acme enterprises pvt ltd"
        assert extract_legal_suffix(name) == "pvt ltd"
        assert extract_legal_suffix(name) != "ltd"

    def test_normalized_variants_handled_through_canonical_form(self):
        """Verify normalized variants (e.g. private limited -> pvt ltd) match through canonical form."""
        # Both 'Acme Private Limited' and 'Acme Pvt Limited' normalize to 'acme pvt ltd'
        rep_a = build_name_representation("acme pvt ltd", ["acme", "pvt", "ltd"])
        rep_b = build_name_representation("apex pvt ltd", ["apex", "pvt", "ltd"])
        assert extract_legal_suffix(rep_a.clean_name) == "pvt ltd"
        assert extract_legal_suffix(rep_b.clean_name) == "pvt ltd"
        feats = compute_name_features(rep_a, rep_b)
        assert feats["name_legal_suffix_match"] == 1

    def test_suffixes_not_in_rules_not_treated_as_legal_suffixes(self):
        """Verify suffixes not in LEGAL_SUFFIX_RULES (e.g. sarl, sas, sasu, sa, gmbh) are not recognized."""
        non_canonical = ["sarl", "sas", "sasu", "sa", "gmbh", "ag", "bv"]
        for suffix in non_canonical:
            assert suffix not in CANONICAL_LEGAL_SUFFIXES
            name = f"acme enterprise {suffix}"
            assert extract_legal_suffix(name) is None

            rep_a = build_name_representation(f"acme enterprise {suffix}", ["acme", "enterprise", suffix])
            rep_b = build_name_representation(f"apex enterprise {suffix}", ["apex", "enterprise", suffix])
            feats = compute_name_features(rep_a, rep_b)
            assert feats["name_legal_suffix_match"] == 0

    def test_boundary_aware_token_checks(self):
        """Verify boundary awareness: 'zinc' != 'inc', 'balsa' != 'sa', 'coca' != 'co'."""
        assert extract_legal_suffix("zinc") is None
        assert extract_legal_suffix("balsa") is None
        assert extract_legal_suffix("coca") is None
        assert extract_legal_suffix("incorporation") is None
        assert extract_legal_suffix("companywide") is None
        # Whole token matches should succeed
        assert extract_legal_suffix("acme zinc inc") == "inc"
        assert extract_legal_suffix("acme building corp") == "corp"

    def test_empty_suffix_does_not_create_match(self):
        """Verify empty names or names without suffixes return None and match = 0."""
        assert extract_legal_suffix("") is None
        assert extract_legal_suffix("   ") is None
        rep_a = build_name_representation("acme", ["acme"])
        rep_b = build_name_representation("apex", ["apex"])
        assert extract_legal_suffix(rep_a.clean_name) is None
        assert extract_legal_suffix(rep_b.clean_name) is None
        feats = compute_name_features(rep_a, rep_b)
        assert feats["name_legal_suffix_match"] == 0

    def test_case_o_legal_suffix_disagreement(self):
        """O. Legal suffix disagreement or missing suffix."""
        rep_a = build_name_representation("acme pvt ltd", ["acme", "pvt", "ltd"])
        rep_b = build_name_representation("acme llp", ["acme", "llp"])
        feats1 = compute_name_features(rep_a, rep_b)
        assert feats1["name_legal_suffix_match"] == 0

        # One side has suffix, other side has no suffix
        rep_c = build_name_representation("acme", ["acme"])
        feats2 = compute_name_features(rep_a, rep_c)
        assert feats2["name_legal_suffix_match"] == 0

        # Neither side has a suffix
        rep_d = build_name_representation("apex", ["apex"])
        feats3 = compute_name_features(rep_c, rep_d)
        assert feats3["name_legal_suffix_match"] == 0

    def test_case_p_deterministic_repeated_calls(self):
        """P. Repeated invocations produce strictly identical results."""
        rep_a = build_name_representation("alpha beta gamma pvt ltd", ["alpha", "beta", "gamma", "pvt", "ltd"])
        rep_b = build_name_representation("alpha gamma corp", ["alpha", "gamma", "corp"])

        feats1 = compute_name_features(rep_a, rep_b, translit_a="alpha beta", translit_b="alpha gamma")
        feats2 = compute_name_features(rep_a, rep_b, translit_a="alpha beta", translit_b="alpha gamma")

        assert feats1 == feats2
        assert hash(tuple(feats1.items())) == hash(tuple(feats2.items()))

    def test_empty_token_sets_do_not_create_similarity_of_one(self):
        """Verify empty token sets yield 0.0 similarity, NOT 1.0."""
        rep_a = build_name_representation("a", [])
        rep_b = build_name_representation("b", [])
        assert rep_a.token_set == frozenset()
        assert rep_b.token_set == frozenset()

        feats = compute_name_features(rep_a, rep_b)
        assert feats["name_token_jaccard"] == 0.0
        assert feats["name_token_overlap"] == 0.0
        assert feats["name_token_dice"] == 0.0

    def test_empty_acronyms_do_not_match(self):
        """Verify two empty acronyms yield name_acronym_match = 0."""
        rep_a = build_name_representation("a", [])
        rep_b = build_name_representation("b", [])
        assert rep_a.acronym == ""
        assert rep_b.acronym == ""

        feats = compute_name_features(rep_a, rep_b)
        assert feats["name_acronym_match"] == 0
