"""Unit tests for text similarity and pairwise feature extraction utilities."""

import unittest

from src.analysis.text_similarity import (
    jaccard_similarity,
    overlap_coefficient,
    dice_coefficient,
    extract_numeric_tokens,
    character_ngram_jaccard,
    levenshtein_similarity,
    calculate_pair_features,
)


class TestAnalysisUtils(unittest.TestCase):

    def test_jaccard_similarity(self):
        self.assertAlmostEqual(jaccard_similarity(["a", "b"], ["b", "c"]), 1 / 3)
        self.assertEqual(jaccard_similarity([], []), 1.0)
        self.assertEqual(jaccard_similarity(["a"], []), 0.0)

    def test_overlap_coefficient(self):
        # When one set is a subset of another, overlap should be 1.0
        self.assertEqual(overlap_coefficient(["apple", "inc"], ["apple"]), 1.0)
        self.assertEqual(overlap_coefficient(["a", "b"], ["c", "d"]), 0.0)

    def test_dice_coefficient(self):
        self.assertAlmostEqual(dice_coefficient(["a", "b"], ["b", "c"]), 2 / 4)

    def test_extract_numeric_tokens(self):
        text = "Plot No. 570/13, Suite 402, 12-A Baker St"
        nums = extract_numeric_tokens(text)
        self.assertIn("570/13", nums)
        self.assertIn("402", nums)
        self.assertIn("12-a", nums)

    def test_character_ngram_jaccard(self):
        score_exact = character_ngram_jaccard("google", "google")
        self.assertEqual(score_exact, 1.0)
        score_typo = character_ngram_jaccard("google", "gogle")
        self.assertGreater(score_typo, 0.5)

    def test_levenshtein_similarity(self):
        self.assertEqual(levenshtein_similarity("amazon", "amazon"), 1.0)
        self.assertAlmostEqual(levenshtein_similarity("amazon", "amazn"), 5 / 6)

    def test_calculate_pair_features(self):
        s1 = {
            "entity_id": "S1-1",
            "business_name": "Acme Corp",
            "business_name_normalized": "acme corp",
            "business_name_tokens": ["acme", "corp"],
            "business_address": "123 Main St",
            "business_address_normalized": "123 main st",
            "business_address_tokens": ["123", "main", "st"],
            "country": "US",
            "country_normalized": "united states",
        }
        s2 = {
            "entity_id": "S2-1",
            "source": "source2",
            "business_name": "Acme Corporation",
            "business_name_normalized": "acme corporation",
            "business_name_tokens": ["acme", "corporation"],
            "business_address": "123 Main Street",
            "business_address_normalized": "123 main street",
            "business_address_tokens": ["123", "main", "street"],
            "country": "US",
            "country_normalized": "united states",
        }
        feats = calculate_pair_features(s1, s2)
        self.assertEqual(feats["country_match_norm"], 1)
        self.assertEqual(feats["has_shared_num"], 1)
        self.assertIn("123", extract_numeric_tokens(s1["business_address"]))
        self.assertGreater(feats["name_token_overlap"], 0.0)


if __name__ == "__main__":
    unittest.main()
