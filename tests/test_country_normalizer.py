import unittest
from src.normalization.country_normalizer import normalize_country


class TestCountryNormalizer(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertIsNone(normalize_country(None))
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country("   "), "")

    def test_india_aliases(self):
        cases = ["IN", "in", "IND", "India", "india", "Republic of India", "Bharat"]
        for c in cases:
            with self.subTest(c=c):
                self.assertEqual(normalize_country(c), "india")

    def test_usa_aliases(self):
        cases = ["US", "USA", "U.S.A.", "United States", "United States of America"]
        for c in cases:
            with self.subTest(c=c):
                self.assertEqual(normalize_country(c), "united states")

    def test_france_aliases(self):
        cases = ["FR", "FRA", "France", "france", "République Française"]
        for c in cases:
            with self.subTest(c=c):
                self.assertEqual(normalize_country(c), "france")

    def test_unknown_country_fallback(self):
        # Graceful normalization without losing information
        self.assertEqual(normalize_country("Japan"), "japan")
        self.assertEqual(normalize_country("BR"), "br")
        self.assertEqual(normalize_country("South Africa"), "south africa")


if __name__ == "__main__":
    unittest.main()
