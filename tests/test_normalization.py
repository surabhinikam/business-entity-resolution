import unittest
from src.normalization.text_normalizer import normalize_text, strip_diacritics


class TestTextNormalizer(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertIsNone(normalize_text(None))
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("   "), "")

    def test_strip_diacritics(self):
        # IAST transliteration characters
        self.assertEqual(strip_diacritics("prāiveṭa limiṭeḍa"), "praiveta limiteda")
        self.assertEqual(strip_diacritics("śrī gaṇēśa"), "sri ganesa")
        self.assertEqual(strip_diacritics("Café"), "Cafe")

    def test_normalize_text_basic(self):
        self.assertEqual(
            normalize_text("  ACME   Corporation  "),
            "acme corporation",
        )

    def test_normalize_text_with_punctuation_preservation(self):
        # preserve_safe_symbols keeps hyphen, slash, ampersand, hash
        res = normalize_text(
            "Plot No. 570/13, Sector-12 #402 & Co.!",
            strip_punctuation=True,
            preserve_safe_symbols=True,
        )
        self.assertEqual(res, "plot no 570/13 sector-12 #402 & co")

    def test_normalize_text_strip_all_punctuation(self):
        res = normalize_text(
            "U.S.A. / India - #1",
            strip_punctuation=True,
            preserve_safe_symbols=False,
        )
        self.assertEqual(res, "u s a india 1")


if __name__ == "__main__":
    unittest.main()
