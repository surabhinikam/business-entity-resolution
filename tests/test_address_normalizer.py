import unittest
from src.normalization.address_normalizer import (
    normalize_address,
    tokenize_address,
)


class TestAddressNormalizer(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertIsNone(normalize_address(None))
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(tokenize_address(None), [])
        self.assertEqual(tokenize_address(""), [])

    def test_preserve_alphanumeric_and_structural_patterns(self):
        # Must preserve 570/13, 12-A, #402, plot numbers, etc.
        raw = "Plot No. 3829, Flat No: 570/13, Sector 12-A, #402"
        normalized = normalize_address(raw)
        self.assertEqual(
            normalized,
            "plot no 3829 flat no 570/13 sector 12-a #402",
        )

    def test_transliterated_indic_address(self):
        # Mixed script transliterated address
        raw_transliterated = "12-b, mg road, baṅgalōru, karnāṭaka - 560001"
        normalized = normalize_address(
            address="12-B, MG Road, ಬೆಂಗಳೂರು, ಕರ್ನಾಟಕ - 560001",
            transliterated=raw_transliterated,
        )
        self.assertEqual(
            normalized,
            "12-b mg road bangaloru karnataka - 560001",
        )

    def test_tokenization_preserves_segments(self):
        addr = "plot no 3829 flat no 570/13 sector 12-a #402"
        tokens = tokenize_address(addr)
        self.assertIn("570/13", tokens)
        self.assertIn("12-a", tokens)
        self.assertIn("#402", tokens)
        self.assertIn("3829", tokens)
        self.assertIn("no", tokens)


if __name__ == "__main__":
    unittest.main()
