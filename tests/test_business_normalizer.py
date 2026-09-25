import unittest
from src.normalization.business_normalizer import (
    normalize_business_name,
    tokenize_business_name,
    canonicalize_legal_suffix,
)


class TestBusinessNormalizer(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertIsNone(normalize_business_name(None))
        self.assertEqual(normalize_business_name(""), "")
        self.assertEqual(tokenize_business_name(None), [])
        self.assertEqual(tokenize_business_name(""), [])

    def test_legal_suffix_canonicalization(self):
        cases = [
            ("Reliance Private Limited", "reliance pvt ltd"),
            ("Infosys Pvt. Ltd.", "infosys pvt ltd"),
            ("Tata Sons Ltd.", "tata sons ltd"),
            ("Alphabet Inc.", "alphabet inc"),
            ("Microsoft Corp.", "microsoft corp"),
            ("Acme Corporation", "acme corp"),
            ("Wipro Limited", "wipro ltd"),
            ("Global Services LLC", "global services llc"),
            ("Capital LLP", "capital llp"),
            ("Vodafone Public Limited", "vodafone plc"),
            ("Apex Co.", "apex co"),
        ]
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(normalize_business_name(original), expected)

    def test_transliterated_indic_suffixes(self):
        # After transliteration and diacritics stripping:
        # 'प्राइवेट लिमिटेड' -> 'praiveta limiteda' -> 'pvt ltd'
        self.assertEqual(
            normalize_business_name(
                name="अरिहंत प्राइवेट लिमिटेड",
                transliterated="arihanta praiveta limiteda",
            ),
            "arihanta pvt ltd",
        )
        # 'एलएलपी' -> 'elaelapi' -> 'llp'
        self.assertEqual(
            normalize_business_name(
                name="अरिहंत एलएलपी",
                transliterated="arihanta elaelapi",
            ),
            "arihanta llp",
        )

    def test_punctuation_and_symbols_in_name(self):
        # Preserves & and hyphen in company names
        self.assertEqual(
            normalize_business_name("Johnson & Johnson, Inc."),
            "johnson & johnson inc",
        )
        self.assertEqual(
            normalize_business_name("A-One Solutions Pvt Ltd"),
            "a-one solutions pvt ltd",
        )

    def test_tokenization(self):
        tokens = tokenize_business_name("johnson & johnson inc")
        self.assertEqual(tokens, ["johnson", "&", "johnson", "inc"])

        tokens2 = tokenize_business_name("a-one solutions pvt ltd")
        self.assertEqual(tokens2, ["a-one", "solutions", "pvt", "ltd"])


if __name__ == "__main__":
    unittest.main()
