"""Unit tests for offline Indic-to-Latin transliteration."""

import unittest
from unittest.mock import patch
import socket

from src.transliteration.transliterator import (
    IndicTransliterator,
    transliterate_text,
    BaseTransliterator,
)


class TestTransliterator(unittest.TestCase):
    """Test suite for offline transliteration abstraction and IndicTransliterator."""

    def setUp(self):
        self.transliterator = IndicTransliterator()

    def test_implements_interface(self):
        """Ensure IndicTransliterator satisfies the BaseTransliterator abstraction."""
        self.assertIsInstance(self.transliterator, BaseTransliterator)

    def test_latin_passthrough(self):
        """Latin strings must pass through completely unchanged."""
        samples = [
            "Vision Partners Corp",
            "1064 Newton Rd, Unit 11, Iowa City, IA",
            "Thermal & Fils SASU, 20 Rue Parmentier",
            "Guerra And Krueger Table, LLC",
        ]
        for s in samples:
            self.assertEqual(self.transliterator.transliterate(s), s)
            self.assertEqual(transliterate_text(s, script="Latin"), s)

    def test_supported_indic_scripts(self):
        """Test transliteration across all 9 supported Indic scripts."""
        test_cases = [
            ("Devanagari", "राम मार्केटिंग", "rāma"),
            ("Bengali", "গোল্ড প্রডিউসার", "golḍa"),
            ("Gujarati", "શક્તિ અર્બન", "śakti"),
            ("Gurmukhi", "ਗੁਰੂ ਨਾਨਕ", "gurū"),
            ("Kannada", "ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್", "māḍarn"),
            ("Malayalam", "ശക്തി ഇംപെക്സ്", "śakti"),
            ("Odia", "ଶ୍ରୀ ରାମ", "śrī"),
            ("Tamil", "குளோபல் பிசினஸ்", "ghuḻobhal"),
            ("Telugu", "కృష్ణా ఇంపెక్స్", "kṛṣṇā"),
        ]
        for script_name, indic_text, expected_token in test_cases:
            with self.subTest(script=script_name):
                result = self.transliterator.transliterate(indic_text, script=script_name)
                self.assertIsNotNone(result)
                # Verify that no Indic characters remain
                has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in result)
                self.assertFalse(has_indic, f"Indic characters remained in transliteration of {indic_text}: {result}")
                # Verify expected romanized token is present
                self.assertIn(expected_token, result)

    def test_mixed_script(self):
        """Mixed Latin and Indic text must preserve Latin words while romanizing Indic words."""
        mixed_text = "Tech Food પ્રાઇવેટ લિમિટેડ"
        result = self.transliterator.transliterate(mixed_text)
        self.assertTrue(result.startswith("Tech Food "))
        self.assertIn("prāiveṭa", result)

        tamil_mixed = "அரிஹந்த் Foundation Private Limited"
        res_tamil = self.transliterator.transliterate(tamil_mixed)
        self.assertIn("Foundation Private Limited", res_tamil)

    def test_empty_and_none(self):
        """Verify empty strings and None are preserved."""
        self.assertIsNone(self.transliterator.transliterate(None))
        self.assertEqual(self.transliterator.transliterate(""), "")

    def test_punctuation_and_numbers(self):
        """Punctuation, digits, and special characters must remain intact."""
        raw = "12345 / 67890 - #@!& (No. 42)"
        result = self.transliterator.transliterate(raw)
        self.assertEqual(result, raw)

    def test_preservation_of_original_input(self):
        """Ensure input string object is not modified in place."""
        original = "राम मार्केटिंग प्राइवेट लिमिटेड"
        original_copy = str(original)
        _ = self.transliterator.transliterate(original)
        self.assertEqual(original, original_copy)

    def test_strictly_offline_no_network_calls(self):
        """Verify that transliteration execution triggers zero socket/network activity."""
        # Block socket.create_connection and socket.socket.connect
        with patch.object(socket, "socket", side_effect=RuntimeError("Network access forbidden")):
            result = self.transliterator.transliterate("राम मार्केटिंग प्राइवेट लिमिटेड")
            self.assertIsNotNone(result)
            self.assertIn("rāma", result)


if __name__ == "__main__":
    unittest.main()
