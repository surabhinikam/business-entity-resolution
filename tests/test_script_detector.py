"""Unit tests for offline Unicode script detection with exact mixed-script logic."""

import unittest
from src.transliteration.script_detector import (
    ScriptDetector,
    detect_script,
    SUPPORTED_SCRIPTS,
)


class TestScriptDetector(unittest.TestCase):
    """Test suite for script detection across Indic, Latin, mixed, and edge cases."""

    def setUp(self):
        self.detector = ScriptDetector()

    def test_supported_scripts_list(self):
        """Verify all target scripts are present in the supported scripts list."""
        required = [
            "Latin",
            "Devanagari",
            "Bengali",
            "Gujarati",
            "Gurmukhi",
            "Kannada",
            "Malayalam",
            "Odia",
            "Tamil",
            "Telugu",
            "Mixed",
            "Unknown",
        ]
        for script in required:
            self.assertIn(script, SUPPORTED_SCRIPTS)

    def test_pure_latin(self):
        """Pure Latin strings (with spaces, numbers, punctuation) must return 'Latin'."""
        self.assertEqual(detect_script("Vision Partners Corp"), "Latin")
        self.assertEqual(detect_script("1064 Newton Rd, Unit 11, Iowa City, IA"), "Latin")
        self.assertEqual(detect_script("Thermal & Fils SASU, 20 Rue Parmentier"), "Latin")
        self.assertEqual(detect_script("Guerra And Krueger Table, LLC - (US)"), "Latin")

    def test_each_supported_indic_script(self):
        """Pure Indic strings must return their specific script name."""
        test_cases = [
            ("Devanagari", "राम मार्केटिंग प्राइवेट लिमिटेड"),
            ("Bengali", "গোল্ড প্রডিউসার স্টোর্স লিমিটেড"),
            ("Gujarati", "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ"),
            ("Gurmukhi", "ਗੁਰੂ ਨਾਨਕ ਐਂਟਰਪ੍ਰਾਈਜਿਜ਼"),
            ("Kannada", "ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"),
            ("Malayalam", "ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്"),
            ("Odia", "ଶ୍ରୀ ରାମ ଟ୍ରେଡର୍ସ"),
            ("Tamil", "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்"),
            ("Telugu", "కృష్ణా ఇంపెక్స్ లిమిటెడ్"),
        ]
        for expected_script, sample_text in test_cases:
            with self.subTest(script=expected_script):
                self.assertEqual(detect_script(sample_text), expected_script)

    def test_latin_plus_devanagari_mixed(self):
        """Strings containing both Latin and Devanagari must return 'Mixed'."""
        self.assertEqual(
            detect_script("ब्लू Technologies Pvt Ltd, Plot No: 3829, New Delhi"),
            "Mixed",
        )
        self.assertEqual(
            detect_script("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi, महाराष्ट्र"),
            "Mixed",
        )

    def test_latin_plus_tamil_mixed(self):
        """Strings containing both Latin and Tamil must return 'Mixed'."""
        self.assertEqual(
            detect_script("அரிஹந்த் Foundation Private Limited"),
            "Mixed",
        )
        self.assertEqual(
            detect_script("#44, KILATHOPE STREET, MADURAI, தமிழ்நாடு"),
            "Mixed",
        )

    def test_latin_plus_gujarati_mixed(self):
        """Strings containing both Latin and Gujarati must return 'Mixed'."""
        self.assertEqual(
            detect_script("Tech Food પ્રાઇવેટ લિમિટેડ"),
            "Mixed",
        )
        self.assertEqual(
            detect_script("PLOT 008, CHAITANYA IND AREA, KOTDA SANGHANI, ગુજરાત"),
            "Mixed",
        )

    def test_two_different_indic_scripts_mixed(self):
        """Strings containing characters from two distinct Indic scripts must return 'Mixed'."""
        # Devanagari + Bengali
        devanagari_and_bengali = "राम গোল্ড"
        self.assertEqual(detect_script(devanagari_and_bengali), "Mixed")

        # Tamil + Telugu
        tamil_and_telugu = "தமிழ் తెలుగు"
        self.assertEqual(detect_script(tamil_and_telugu), "Mixed")

    def test_punctuation_and_numbers_only(self):
        """Strings containing only digits, punctuation, and symbols must return 'Unknown'."""
        self.assertEqual(detect_script("1234567890"), "Unknown")
        self.assertEqual(detect_script("--- ,,, ... !!! @#$%^&*()"), "Unknown")
        self.assertEqual(detect_script("105 / 24, #12-"), "Unknown")
        self.assertEqual(detect_script("\t\r\n   "), "Unknown")

    def test_empty_string(self):
        """Empty or whitespace-only strings must return 'Unknown'."""
        self.assertEqual(detect_script(""), "Unknown")
        self.assertEqual(detect_script("   "), "Unknown")

    def test_none_value(self):
        """None input must return 'Unknown'."""
        self.assertEqual(detect_script(None), "Unknown")

    def test_detailed_detection_result(self):
        """Verify detailed detection contains counts, percentages, and dominant script."""
        res = self.detector.detect_detailed("அரிஹந்த் Foundation Private Limited")
        self.assertTrue(res.is_mixed)
        self.assertEqual(res.detected_script, "Mixed")
        self.assertEqual(res.dominant_script, "Latin")  # Dominant character count is Latin
        self.assertIn("Tamil", res.script_counts)
        self.assertIn("Latin", res.script_counts)
        self.assertGreater(res.total_meaningful_chars, 0)


if __name__ == "__main__":
    unittest.main()
