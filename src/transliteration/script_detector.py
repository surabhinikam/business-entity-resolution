"""Offline Unicode-based script detection for multilingual entity resolution."""

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SUPPORTED_SCRIPTS = [
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

# Primary Unicode blocks for target Indic scripts
INDIC_SCRIPT_RANGES: List[Tuple[str, int, int]] = [
    ("Devanagari", 0x0900, 0x097F),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
]


@dataclass
class ScriptDetectionResult:
    """Detailed script detection results."""
    detected_script: str  # Single script name, "Mixed" if multiple scripts, or "Unknown"
    dominant_script: str  # The script with highest character count (or "Unknown")
    script_counts: Dict[str, int]
    script_percentages: Dict[str, float]
    total_meaningful_chars: int
    is_mixed: bool


class ScriptDetector:
    """Detects scripts from meaningful Unicode characters (letters and marks).
    
    Rules:
    - If exactly one meaningful script is present -> returns that script.
    - If two or more meaningful scripts are present -> returns 'Mixed'.
    - Ignores punctuation, digits, whitespace, and non-letter/non-mark characters.
    - Returns 'Unknown' for strings containing no meaningful script characters.
    """

    @staticmethod
    def identify_char_script(char: str) -> Optional[str]:
        """Identify the script of an individual character if it is a meaningful alphabetic/mark character."""
        # Include Letters (L*) and Combining Marks / Matras (M*)
        cat = unicodedata.category(char)
        if not (cat.startswith("L") or cat.startswith("M")):
            return None

        cp = ord(char)

        # Check Indic blocks
        for script_name, start, end in INDIC_SCRIPT_RANGES:
            if start <= cp <= end:
                return script_name

        # Check Latin ranges (Basic, Latin-1, Extended-A, Extended-B, Extended Additional)
        if (
            (0x0041 <= cp <= 0x005A) or  # A-Z
            (0x0061 <= cp <= 0x007A) or  # a-z
            (0x00C0 <= cp <= 0x024F) or  # Latin-1 Supp & Extended A/B
            (0x1E00 <= cp <= 0x1EFF)     # Latin Extended Additional
        ):
            return "Latin"

        # Other alphabetic characters outside the primary supported set
        return "Other"

    def detect_detailed(self, text: Optional[str]) -> ScriptDetectionResult:
        """Analyze text and return comprehensive script distribution statistics."""
        if not text or not isinstance(text, str):
            return ScriptDetectionResult(
                detected_script="Unknown",
                dominant_script="Unknown",
                script_counts={},
                script_percentages={},
                total_meaningful_chars=0,
                is_mixed=False,
            )

        counts: Counter = Counter()
        for char in text:
            script = self.identify_char_script(char)
            if script:
                counts[script] += 1

        total = sum(counts.values())
        if total == 0:
            return ScriptDetectionResult(
                detected_script="Unknown",
                dominant_script="Unknown",
                script_counts={},
                script_percentages={},
                total_meaningful_chars=0,
                is_mixed=False,
            )

        percentages = {k: round(v / total, 4) for k, v in counts.items()}
        distinct_scripts = list(counts.keys())
        top_script, _ = counts.most_common(1)[0]

        if len(distinct_scripts) == 1:
            detected = distinct_scripts[0]
            is_mixed = False
        else:
            detected = "Mixed"
            is_mixed = True

        return ScriptDetectionResult(
            detected_script=detected,
            dominant_script=top_script,
            script_counts=dict(counts),
            script_percentages=percentages,
            total_meaningful_chars=total,
            is_mixed=is_mixed,
        )

    def detect(self, text: Optional[str]) -> str:
        """Return the script classification: single script name, 'Mixed', or 'Unknown'."""
        return self.detect_detailed(text).detected_script


# Default detector singleton instance
_default_detector = ScriptDetector()


def detect_script(text: Optional[str]) -> str:
    """Convenience helper to detect script using default settings."""
    return _default_detector.detect(text)
