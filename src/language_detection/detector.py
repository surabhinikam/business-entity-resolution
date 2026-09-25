"""
Language detection module for business entity records.

Provides offline script-informed language inference.
"""

from __future__ import annotations

from typing import Optional


# Script-to-language mapping for Indic scripts (which correlate closely with languages)
_SCRIPT_TO_LANGUAGE = {
    "Devanagari": "hindi",
    "Bengali": "bengali",
    "Gujarati": "gujarati",
    "Gurmukhi": "punjabi",
    "Kannada": "kannada",
    "Malayalam": "malayalam",
    "Odia": "odia",
    "Tamil": "tamil",
    "Telugu": "telugu",
    "Latin": "english",
    "Mixed": "mixed",
}


class LanguageDetector:
    """Offline language detector based on Unicode script and lexical cues."""

    def __init__(self):
        pass

    def detect(self, text: Optional[str], script: Optional[str] = None) -> Optional[str]:
        """
        Detect or infer language for a text given its detected script.
        Returns language code / name or None if input is empty or unknown.
        """
        if text is None:
            return None

        cleaned = text.strip()
        if not cleaned:
            return None

        if script and script in _SCRIPT_TO_LANGUAGE:
            return _SCRIPT_TO_LANGUAGE[script]

        return None
