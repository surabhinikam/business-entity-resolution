"""Transliteration package exports."""

from src.transliteration.script_detector import (
    SUPPORTED_SCRIPTS,
    ScriptDetector,
    ScriptDetectionResult,
    detect_script,
)
from src.transliteration.transliterator import (
    BaseTransliterator,
    IndicTransliterator,
    transliterate_text,
    SCRIPT_TO_SANSCRIPT_SCHEME,
)

__all__ = [
    "SUPPORTED_SCRIPTS",
    "ScriptDetector",
    "ScriptDetectionResult",
    "detect_script",
    "BaseTransliterator",
    "IndicTransliterator",
    "transliterate_text",
    "SCRIPT_TO_SANSCRIPT_SCHEME",
]
