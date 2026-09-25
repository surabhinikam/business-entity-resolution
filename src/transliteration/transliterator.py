"""Offline transliteration abstraction and Indic-to-Latin transliterator."""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from indic_transliteration import sanscript

from src.transliteration.script_detector import INDIC_SCRIPT_RANGES

SCRIPT_TO_SANSCRIPT_SCHEME = {
    "Devanagari": sanscript.DEVANAGARI,
    "Bengali": sanscript.BENGALI,
    "Gurmukhi": sanscript.GURMUKHI,
    "Gujarati": sanscript.GUJARATI,
    "Odia": sanscript.ORIYA,
    "Tamil": sanscript.TAMIL,
    "Telugu": sanscript.TELUGU,
    "Kannada": sanscript.KANNADA,
    "Malayalam": sanscript.MALAYALAM,
}

# Post-processing fallback table for characters that sanscript may leave untransliterated.
# Covers: Devanagari Candra vowels, Malayalam Chillu letters, Odia/Gurmukhi edge cases.
INDIC_FALLBACK_CHARS = {
    # -- Nukta marks (modifier) --
    "\u0a3c": "",      # Gurmukhi nukta
    "\u093c": "",      # Devanagari nukta
    "\u09bc": "",      # Bengali nukta
    "\u0abc": "",      # Gujarati nukta
    "\u0b3c": "",      # Odia nukta

    # -- Devanagari Candra/extended vowels (not in classical Sanskrit IAST) --
    "\u0911": "o",     # DEVANAGARI LETTER SHORT A (ऑ) — short rounded, closest to 'o'
    "\u0949": "o",     # DEVANAGARI VOWEL SIGN CANDRA O (ॉ) — short-O matra
    "\u090d": "e",     # DEVANAGARI LETTER SHORT E (ऍ) — short-E
    "\u0945": "e",     # DEVANAGARI VOWEL SIGN SHORT E (ॅ)
    "\u0972": "a",     # DEVANAGARI LETTER CANDRA A (ॲ)

    # -- Malayalam Chillu letters (U+0D7A–0D7F, outside base block handled by sanscript) --
    "\u0d7a": "nn",    # MALAYALAM LETTER CHILLU NN (ൺ)
    "\u0d7b": "n",     # MALAYALAM LETTER CHILLU N (ൻ)
    "\u0d7c": "r",     # MALAYALAM LETTER CHILLU RR (ർ)
    "\u0d7d": "l",     # MALAYALAM LETTER CHILLU L (ൽ)
    "\u0d7e": "ll",    # MALAYALAM LETTER CHILLU LL (ൾ)
    "\u0d7f": "k",     # MALAYALAM LETTER CHILLU K (ൿ)

    # -- Tamil special letters --
    "\u0ba9": "n",     # Tamil NNNA
    "\u0bb1": "r",     # Tamil RRA
    "\u0bb4": "zh",    # Tamil LLLA
    "\u0b9f": "t",     # Tamil TTA
    "\u0bf0": "10",    # Tamil number ten
    "\u0bf1": "100",   # Tamil number hundred
    "\u0bf2": "1000",  # Tamil number thousand

    # -- Odia Wa letter (U+0B71) not in base script table --
    "\u0b71": "w",     # ORIYA LETTER WA (ୱ)
}


def _get_indic_script(char: str) -> Optional[str]:
    """Check if character belongs to any supported Indic script block."""
    cp = ord(char)
    for name, start, end in INDIC_SCRIPT_RANGES:
        if start <= cp <= end:
            return name
    return None


class BaseTransliterator(ABC):
    """Abstract base interface for offline text transliteration."""

    @abstractmethod
    def transliterate(self, text: Optional[str], script: Optional[str] = None) -> Optional[str]:
        """Transliterate input text into Latin characters.
        
        Args:
            text: Input string.
            script: Optional pre-detected script name.
            
        Returns:
            Transliterated string, or original text if Latin/empty, or None if input was None.
        """
        pass


class IndicTransliterator(BaseTransliterator):
    """Offline Indic-to-Latin transliterator powered by sanscript.
    
    Guarantees:
    - Zero external network or cloud API calls (100% offline).
    - Latin text passes through completely unchanged.
    - Numbers, punctuation, whitespace, and formatting are preserved exactly.
    - Mixed-script strings are handled segment-by-segment.
    - Original input text is never modified in place.
    - Does not mix transliteration with normalization (casing, punctuation, etc. are untouched).
    """

    def __init__(self, target_scheme: str = sanscript.IAST):
        """
        Args:
            target_scheme: Target romanization scheme (sanscript.IAST, sanscript.ITRANS, sanscript.ISO).
                           Defaults to IAST (complete phoneme coverage across all Indic scripts).
        """
        self.target_scheme = target_scheme

    def _transliterate_segment(self, segment: str, script: str) -> str:
        """Transliterate a single contiguous Indic segment."""
        scheme = SCRIPT_TO_SANSCRIPT_SCHEME.get(script)
        if not scheme:
            return segment

        transliterated = sanscript.transliterate(segment, scheme, self.target_scheme)

        # Apply fallback table for any unmapped nuktas or script edge characters
        for char, replacement in INDIC_FALLBACK_CHARS.items():
            if char in transliterated:
                transliterated = transliterated.replace(char, replacement)

        return transliterated

    def transliterate(self, text: Optional[str], script: Optional[str] = None) -> Optional[str]:
        """Transliterate non-Latin Indic text to Latin while keeping Latin unchanged."""
        if text is None:
            return None
        if not text or not isinstance(text, str):
            return text

        # Quick check: if no Indic characters are present, return text unchanged
        has_indic = False
        for char in text:
            if _get_indic_script(char):
                has_indic = True
                break

        if not has_indic:
            return text

        # Segment-by-segment transliteration for contiguous Indic vs non-Indic sequences
        segments: List[Tuple[Optional[str], str]] = []
        current_chars: List[str] = []
        current_script: Optional[str] = None

        for char in text:
            char_script = _get_indic_script(char)
            if char_script == current_script:
                current_chars.append(char)
            else:
                if current_chars:
                    segments.append((current_script, "".join(current_chars)))
                current_script = char_script
                current_chars = [char]

        if current_chars:
            segments.append((current_script, "".join(current_chars)))

        # Reconstruct string preserving non-Indic tokens exactly
        result_parts: List[str] = []
        for seg_script, segment_text in segments:
            if seg_script and seg_script in SCRIPT_TO_SANSCRIPT_SCHEME:
                result_parts.append(self._transliterate_segment(segment_text, seg_script))
            else:
                result_parts.append(segment_text)

        return "".join(result_parts)


# Default transliterator singleton
_default_transliterator = IndicTransliterator()


def transliterate_text(text: Optional[str], script: Optional[str] = None) -> Optional[str]:
    """Convenience helper to transliterate text using default IndicTransliterator."""
    return _default_transliterator.transliterate(text, script=script)
