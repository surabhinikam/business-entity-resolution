"""
General text normalization utilities for the business entity resolution pipeline.

Handles Unicode NFC normalization, IAST / diacritic stripping,
case folding, whitespace collapsing, and punctuation normalization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# Regex to match non-alphanumeric, non-whitespace characters except preserved ones
# Preserved punctuation in general text: hyphen/dash, slash, hash/number, ampersand, apostrophe/single quote, period
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_EXCEPT_SAFE = re.compile(r"[^\w\s\-/&#']", re.UNICODE)


def strip_diacritics(text: str) -> str:
    """
    Decompose Unicode characters and strip combining diacritical marks.
    E.g., IAST 'prāiveṭa limiṭeḍa' -> 'praiveta limiteda'.
    """
    # Decompose into base characters and combining marks (NFD)
    nfd_form = unicodedata.normalize("NFD", text)
    # Filter out combining diacritical characters
    stripped = "".join(ch for ch in nfd_form if unicodedata.category(ch) != "Mn")
    # Return composed NFC form
    return unicodedata.normalize("NFC", stripped)


def normalize_text(
    text: Optional[str],
    strip_punctuation: bool = False,
    preserve_safe_symbols: bool = True,
) -> Optional[str]:
    """
    Normalize arbitrary text:
    1. Handle None / empty strings safely.
    2. Normalize Unicode to NFC.
    3. Strip diacritical marks (IAST transliterations, accents, etc.).
    4. Convert to lowercase.
    5. Handle punctuation according to parameters:
       - If strip_punctuation is True and preserve_safe_symbols is True:
         preserves alphanumeric, whitespace, '-', '/', '&', '#', and converts other punctuation to spaces.
       - If strip_punctuation is True and preserve_safe_symbols is False:
         removes all punctuation, converting to spaces.
       - If strip_punctuation is False:
         leaves punctuation in place.
    6. Collapse redundant whitespace and trim ends.

    Returns None if text is None, or empty string if input is blank.
    """
    if text is None:
        return None

    cleaned = text.strip()
    if not cleaned:
        return ""

    # Unicode NFC normalization
    cleaned = unicodedata.normalize("NFC", cleaned)

    # Strip diacritical marks
    cleaned = strip_diacritics(cleaned)

    # Lowercase
    cleaned = cleaned.lower()

    if strip_punctuation:
        if preserve_safe_symbols:
            cleaned = _PUNCT_EXCEPT_SAFE.sub(" ", cleaned)
        else:
            cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)

    # Collapse multiple whitespace
    cleaned = _MULTI_WHITESPACE_RE.sub(" ", cleaned).strip()

    return cleaned
