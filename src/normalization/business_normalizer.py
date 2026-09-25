"""
Business name normalization module.

Handles normalization of business names, including:
- Stripping diacritics / lowercasing / whitespace via text_normalizer
- Standardizing corporate/legal suffixes (e.g. 'private limited' -> 'pvt ltd', 'inc.' -> 'inc')
- Handling transliterated Indic suffix equivalents (e.g. 'praiveta limiteda' -> 'pvt ltd', 'elaelapi' -> 'llp')
- Tokenization of normalized business names
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.normalization.text_normalizer import normalize_text


# Ordered list of suffix replacement patterns (longest / most specific first)
# Each pattern matches at the end of the normalized business name.
# Target canonical suffixes:
# - 'pvt ltd'
# - 'ltd'
# - 'llc'
# - 'inc'
# - 'corp'
# - 'llp'
# - 'plc'
# - 'co'
_LEGAL_SUFFIX_RULES = [
    # Private Limited variants (English, abbreviated, and transliterated Indic forms)
    # e.g., 'private limited', 'pvt ltd', 'pvt. ltd.', 'praiveta limiteda', 'pra li'
    (
        re.compile(
            r"\b(?:private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|pvt|praiveta\s+limiteda|pra\s+li)\b\.?$",
            re.IGNORECASE,
        ),
        "pvt ltd",
    ),
    # Public limited / plc
    (
        re.compile(r"\b(?:public\s+limited|plc)\b\.?$", re.IGNORECASE),
        "plc",
    ),
    # Limited
    (
        re.compile(r"\b(?:limited|ltd|limiteda)\b\.?$", re.IGNORECASE),
        "ltd",
    ),
    # LLC
    (
        re.compile(r"\b(?:limited\s+liability\s+company|llc)\b\.?$", re.IGNORECASE),
        "llc",
    ),
    # LLP / transliterated 'elaelapi'
    (
        re.compile(r"\b(?:limited\s+liability\s+partnership|llp|elaelapi)\b\.?$", re.IGNORECASE),
        "llp",
    ),
    # Inc / Incorporated
    (
        re.compile(r"\b(?:incorporated|inc)\b\.?$", re.IGNORECASE),
        "inc",
    ),
    # Corporation / Corp
    (
        re.compile(r"\b(?:corporation|corp)\b\.?$", re.IGNORECASE),
        "corp",
    ),
    # Company / Co (avoid replacing if part of a word or alone)
    (
        re.compile(r"\b(?:company|co)\b\.?$", re.IGNORECASE),
        "co",
    ),
]


def canonicalize_legal_suffix(name: str) -> str:
    """
    Standardize the legal/corporate suffix at the end of a business name string.
    Expects name to already be lowercased and stripped of diacritics.
    """
    # Remove trailing commas, periods, or extra spaces before suffix check
    trimmed = re.sub(r"[,.\s]+$", "", name).strip()

    for pattern, canonical in _LEGAL_SUFFIX_RULES:
        # Check if matched at the end of string
        sub_name, count = pattern.subn(canonical, trimmed)
        if count > 0:
            # Re-clean multiple spaces
            return re.sub(r"\s+", " ", sub_name).strip()

    return trimmed


def normalize_business_name(
    name: Optional[str],
    transliterated: Optional[str] = None,
) -> Optional[str]:
    """
    Normalize business name using either the transliterated form (preferred if provided)
    or the original text.

    Steps:
    1. Select transliterated if available, else original.
    2. Lowercase, strip diacritics, NFC normalize, convert punctuation to spaces
       (preserving '&', '-', and '/').
    3. Standardize legal suffixes to canonical forms.
    4. Collapse spaces and trim.
    """
    input_text = transliterated if transliterated is not None else name
    if input_text is None:
        return None

    cleaned = input_text.strip()
    if not cleaned:
        return ""

    # First text normalization: strip diacritics, lowercase, remove punctuation except safe symbols
    norm = normalize_text(cleaned, strip_punctuation=True, preserve_safe_symbols=True)
    if not norm:
        return ""

    # Canonicalize legal suffix
    canonical = canonicalize_legal_suffix(norm)

    return canonical


def tokenize_business_name(normalized_name: Optional[str]) -> List[str]:
    """
    Tokenize a normalized business name into alphanumeric/meaningful tokens.
    Filters out empty strings.
    """
    if not normalized_name:
        return []

    # Split on whitespace and non-alphanumeric/non-safe characters
    tokens = [t for t in re.split(r"[^\w&/-]+", normalized_name, flags=re.UNICODE) if t]
    return tokens
