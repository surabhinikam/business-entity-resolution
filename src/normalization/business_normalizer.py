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
from src.normalization.business_vocabulary import (
    apply_legal_suffix,
    normalize_business_vocabulary,
    LEGAL_SUFFIX_RULES,
    BUSINESS_TERMS,
)


def canonicalize_legal_suffix(name: str) -> str:
    """
    Standardize the legal/corporate suffix at the end of a business name string.
    Expects name to already be lowercased and stripped of diacritics.
    """
    return apply_legal_suffix(name)


def normalize_business_name(
    name: Optional[str],
    transliterated: Optional[str] = None,
) -> Optional[str]:
    """
    Normalize business name using either the transliterated form (preferred if provided)
    or the original text.

    Pipeline:
    1. Select transliterated if available, else original.
    2. Text normalization: Lowercase, strip diacritics, NFC normalize, convert punctuation to spaces
       (preserving '&', '-', and '/').
    3. Business vocabulary normalization: Map high-confidence transliterated business tokens.
    4. Legal suffix normalization: Standardize end-of-name corporate suffixes to canonical forms.
    5. Collapse spaces and trim.
    """
    input_text = transliterated if transliterated is not None else name
    if input_text is None:
        return None

    cleaned = input_text.strip()
    if not cleaned:
        return ""

    # 1. Text normalization
    norm = normalize_text(cleaned, strip_punctuation=True, preserve_safe_symbols=True)
    if not norm:
        return ""

    # 2. Business vocabulary normalization (exact token matching)
    vocab_norm = normalize_business_vocabulary(norm)

    # 3. Legal suffix canonicalization (boundary-aware, end of name only)
    canonical = apply_legal_suffix(vocab_norm)

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
