"""
Address normalization module.

Handles normalization of business addresses:
- Stripping diacritics / lowercasing / whitespace via text_normalizer
- Preserving critical alphanumeric address structure:
  e.g. '570/13', '12-A', '#402', 'Plot No. 3829'
- Standardizing common standalone abbreviations (e.g. 'no.' -> 'no')
- Tokenization of normalized addresses without losing numeric segments
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.normalization.text_normalizer import normalize_text


# Regex patterns for address standardization
# Replace standalone 'no.' or 'no:' with 'no'
_NO_ABBREV_RE = re.compile(r"\bno[.:]\b", re.IGNORECASE)

# Pattern to clean up noise punctuation like commas, semicolons, brackets, but preserve '/', '-', '#', '&'
_ADDR_PUNCT_RE = re.compile(r"[^\w\s\-/&#]", re.UNICODE)


def normalize_address(
    address: Optional[str],
    transliterated: Optional[str] = None,
) -> Optional[str]:
    """
    Normalize business address:
    1. Select transliterated form if provided, else original address.
    2. Lowercase, strip diacritics (IAST marks), normalize Unicode NFC.
    3. Standardize standalone 'no.' / 'no:' to 'no'.
    4. Preserve '/', '-', '#', '&', and alphanumeric tokens.
    5. Clean redundant whitespace and boundary punctuation.

    Returns None if address is None, or empty string if input is blank.
    """
    input_text = transliterated if transliterated is not None else address
    if input_text is None:
        return None

    cleaned = input_text.strip()
    if not cleaned:
        return ""

    # Text normalization: strip diacritics, NFC, lowercase
    norm = normalize_text(cleaned, strip_punctuation=False)
    if not norm:
        return ""

    # Standardize 'no.' -> 'no'
    norm = _NO_ABBREV_RE.sub("no", norm)

    # Clean punctuation except safe address characters: alphanumeric, spaces, '-', '/', '&', '#'
    norm = _ADDR_PUNCT_RE.sub(" ", norm)

    # Collapse multiple whitespace
    norm = re.sub(r"\s+", " ", norm).strip()

    # Clean leading/trailing stray punctuation
    norm = norm.strip(" -/,&#")

    return norm


def tokenize_address(normalized_address: Optional[str]) -> List[str]:
    """
    Tokenize normalized address into meaningful tokens.
    Preserves tokens such as '570/13', '12-a', '#402', 'no', '3829'.
    """
    if not normalized_address:
        return []

    # Split by spaces and commas, keeping tokens that contain words or digits
    raw_tokens = normalized_address.split()
    tokens = []
    for token in raw_tokens:
        t = token.strip(" ,.;:")
        if t:
            tokens.append(t)
    return tokens
