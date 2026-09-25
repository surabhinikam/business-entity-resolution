"""
Country normalization module.

Handles standardizing country names and ISO codes to a canonical lowercase representation.
Supports India, United States, France, and gracefully falls back to cleaned lowercase
for any other country observed in train/test splits.
"""

from __future__ import annotations

import re
from typing import Optional

from src.normalization.text_normalizer import normalize_text


# Canonical mapping for countries observed in the dataset or standard aliases
# Keys are lowercase stripped strings (no punctuation)
_COUNTRY_MAP = {
    # India
    "in": "india",
    "ind": "india",
    "india": "india",
    "republic of india": "india",
    "bharat": "india",
    # United States
    "us": "united states",
    "usa": "united states",
    "u s a": "united states",
    "u s": "united states",
    "united states": "united states",
    "united states of america": "united states",
    # France
    "fr": "france",
    "fra": "france",
    "france": "france",
    "french republic": "france",
    "republique francaise": "france",
    # United Kingdom
    "uk": "united kingdom",
    "gbr": "united kingdom",
    "gb": "united kingdom",
    "united kingdom": "united kingdom",
    "great britain": "united kingdom",
    # Germany
    "de": "germany",
    "deu": "germany",
    "germany": "germany",
    "deutschland": "germany",
    # Canada
    "ca": "canada",
    "can": "canada",
    "canada": "canada",
}


def normalize_country(country: Optional[str]) -> Optional[str]:
    """
    Normalize country names/codes into canonical lowercase form.
    E.g.:
      'IN' -> 'india'
      'India' -> 'india'
      'U.S.A.' -> 'united states'
      'FRA' -> 'france'
      'France' -> 'france'

    Unknown countries are normalized (lowercased, stripped, diacritics removed)
    and returned directly without crashing or dropping information.
    """
    if country is None:
        return None

    cleaned = country.strip()
    if not cleaned:
        return ""

    # Strip diacritics and lowercase
    norm = normalize_text(cleaned, strip_punctuation=True, preserve_safe_symbols=False)
    if not norm:
        return ""

    # Check canonical dictionary
    if norm in _COUNTRY_MAP:
        return _COUNTRY_MAP[norm]

    # Return normalized unknown country
    return norm
