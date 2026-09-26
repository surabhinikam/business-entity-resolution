"""
Record-level representation layer for business entity resolution.

Provides lightweight, immutable, and deterministic precomputed representations
for business names to avoid redundant computation across pairwise candidate comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Any

from src.analysis.text_similarity import character_ngrams


@dataclass(frozen=True, slots=True)
class NameRepresentation:
    """
    Lightweight, immutable precomputed representation of a business name.

    Attributes:
        clean_name: Normalized, lowercase business name string without leading/trailing whitespace.
        token_set: Unique valid business name tokens.
        char_3grams: Character 3-grams generated using the existing character_ngrams implementation.
        char_length: Character length of clean_name.
        token_count: Number of valid tokens in the business name.
        acronym: Deterministic initialism derived from ordered valid tokens.
    """

    clean_name: str
    token_set: frozenset[str]
    char_3grams: frozenset[str]
    char_length: int
    token_count: int
    acronym: str


EMPTY_NAME_REPRESENTATION = NameRepresentation(
    clean_name="",
    token_set=frozenset(),
    char_3grams=frozenset(),
    char_length=0,
    token_count=0,
    acronym="",
)


_SENTINEL_EMPTY_STRINGS = frozenset({"", "none", "nan", "null"})


def build_name_representation(
    normalized_name: Optional[str],
    tokens: Optional[Sequence[Optional[Any]]] = None,
) -> NameRepresentation:
    """
    Build a deterministic, immutable NameRepresentation for a business record.

    Parameters:
        normalized_name: The canonical normalized business name string (e.g. from business_name_normalized).
        tokens: The pre-split token collection (e.g. from business_name_tokens).
                If None, tokens are safely derived from normalized_name.
                If an empty list/sequence is passed, it is respected.

    Returns:
        A frozen NameRepresentation instance. If the input name is missing or empty,
        returns EMPTY_NAME_REPRESENTATION without raising exceptions.
    """
    if normalized_name is None:
        return EMPTY_NAME_REPRESENTATION

    if not isinstance(normalized_name, str):
        # Handle non-string types such as float('nan') or unexpected objects
        s = str(normalized_name).strip()
    else:
        s = normalized_name.strip()

    if not s or s.lower() in _SENTINEL_EMPTY_STRINGS:
        return EMPTY_NAME_REPRESENTATION

    clean_name = s.lower()

    # Process tokens safely
    if tokens is not None:
        clean_tokens: list[str] = []
        for t in tokens:
            if t is None:
                continue
            if not isinstance(t, str):
                t_str = str(t).strip()
            else:
                t_str = t.strip()
            if not t_str or t_str.lower() in _SENTINEL_EMPTY_STRINGS:
                continue
            clean_tokens.append(t_str.lower())
    else:
        # Fallback to whitespace split of clean_name when tokens were not pre-extracted
        clean_tokens = [
            t for t in clean_name.split() if t and t.lower() not in _SENTINEL_EMPTY_STRINGS
        ]

    # Generate character 3-grams reusing existing text_similarity implementation
    char_3grams = frozenset(character_ngrams(clean_name, n=3))

    token_count = len(clean_tokens)
    token_set = frozenset(clean_tokens)

    # Derive acronym from first character of each valid ordered token
    acronym = "".join(t[0] for t in clean_tokens if t)

    return NameRepresentation(
        clean_name=clean_name,
        token_set=token_set,
        char_3grams=char_3grams,
        char_length=len(clean_name),
        token_count=token_count,
        acronym=acronym,
    )
