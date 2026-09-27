"""
Record-level representation layer for business entity resolution.

Provides lightweight, immutable, and deterministic precomputed representations
for business names and addresses to avoid redundant computation across pairwise candidate comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence, Optional, Any, Set, Tuple

from src.analysis.text_similarity import character_ngrams, extract_numeric_tokens
from src.candidate_generation.address_parser import extract_address_number


# =============================================================================
# 1. Business Name Representation (Person 1)
# =============================================================================

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


# =============================================================================
# 2. Business Address Representation (Person 2)
# =============================================================================

_PIN_RE_IN = re.compile(r"\b([1-9]\d{5})\b")
_ZIP_RE_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def extract_postal_code(
    normalized_address: Optional[str],
    country: Optional[str] = None,
) -> Optional[str]:
    """
    Deterministic postal-code extraction helper.

    Formats handled:
    - US ZIP: 5 digits (skips leading house numbers in multi-token addresses)
    - Indian PIN: 6 digits (range 100000-999999, skips leading house numbers)

    Parameters:
        normalized_address: Clean normalized address string.
        country: Optional country string (e.g. 'united states', 'india').

    Returns:
        Extracted postal/PIN code string, or None if unextractable or missing.
    """
    if not normalized_address:
        return None

    addr = str(normalized_address).strip().lower()
    if not addr or addr in _SENTINEL_EMPTY_STRINGS:
        return None

    tokens = addr.split()
    if not tokens:
        return None

    c = country.strip().lower() if country else None

    # 1. India specific: 6 digits starting with 1-9
    if c == "india":
        matches = _PIN_RE_IN.findall(addr)
        if not matches:
            return None
        for m in reversed(matches):
            if len(tokens) > 1 and tokens[0] == m:
                continue
            return m
        return matches[-1] if len(tokens) == 1 else None

    # 2. US specific: 5 digits
    if c in ("united states", "us"):
        matches = _ZIP_RE_US.findall(addr)
        if not matches:
            return None
        for m in reversed(matches):
            if len(tokens) > 1 and tokens[0] == m:
                continue
            return m
        return matches[-1] if len(tokens) == 1 else None

    # 3. Open-set / Unknown country fallback
    matches_6 = _PIN_RE_IN.findall(addr)
    for m in reversed(matches_6):
        if len(tokens) > 1 and tokens[0] == m:
            continue
        return m

    matches_5 = _ZIP_RE_US.findall(addr)
    for m in reversed(matches_5):
        if len(tokens) > 1 and tokens[0] == m:
            continue
        return m

    if len(tokens) == 1:
        if matches_6:
            return matches_6[-1]
        if matches_5:
            return matches_5[-1]

    return None


@dataclass(frozen=True, slots=True)
class AddressRepresentation:
    """
    Lightweight, immutable precomputed representation of a business address.

    Attributes:
        clean_address: Normalized address string.
        token_set: Unique valid address tokens.
        char_3grams: Character 3-grams generated from clean_address.
        char_length: Character length of clean_address.
        token_count: Number of valid tokens.
        numeric_tokens: Set of numeric / unit tokens extracted for shared number counts.
        primary_number: Primary street/house number extracted via address_parser.
        postal_code: Extracted 5-digit US ZIP or 6-digit Indian PIN.
        is_missing: Boolean flag indicating if the address is missing/empty.
        ordered_tokens: Preserved ordered sequence of tokens.
        entity_id: Optional entity identifier for linkage.
        source: Optional source namespace (e.g. 'source1', 'source2', 'source3').
        country_normalized: Optional normalized country.
    """

    clean_address: str
    token_set: frozenset[str]
    char_3grams: frozenset[str]
    char_length: int
    token_count: int
    numeric_tokens: frozenset[str]
    primary_number: Optional[str]
    postal_code: Optional[str]
    is_missing: bool
    ordered_tokens: tuple[str, ...] = field(default_factory=tuple)
    entity_id: str = ""
    source: str = ""
    country_normalized: Optional[str] = None

    # Backwards-compatibility properties with AddressRecordRepresentation
    @property
    def is_address_missing(self) -> bool:
        return self.is_missing

    @property
    def normalized_address(self) -> Optional[str]:
        return self.clean_address if not self.is_missing else None

    @property
    def address_tokens(self) -> list[str]:
        return list(self.ordered_tokens) if self.ordered_tokens else list(self.token_set)

    @property
    def primary_address_number(self) -> Optional[str]:
        return self.primary_number

    @property
    def all_numeric_tokens(self) -> set[str]:
        return set(self.numeric_tokens)


EMPTY_ADDRESS_REPRESENTATION = AddressRepresentation(
    clean_address="",
    token_set=frozenset(),
    char_3grams=frozenset(),
    char_length=0,
    token_count=0,
    numeric_tokens=frozenset(),
    primary_number=None,
    postal_code=None,
    is_missing=True,
    ordered_tokens=(),
    entity_id="",
    source="",
    country_normalized=None,
)


def build_address_representation(
    normalized_address: Optional[str] | dict[str, Any] = None,
    tokens: Optional[Sequence[Optional[Any]]] = None,
    country: Optional[str] = None,
    entity_id: str = "",
    source: str = "",
) -> AddressRepresentation:
    """
    Build a deterministic, immutable AddressRepresentation for a business record.

    Accepts either:
    1. A dictionary record (e.g. from processed parquet with 'business_address_normalized')
    2. Explicit arguments: normalized_address, tokens, country, entity_id, source

    Reuses existing precomputed fields (business_address_normalized, business_address_tokens)
    and extracts numeric tokens and primary address numbers once per entity.
    """
    if isinstance(normalized_address, dict):
        rec = normalized_address
        normalized_address = rec.get("clean_address", rec.get("normalized_address", rec.get("business_address_normalized")))
        if tokens is None:
            tokens = rec.get("token_set", rec.get("address_tokens", rec.get("business_address_tokens")))
        if country is None:
            country = rec.get("country_normalized")
        if not entity_id:
            entity_id = rec.get("entity_id", "")
        if not source:
            source = rec.get("source", "")

    _missing_kwargs: dict[str, Any] = dict(
        clean_address="",
        token_set=frozenset(),
        char_3grams=frozenset(),
        char_length=0,
        token_count=0,
        numeric_tokens=frozenset(),
        primary_number=None,
        postal_code=None,
        is_missing=True,
        ordered_tokens=(),
        entity_id=entity_id,
        source=source,
        country_normalized=country,
    )

    if normalized_address is None:
        return AddressRepresentation(**_missing_kwargs)

    s = str(normalized_address).strip()
    if not s or s.lower() in _SENTINEL_EMPTY_STRINGS:
        return AddressRepresentation(**_missing_kwargs)

    clean_address = s.lower()

    if tokens is not None:
        clean_tokens: list[str] = []
        for t in tokens:
            if t is None:
                continue
            t_str = str(t).strip()
            if not t_str or t_str.lower() in _SENTINEL_EMPTY_STRINGS:
                continue
            clean_tokens.append(t_str.lower())
    else:
        clean_tokens = [
            t for t in clean_address.split() if t and t.lower() not in _SENTINEL_EMPTY_STRINGS
        ]

    token_set = frozenset(clean_tokens)
    ordered_tokens = tuple(clean_tokens)
    char_3grams = frozenset(character_ngrams(clean_address, n=3))
    numeric_tokens = frozenset(extract_numeric_tokens(clean_address))

    num, has_num = extract_address_number(clean_address)
    primary_number = num if has_num else None

    postal_code = extract_postal_code(clean_address, country=country)

    return AddressRepresentation(
        clean_address=clean_address,
        token_set=token_set,
        char_3grams=char_3grams,
        char_length=len(clean_address),
        token_count=len(clean_tokens),
        numeric_tokens=numeric_tokens,
        primary_number=primary_number,
        postal_code=postal_code,
        is_missing=False,
        ordered_tokens=ordered_tokens,
        entity_id=entity_id,
        source=source,
        country_normalized=country,
    )
