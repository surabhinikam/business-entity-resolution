"""
Blocking key generation functions for Business Entity Resolution.

Each key function takes a record dict and returns a set of blocking keys.
All keys are prefixed with the normalized country.

Keys:
  A: country + exact normalized business name
  B: country + exact transliterated business name
  C: country + first two meaningful name tokens (after stopword filtering)
  D: country + first meaningful name token + first extracted address number
  E: country + first meaningful name token (ONLY when no address number extracted)
  F: country + exact normalized business address
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.candidate_generation.name_tokens import extract_meaningful_tokens
from src.candidate_generation.address_parser import extract_address_number


# Human-readable names for each blocking key
BLOCKING_KEY_NAMES: Dict[str, str] = {
    "A": "Country + Exact Normalized Name",
    "B": "Country + Exact Transliterated Name",
    "C": "Country + First Two Meaningful Tokens",
    "D": "Country + First Meaningful Token + Address Number",
    "E": "Country + First Meaningful Token (no addr num fallback)",
    "F": "Country + Exact Normalized Address",
}

# Separator used to join key components — must not appear in normalized data
_SEP = "||"


def _get_country(record: Dict[str, Any]) -> Optional[str]:
    """Extract normalized country from a record."""
    country = record.get("country_normalized") or ""
    country = country.strip().lower()
    return country if country else None


def key_a_exact_norm_name(record: Dict[str, Any]) -> Set[str]:
    """
    Key A: country + exact normalized business name.
    High precision, catches exact matches after normalization.
    """
    country = _get_country(record)
    if not country:
        return set()

    norm_name = record.get("business_name_normalized") or ""
    norm_name = norm_name.strip()
    if not norm_name:
        return set()

    return {f"A{_SEP}{country}{_SEP}{norm_name}"}


def key_b_exact_translit_name(record: Dict[str, Any]) -> Set[str]:
    """
    Key B: country + exact transliterated business name.
    Catches matches where transliteration produces the same string
    even if normalization diverges.
    """
    country = _get_country(record)
    if not country:
        return set()

    translit_name = record.get("business_name_transliterated") or ""
    translit_name = translit_name.strip().lower()
    if not translit_name:
        return set()

    return {f"B{_SEP}{country}{_SEP}{translit_name}"}


def key_c_first_two_meaningful_tokens(record: Dict[str, Any]) -> Set[str]:
    """
    Key C: country + first two meaningful name tokens.
    Uses stopword-filtered tokens to avoid generic blocks.
    Requires at least 2 meaningful tokens.
    """
    country = _get_country(record)
    if not country:
        return set()

    norm_name = record.get("business_name_normalized") or ""
    tokens = extract_meaningful_tokens(norm_name)

    if len(tokens) < 2:
        return set()

    key_value = f"{tokens[0]} {tokens[1]}"
    return {f"C{_SEP}{country}{_SEP}{key_value}"}


def key_d_first_token_addr_num(record: Dict[str, Any]) -> Set[str]:
    """
    Key D: country + first meaningful name token + first address number.
    Only emitted when an address number can be extracted.
    """
    country = _get_country(record)
    if not country:
        return set()

    norm_name = record.get("business_name_normalized") or ""
    tokens = extract_meaningful_tokens(norm_name)
    if not tokens:
        return set()

    norm_addr = record.get("business_address_normalized") or ""
    addr_num, has_num = extract_address_number(norm_addr)
    if not has_num:
        return set()

    return {f"D{_SEP}{country}{_SEP}{tokens[0]}{_SEP}{addr_num}"}


def key_e_first_token_fallback(record: Dict[str, Any]) -> Set[str]:
    """
    Key E: country + first meaningful name token.
    FALLBACK ONLY: emitted only when NO address number was
    successfully extracted (regardless of whether address is null or not).

    This means:
    - Address is null -> Key E emitted (no number possible)
    - Address is "Suite B, Oak Plaza" -> Key E emitted (no number extractable)
    - Address is "108 Main St" -> Key E NOT emitted (number 108 extracted)
    """
    country = _get_country(record)
    if not country:
        return set()

    norm_name = record.get("business_name_normalized") or ""
    tokens = extract_meaningful_tokens(norm_name)
    if not tokens:
        return set()

    norm_addr = record.get("business_address_normalized") or ""
    _, has_num = extract_address_number(norm_addr)
    if has_num:
        # Address number exists -> use Key D instead, not this fallback
        return set()

    return {f"E{_SEP}{country}{_SEP}{tokens[0]}"}


def key_f_exact_norm_address(record: Dict[str, Any]) -> Set[str]:
    """
    Key F: country + exact normalized business address.
    High precision for records sharing the exact same normalized address.
    """
    country = _get_country(record)
    if not country:
        return set()

    norm_addr = record.get("business_address_normalized") or ""
    norm_addr = norm_addr.strip()
    if not norm_addr:
        return set()

    return {f"F{_SEP}{country}{_SEP}{norm_addr}"}


# Mapping from key label to key function
_KEY_FUNCTIONS = {
    "A": key_a_exact_norm_name,
    "B": key_b_exact_translit_name,
    "C": key_c_first_two_meaningful_tokens,
    "D": key_d_first_token_addr_num,
    "E": key_e_first_token_fallback,
    "F": key_f_exact_norm_address,
}


def generate_all_keys(
    record: Dict[str, Any],
    active_keys: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Generate all active blocking keys for a single record.

    Parameters:
        record: Dictionary with normalized fields.
        active_keys: List of key labels to generate (default: all A-F).

    Returns:
        Dict mapping key label -> set of blocking key strings.
        Also includes "ALL" -> union of all keys.
    """
    if active_keys is None:
        active_keys = list(_KEY_FUNCTIONS.keys())

    result: Dict[str, Set[str]] = {}
    all_keys: Set[str] = set()

    for key_label in active_keys:
        func = _KEY_FUNCTIONS.get(key_label)
        if func is None:
            continue
        keys = func(record)
        result[key_label] = keys
        all_keys |= keys

    result["ALL"] = all_keys
    return result
