"""
Local text similarity and comparison utilities for business entity resolution.
Strictly offline, using no external APIs or services.
"""

from __future__ import annotations

import re
from typing import List, Set, Optional, Dict, Any, Tuple


# Regex for extracting numeric sequences from address or text (e.g., 570/13, 105, 402, 12-a, 12a)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:[/-][a-zA-Z0-9]+)?\b|\b\d+[a-zA-Z]?\b")


def extract_numeric_tokens(text: Optional[str]) -> Set[str]:
    """Extract numeric sequences from text (house numbers, plot numbers, pincodes)."""
    if not text:
        return set()
    return set(_NUMERIC_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(tokens_a: List[str] | Set[str], tokens_b: List[str] | Set[str]) -> float:
    """Calculate Jaccard similarity between two token collections."""
    set_a = set(tokens_a) if isinstance(tokens_a, list) else tokens_a
    set_b = set(tokens_b) if isinstance(tokens_b, list) else tokens_b
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def overlap_coefficient(tokens_a: List[str] | Set[str], tokens_b: List[str] | Set[str]) -> float:
    """
    Calculate Szymkiewicz-Simpson Overlap coefficient (|A n B| / min(|A|, |B|)).
    High score indicates one set is a subset of another (e.g. truncated address).
    """
    set_a = set(tokens_a) if isinstance(tokens_a, list) else tokens_a
    set_b = set(tokens_b) if isinstance(tokens_b, list) else tokens_b
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    min_len = min(len(set_a), len(set_b))
    return len(set_a & set_b) / min_len if min_len > 0 else 0.0


def dice_coefficient(tokens_a: List[str] | Set[str], tokens_b: List[str] | Set[str]) -> float:
    """Calculate Sørensen-Dice coefficient (2 * |A n B| / (|A| + |B|))."""
    set_a = set(tokens_a) if isinstance(tokens_a, list) else tokens_a
    set_b = set(tokens_b) if isinstance(tokens_b, list) else tokens_b
    total = len(set_a) + len(set_b)
    if total == 0:
        return 1.0
    return (2.0 * len(set_a & set_b)) / total


def character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from a string."""
    if not text:
        return set()
    cleaned = f" {text.lower().strip()} "
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def character_ngram_jaccard(str_a: Optional[str], str_b: Optional[str], n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    if str_a is None and str_b is None:
        return 1.0
    if str_a is None or str_b is None:
        return 0.0
    grams_a = character_ngrams(str_a, n=n)
    grams_b = character_ngrams(str_b, n=n)
    return jaccard_similarity(grams_a, grams_b)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Use two-row dynamic programming
    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = list(v1)

    return v0[len(s2)]


def levenshtein_similarity(str_a: Optional[str], str_b: Optional[str]) -> float:
    """
    Normalized Levenshtein similarity: 1 - (dist / max(len(a), len(b))).
    Range [0.0, 1.0].
    """
    if str_a is None and str_b is None:
        return 1.0
    if str_a is None or str_b is None:
        return 0.0
    a = str_a.strip().lower()
    b = str_b.strip().lower()
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    dist = levenshtein_distance(a, b)
    return max(0.0, 1.0 - (dist / max_len))


def calculate_pair_features(
    s1_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract comprehensive similarity features between an S1 entity and a candidate (S2 or S3) entity.
    Expects dictionary containing both raw and processed fields.
    """
    # 1. Names
    raw_name_1 = s1_row.get("business_name") or ""
    raw_name_2 = candidate_row.get("business_name") or ""
    norm_name_1 = s1_row.get("business_name_normalized") or ""
    norm_name_2 = candidate_row.get("business_name_normalized") or ""
    translit_name_1 = s1_row.get("business_name_transliterated") or ""
    translit_name_2 = candidate_row.get("business_name_transliterated") or ""

    tokens_name_1 = s1_row.get("business_name_tokens") or []
    tokens_name_2 = candidate_row.get("business_name_tokens") or []
    if isinstance(tokens_name_1, str):
        tokens_name_1 = tokens_name_1.split()
    if isinstance(tokens_name_2, str):
        tokens_name_2 = tokens_name_2.split()

    name_exact_raw = int(raw_name_1.strip().lower() == raw_name_2.strip().lower() and bool(raw_name_1))
    name_exact_norm = int(norm_name_1.strip() == norm_name_2.strip() and bool(norm_name_1))
    name_exact_translit = int(
        translit_name_1.strip().lower() == translit_name_2.strip().lower() and bool(translit_name_1)
    )

    name_token_jaccard = jaccard_similarity(tokens_name_1, tokens_name_2)
    name_token_overlap = overlap_coefficient(tokens_name_1, tokens_name_2)
    name_char_ngram_jaccard = character_ngram_jaccard(norm_name_1 or raw_name_1, norm_name_2 or raw_name_2)
    name_len_diff = abs(len(raw_name_1) - len(raw_name_2))
    name_token_len_diff = abs(len(tokens_name_1) - len(tokens_name_2))

    # 2. Addresses
    raw_addr_1 = s1_row.get("business_address") or ""
    raw_addr_2 = candidate_row.get("business_address") or ""
    norm_addr_1 = s1_row.get("business_address_normalized") or ""
    norm_addr_2 = candidate_row.get("business_address_normalized") or ""
    translit_addr_1 = s1_row.get("business_address_transliterated") or ""
    translit_addr_2 = candidate_row.get("business_address_transliterated") or ""

    tokens_addr_1 = s1_row.get("business_address_tokens") or []
    tokens_addr_2 = candidate_row.get("business_address_tokens") or []
    if isinstance(tokens_addr_1, str):
        tokens_addr_1 = tokens_addr_1.split()
    if isinstance(tokens_addr_2, str):
        tokens_addr_2 = tokens_addr_2.split()

    addr_exact_raw = int(raw_addr_1.strip().lower() == raw_addr_2.strip().lower() and bool(raw_addr_1))
    addr_exact_norm = int(norm_addr_1.strip() == norm_addr_2.strip() and bool(norm_addr_1))
    addr_exact_translit = int(
        translit_addr_1.strip().lower() == translit_addr_2.strip().lower() and bool(translit_addr_1)
    )

    addr_token_jaccard = jaccard_similarity(tokens_addr_1, tokens_addr_2)
    addr_token_overlap = overlap_coefficient(tokens_addr_1, tokens_addr_2)
    addr_char_ngram_jaccard = character_ngram_jaccard(norm_addr_1 or raw_addr_1, norm_addr_2 or raw_addr_2)
    addr_len_diff = abs(len(raw_addr_1) - len(raw_addr_2))
    addr_token_len_diff = abs(len(tokens_addr_1) - len(tokens_addr_2))

    # Shared numeric tokens in addresses
    num_addr_1 = extract_numeric_tokens(norm_addr_1 or raw_addr_1)
    num_addr_2 = extract_numeric_tokens(norm_addr_2 or raw_addr_2)
    shared_numbers = num_addr_1 & num_addr_2
    shared_num_count = len(shared_numbers)
    shared_num_jaccard = jaccard_similarity(num_addr_1, num_addr_2)
    has_shared_num = int(shared_num_count > 0)

    # 3. Country
    raw_country_1 = (s1_row.get("country") or "").strip().lower()
    raw_country_2 = (candidate_row.get("country") or "").strip().lower()
    norm_country_1 = (s1_row.get("country_normalized") or "").strip().lower()
    norm_country_2 = (candidate_row.get("country_normalized") or "").strip().lower()

    country_match_raw = int(raw_country_1 == raw_country_2 and bool(raw_country_1))
    country_match_norm = int(norm_country_1 == norm_country_2 and bool(norm_country_1))

    return {
        "s1_entity_id": s1_row.get("entity_id"),
        "candidate_entity_id": candidate_row.get("entity_id"),
        "candidate_source": candidate_row.get("source"),
        # Name metrics
        "name_exact_raw": name_exact_raw,
        "name_exact_norm": name_exact_norm,
        "name_exact_translit": name_exact_translit,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_char_ngram_jaccard": name_char_ngram_jaccard,
        "name_len_diff": name_len_diff,
        "name_token_len_diff": name_token_len_diff,
        # Address metrics
        "addr_exact_raw": addr_exact_raw,
        "addr_exact_norm": addr_exact_norm,
        "addr_exact_translit": addr_exact_translit,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_token_overlap": addr_token_overlap,
        "addr_char_ngram_jaccard": addr_char_ngram_jaccard,
        "addr_len_diff": addr_len_diff,
        "addr_token_len_diff": addr_token_len_diff,
        "shared_num_count": shared_num_count,
        "shared_num_jaccard": shared_num_jaccard,
        "has_shared_num": has_shared_num,
        # Country metrics
        "country_match_raw": country_match_raw,
        "country_match_norm": country_match_norm,
    }
