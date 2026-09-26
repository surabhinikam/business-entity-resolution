"""
Pairwise business-name feature extraction module.

Computes pure, deterministic similarity and compatibility features between
two precomputed NameRepresentation objects (and optional transliterated names)
without re-tokenizing or re-extracting character n-grams.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from src.features.record_representation import NameRepresentation
from src.analysis.text_similarity import (
    jaccard_similarity,
    overlap_coefficient,
    dice_coefficient,
)
from src.normalization.business_vocabulary import LEGAL_SUFFIX_RULES

# Canonical corporate/legal suffixes derived directly from normalization layer,
# preserving first-seen and compound-first ordering (e.g. 'pvt ltd' before 'ltd')
CANONICAL_LEGAL_SUFFIXES: Tuple[str, ...] = tuple(
    dict.fromkeys(canonical for _, canonical in LEGAL_SUFFIX_RULES)
)

_SENTINEL_EMPTY_STRINGS = frozenset({"", "none", "nan", "null"})


def extract_legal_suffix(clean_name: str) -> Optional[str]:
    """
    Extract canonical legal suffix from the end of a normalized business name.

    Requires the suffix to be at a token boundary (preceded by a space or
    spanning the entire name). Returns None if no suffix is detected.
    """
    if not clean_name:
        return None

    for suffix in CANONICAL_LEGAL_SUFFIXES:
        if clean_name.endswith(suffix):
            suffix_len = len(suffix)
            if len(clean_name) == suffix_len or clean_name[-(suffix_len + 1)] == " ":
                return suffix
    return None


def compute_name_features(
    rep_a: NameRepresentation,
    rep_b: NameRepresentation,
    translit_a: Optional[str] = None,
    translit_b: Optional[str] = None,
) -> Dict[str, float | int]:
    """
    Compute pairwise business-name features between two NameRepresentation objects.

    Parameters:
        rep_a: Precomputed NameRepresentation for first entity (e.g. Source 1).
        rep_b: Precomputed NameRepresentation for second entity (e.g. Candidate Source 2/3).
        translit_a: Optional transliterated string for entity A.
        translit_b: Optional transliterated string for entity B.

    Returns:
        Dictionary mapping feature names to numeric values (int or float).
        Missing or empty values safely produce 0 or 0.0 (no artificial matches).
    """
    # 1. Exact normalized name match
    name_exact_norm = int(
        bool(rep_a.clean_name) and rep_a.clean_name == rep_b.clean_name
    )

    # 2. Exact transliterated name match (None == None must NOT count as a match)
    if translit_a is not None and translit_b is not None:
        t_a = str(translit_a).strip().lower()
        t_b = str(translit_b).strip().lower()
        if (
            t_a
            and t_b
            and t_a not in _SENTINEL_EMPTY_STRINGS
            and t_b not in _SENTINEL_EMPTY_STRINGS
        ):
            name_exact_translit = int(t_a == t_b)
        else:
            name_exact_translit = 0
    else:
        name_exact_translit = 0

    # 3-5. Token set similarities (Empty sets must produce 0.0, not 1.0)
    if rep_a.token_set and rep_b.token_set:
        name_token_jaccard = float(jaccard_similarity(rep_a.token_set, rep_b.token_set))
        name_token_overlap = float(overlap_coefficient(rep_a.token_set, rep_b.token_set))
        name_token_dice = float(dice_coefficient(rep_a.token_set, rep_b.token_set))
    else:
        name_token_jaccard = 0.0
        name_token_overlap = 0.0
        name_token_dice = 0.0

    # 6. Token count difference
    name_token_count_diff = int(abs(rep_a.token_count - rep_b.token_count))

    # 7. Character 3-gram Jaccard similarity (Empty sets must produce 0.0, not 1.0)
    if rep_a.char_3grams and rep_b.char_3grams:
        name_char_3gram_jaccard = float(
            jaccard_similarity(rep_a.char_3grams, rep_b.char_3grams)
        )
    else:
        name_char_3gram_jaccard = 0.0

    # 8. Character length difference
    name_char_len_diff = int(abs(rep_a.char_length - rep_b.char_length))

    # 9. Character length ratio (0.0 if either name is empty)
    if rep_a.char_length > 0 and rep_b.char_length > 0:
        name_char_len_ratio = float(
            min(rep_a.char_length, rep_b.char_length)
            / max(rep_a.char_length, rep_b.char_length)
        )
    else:
        name_char_len_ratio = 0.0

    # 10. First token exact match (0 if either token count is 0)
    if (
        rep_a.token_count > 0
        and rep_b.token_count > 0
        and rep_a.clean_name
        and rep_b.clean_name
    ):
        first_a = rep_a.clean_name.split()[0]
        first_b = rep_b.clean_name.split()[0]
        name_first_token_exact = int(first_a == first_b)
    else:
        name_first_token_exact = 0

    # 11. Acronym match (0 if either acronym is empty)
    if rep_a.acronym and rep_b.acronym:
        name_acronym_match = int(rep_a.acronym == rep_b.acronym)
    else:
        name_acronym_match = 0

    # 12. Legal suffix match (0 if either has no detectable suffix)
    suffix_a = extract_legal_suffix(rep_a.clean_name)
    suffix_b = extract_legal_suffix(rep_b.clean_name)
    if suffix_a is not None and suffix_b is not None:
        name_legal_suffix_match = int(suffix_a == suffix_b)
    else:
        name_legal_suffix_match = 0

    return {
        "name_exact_norm": name_exact_norm,
        "name_exact_translit": name_exact_translit,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_token_dice": name_token_dice,
        "name_token_count_diff": name_token_count_diff,
        "name_char_3gram_jaccard": name_char_3gram_jaccard,
        "name_char_len_diff": name_char_len_diff,
        "name_char_len_ratio": name_char_len_ratio,
        "name_first_token_exact": name_first_token_exact,
        "name_acronym_match": name_acronym_match,
        "name_legal_suffix_match": name_legal_suffix_match,
    }
