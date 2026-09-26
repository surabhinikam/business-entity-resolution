"""
Address feature extraction module for Person 2.

Computes pairwise address similarity, compatibility, and missingness features
between precomputed AddressRepresentation objects (and batch DataFrames)
without re-normalizing or re-tokenizing text.

Features covered:
1. address_exact: Exact equality of normalized address strings (1=match, 0=diff, -1=missing)
2. address_token_jaccard: Jaccard similarity of precomputed address tokens ([0.0, 1.0])
3. address_token_overlap: Overlap coefficient |A ∩ B| / min(|A|, |B|) ([0.0, 1.0])
4. address_char_3gram_similarity: Character 3-gram Jaccard on normalized addresses ([0.0, 1.0])
5. shared_address_number_count: Number of shared numeric tokens (>= 0)
6. address_number_overlap: Primary street/house number agreement (1=match, 0=diff, -1=missing/none)
7. postal_match: Postal/PIN code agreement (1=match, 0=diff, -1=missing/unextractable)
8. address_length_difference: Absolute length difference between normalized addresses (-1 if missing)
9. address_missing_s1: Missing indicator for Source 1 address (1=missing, 0=present)
10. address_missing_candidate: Missing indicator for Candidate address (1=missing, 0=present)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Sequence, Union
import polars as pl

from src.features.feature_schema import (
    ADDRESS_FEATURE_NAMES,
    PAIR_ID_COLUMNS,
)
from src.features.record_representation import (
    AddressRepresentation,
    EMPTY_ADDRESS_REPRESENTATION,
    build_address_representation,
    extract_postal_code,
)
from src.analysis.text_similarity import character_ngrams, extract_numeric_tokens
from src.candidate_generation.address_parser import extract_address_number

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Single-Pair Address Feature Extraction
# =============================================================================

def compute_address_features(
    rep_a: Any,
    rep_b: Any,
) -> Dict[str, Union[int, float]]:
    """
    Compute pairwise address features between two address representations.

    Parameters:
        rep_a: Precomputed AddressRepresentation (or AddressRecordRepresentation) for Source 1.
        rep_b: Precomputed AddressRepresentation (or AddressRecordRepresentation) for Candidate.

    Returns:
        Dict mapping all 10 ADDRESS_FEATURE_NAMES to numeric values matching schema types.
    """
    # 1. Missingness determination
    is_missing_a = getattr(rep_a, "is_missing", getattr(rep_a, "is_address_missing", False))
    is_missing_b = getattr(rep_b, "is_missing", getattr(rep_b, "is_address_missing", False))

    clean_a = getattr(rep_a, "clean_address", getattr(rep_a, "normalized_address", "")) or ""
    clean_b = getattr(rep_b, "clean_address", getattr(rep_b, "normalized_address", "")) or ""

    clean_a = str(clean_a).strip().lower()
    clean_b = str(clean_b).strip().lower()

    if not clean_a:
        is_missing_a = True
    if not clean_b:
        is_missing_b = True

    missing_s1 = 1 if is_missing_a else 0
    missing_cand = 1 if is_missing_b else 0

    # If either address is missing or empty, apply strict schema null semantics
    if is_missing_a or is_missing_b:
        return {
            "address_exact": -1,
            "address_token_jaccard": 0.0,
            "address_token_overlap": 0.0,
            "address_char_3gram_similarity": 0.0,
            "shared_address_number_count": 0,
            "address_number_overlap": -1,
            "postal_match": -1,
            "address_length_difference": -1,
            "address_missing_s1": missing_s1,
            "address_missing_candidate": missing_cand,
        }

    # 2. Both addresses are present
    # Exact address string equality
    address_exact = 1 if clean_a == clean_b else 0

    # Character length difference
    len_a = getattr(rep_a, "char_length", len(clean_a))
    len_b = getattr(rep_b, "char_length", len(clean_b))
    address_length_difference = int(abs(len_a - len_b))

    # Token sets & similarities
    tokens_a = getattr(rep_a, "token_set", None)
    tokens_b = getattr(rep_b, "token_set", None)
    if tokens_a is None:
        raw_tokens_a = getattr(rep_a, "address_tokens", None)
        tokens_a = frozenset(raw_tokens_a) if raw_tokens_a else frozenset(clean_a.split())
    if tokens_b is None:
        raw_tokens_b = getattr(rep_b, "address_tokens", None)
        tokens_b = frozenset(raw_tokens_b) if raw_tokens_b else frozenset(clean_b.split())

    if not tokens_a or not tokens_b:
        address_token_jaccard = 0.0
        address_token_overlap = 0.0
    else:
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        min_len = min(len(tokens_a), len(tokens_b))
        address_token_jaccard = float(inter / union) if union > 0 else 0.0
        address_token_overlap = float(inter / min_len) if min_len > 0 else 0.0

    # Character 3-gram similarity
    grams_a = getattr(rep_a, "char_3grams", None)
    grams_b = getattr(rep_b, "char_3grams", None)
    if grams_a is None:
        grams_a = frozenset(character_ngrams(clean_a, n=3))
    if grams_b is None:
        grams_b = frozenset(character_ngrams(clean_b, n=3))

    if not grams_a or not grams_b:
        address_char_3gram_similarity = 0.0
    else:
        inter3 = len(grams_a & grams_b)
        union3 = len(grams_a | grams_b)
        address_char_3gram_similarity = float(inter3 / union3) if union3 > 0 else 0.0

    # Numeric tokens count
    nums_a = getattr(rep_a, "numeric_tokens", getattr(rep_a, "all_numeric_tokens", None))
    nums_b = getattr(rep_b, "numeric_tokens", getattr(rep_b, "all_numeric_tokens", None))
    if nums_a is None:
        nums_a = frozenset(extract_numeric_tokens(clean_a))
    if nums_b is None:
        nums_b = frozenset(extract_numeric_tokens(clean_b))

    shared_address_number_count = int(len(set(nums_a) & set(nums_b)))

    # Primary address number overlap
    num_a = getattr(rep_a, "primary_number", getattr(rep_a, "primary_address_number", None))
    num_b = getattr(rep_b, "primary_number", getattr(rep_b, "primary_address_number", None))
    if num_a is None or num_b is None:
        address_number_overlap = -1
    elif str(num_a).strip().lower() == str(num_b).strip().lower():
        address_number_overlap = 1
    else:
        address_number_overlap = 0

    # Postal code match
    post_a = getattr(rep_a, "postal_code", None)
    post_b = getattr(rep_b, "postal_code", None)
    if post_a is None:
        post_a = extract_postal_code(clean_a, getattr(rep_a, "country_normalized", None))
    if post_b is None:
        post_b = extract_postal_code(clean_b, getattr(rep_b, "country_normalized", None))

    if post_a is None or post_b is None:
        postal_match = -1
    elif str(post_a).strip() == str(post_b).strip():
        postal_match = 1
    else:
        postal_match = 0

    return {
        "address_exact": address_exact,
        "address_token_jaccard": address_token_jaccard,
        "address_token_overlap": address_token_overlap,
        "address_char_3gram_similarity": address_char_3gram_similarity,
        "shared_address_number_count": shared_address_number_count,
        "address_number_overlap": address_number_overlap,
        "postal_match": postal_match,
        "address_length_difference": address_length_difference,
        "address_missing_s1": missing_s1,
        "address_missing_candidate": missing_cand,
    }


# Backwards compatibility alias for Part 1 scaffolding
compute_address_pair_features = compute_address_features


# =============================================================================
# 2. Batch Vectorized Address Feature Extraction
# =============================================================================

def _representations_to_lookup_df(
    reps: Union[Dict[str, Any], pl.DataFrame],
) -> pl.DataFrame:
    """Convert a dictionary of AddressRepresentation objects into a Polars lookup DataFrame."""
    if isinstance(reps, pl.DataFrame):
        return reps

    rows: List[Dict[str, Any]] = []
    for eid, r in reps.items():
        if isinstance(r, dict):
            clean_addr = r.get("clean_address", r.get("business_address_normalized", "")) or ""
            is_miss = r.get("is_missing", r.get("is_address_missing", not bool(clean_addr)))
            toks = r.get("token_set", r.get("business_address_tokens", []))
            if isinstance(toks, (set, frozenset)):
                toks = list(toks)
            elif toks is None:
                toks = []
            num_toks = r.get("numeric_tokens", r.get("all_numeric_tokens", []))
            if isinstance(num_toks, (set, frozenset)):
                num_toks = list(num_toks)
            grams = r.get("char_3grams", [])
            if isinstance(grams, (set, frozenset)):
                grams = list(grams)
            primary_num = r.get("primary_number", r.get("primary_address_number"))
            postal = r.get("postal_code")
            char_len = r.get("char_length", len(clean_addr))
        else:
            clean_addr = getattr(r, "clean_address", getattr(r, "normalized_address", "")) or ""
            is_miss = getattr(r, "is_missing", getattr(r, "is_address_missing", not bool(clean_addr)))
            tok_set = getattr(r, "token_set", None)
            if tok_set is not None:
                toks = list(tok_set)
            else:
                toks = list(getattr(r, "address_tokens", []) or clean_addr.split())
            num_toks = list(getattr(r, "numeric_tokens", getattr(r, "all_numeric_tokens", [])))
            grams = list(getattr(r, "char_3grams", []))
            primary_num = getattr(r, "primary_number", getattr(r, "primary_address_number", None))
            postal = getattr(r, "postal_code", None)
            char_len = getattr(r, "char_length", len(clean_addr))

        rows.append({
            "entity_id": str(eid),
            "clean_address": clean_addr,
            "char_length": int(char_len),
            "primary_number": str(primary_num) if primary_num is not None else None,
            "postal_code": str(postal) if postal is not None else None,
            "is_missing": bool(is_miss),
            "token_set": [str(t) for t in toks],
            "numeric_tokens": [str(n) for n in num_toks],
            "char_3grams": [str(g) for g in grams],
        })

    lookup_schema = {
        "entity_id": pl.Utf8,
        "clean_address": pl.Utf8,
        "char_length": pl.Int64,
        "primary_number": pl.Utf8,
        "postal_code": pl.Utf8,
        "is_missing": pl.Boolean,
        "token_set": pl.List(pl.Utf8),
        "numeric_tokens": pl.List(pl.Utf8),
        "char_3grams": pl.List(pl.Utf8),
    }

    if not rows:
        return pl.DataFrame({
            "entity_id": pl.Series([], dtype=pl.Utf8),
            "clean_address": pl.Series([], dtype=pl.Utf8),
            "char_length": pl.Series([], dtype=pl.Int64),
            "primary_number": pl.Series([], dtype=pl.Utf8),
            "postal_code": pl.Series([], dtype=pl.Utf8),
            "is_missing": pl.Series([], dtype=pl.Boolean),
            "token_set": pl.Series([], dtype=pl.List(pl.Utf8)),
            "numeric_tokens": pl.Series([], dtype=pl.List(pl.Utf8)),
            "char_3grams": pl.Series([], dtype=pl.List(pl.Utf8)),
        }, schema=lookup_schema)

    return pl.DataFrame(rows, schema=lookup_schema)


def extract_address_features_batch(
    candidate_pairs_df: pl.DataFrame,
    s1_representations: Union[Dict[str, Any], pl.DataFrame],
    cand_representations: Union[Dict[str, Any], pl.DataFrame],
) -> pl.DataFrame:
    """
    Vectorized batch calculation of all 10 address features.

    Avoids row-by-row Python iteration over millions of pairs by joining lookup
    tables and evaluating SIMD-accelerated Polars expressions.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        s1_representations: Mapping of S1 entity_id -> AddressRepresentation (or lookup DataFrame).
        cand_representations: Mapping of Candidate entity_id -> AddressRepresentation (or lookup DataFrame).

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + ADDRESS_FEATURE_NAMES.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    # Handle empty input
    if candidate_pairs_df.height == 0:
        schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
        schema.update({
            "address_exact": pl.Int8,
            "address_token_jaccard": pl.Float32,
            "address_token_overlap": pl.Float32,
            "address_char_3gram_similarity": pl.Float32,
            "shared_address_number_count": pl.Int16,
            "address_number_overlap": pl.Int8,
            "postal_match": pl.Int8,
            "address_length_difference": pl.Int16,
            "address_missing_s1": pl.Int8,
            "address_missing_candidate": pl.Int8,
        })
        return pl.DataFrame(schema=schema)

    # 1. Build lookup tables
    s1_lookup = _representations_to_lookup_df(s1_representations)
    cand_lookup = _representations_to_lookup_df(cand_representations)

    # 2. Join S1 and Candidate representations
    joined = (
        candidate_pairs_df.select(PAIR_ID_COLUMNS)
        .join(s1_lookup, left_on="source1_entity_id", right_on="entity_id", how="left")
        .join(cand_lookup, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_cand")
        .with_columns([
            pl.col("token_set").fill_null([]),
            pl.col("token_set_cand").fill_null([]),
            pl.col("numeric_tokens").fill_null([]),
            pl.col("numeric_tokens_cand").fill_null([]),
            pl.col("char_3grams").fill_null([]),
            pl.col("char_3grams_cand").fill_null([]),
        ])
    )

    # Missing flags handling null left joins safely
    is_miss_s1 = pl.col("is_missing").fill_null(True)
    is_miss_cand = pl.col("is_missing_cand").fill_null(True)
    either_miss = is_miss_s1 | is_miss_cand

    # List set operations
    token_inter = pl.col("token_set").list.set_intersection(pl.col("token_set_cand")).list.len()
    token_union = pl.col("token_set").list.set_union(pl.col("token_set_cand")).list.len()
    token_min = pl.min_horizontal(pl.col("token_set").list.len(), pl.col("token_set_cand").list.len())

    num_inter = pl.col("numeric_tokens").list.set_intersection(pl.col("numeric_tokens_cand")).list.len()

    char_inter = pl.col("char_3grams").list.set_intersection(pl.col("char_3grams_cand")).list.len()
    char_union = pl.col("char_3grams").list.set_union(pl.col("char_3grams_cand")).list.len()

    # Vectorized expressions
    out_df = joined.select([
        pl.col("source1_entity_id"),
        pl.col("candidate_entity_id"),
        pl.col("candidate_source"),
        # 1. address_exact
        pl.when(either_miss).then(-1)
          .when(pl.col("clean_address") == pl.col("clean_address_cand")).then(1)
          .otherwise(0).cast(pl.Int8).alias("address_exact"),
        # 2. address_token_jaccard
        pl.when(either_miss | (token_union == 0) | token_union.is_null()).then(0.0)
          .otherwise(token_inter / token_union).cast(pl.Float32).alias("address_token_jaccard"),
        # 3. address_token_overlap
        pl.when(either_miss | (token_min == 0) | token_min.is_null()).then(0.0)
          .otherwise(token_inter / token_min).cast(pl.Float32).alias("address_token_overlap"),
        # 4. address_char_3gram_similarity
        pl.when(either_miss | (char_union == 0) | char_union.is_null()).then(0.0)
          .otherwise(char_inter / char_union).cast(pl.Float32).alias("address_char_3gram_similarity"),
        # 5. shared_address_number_count
        pl.when(either_miss | num_inter.is_null()).then(0)
          .otherwise(num_inter).cast(pl.Int16).alias("shared_address_number_count"),
        # 6. address_number_overlap
        pl.when(either_miss | pl.col("primary_number").is_null() | pl.col("primary_number_cand").is_null()).then(-1)
          .when(pl.col("primary_number") == pl.col("primary_number_cand")).then(1)
          .otherwise(0).cast(pl.Int8).alias("address_number_overlap"),
        # 7. postal_match
        pl.when(either_miss | pl.col("postal_code").is_null() | pl.col("postal_code_cand").is_null()).then(-1)
          .when(pl.col("postal_code") == pl.col("postal_code_cand")).then(1)
          .otherwise(0).cast(pl.Int8).alias("postal_match"),
        # 8. address_length_difference
        pl.when(either_miss).then(-1)
          .otherwise((pl.col("char_length") - pl.col("char_length_cand")).abs()).cast(pl.Int16).alias("address_length_difference"),
        # 9. address_missing_s1
        is_miss_s1.cast(pl.Int8).alias("address_missing_s1"),
        # 10. address_missing_candidate
        is_miss_cand.cast(pl.Int8).alias("address_missing_candidate"),
    ])

    return out_df
