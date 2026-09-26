"""
Address feature extraction interfaces and scaffolding for Person 2.

Consumes preprocessed record-level address representations rather than
re-implementing normalization or tokenization.

Features covered:
- address_exact
- address_token_jaccard
- address_token_overlap
- address_char_3gram_similarity
- shared_address_number_count
- address_number_overlap
- postal_match
- address_length_difference
- address_missing_s1
- address_missing_candidate
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import polars as pl

from src.features.feature_schema import (
    AddressRecordRepresentation,
    ADDRESS_FEATURE_NAMES,
    PAIR_ID_COLUMNS,
)
from src.candidate_generation.address_parser import extract_address_number

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Record-Level Representation Builder
# =============================================================================

def build_address_representation(
    record: Dict[str, Any],
) -> AddressRecordRepresentation:
    """
    Constructs an AddressRecordRepresentation from a processed record.
    
    Reuses existing precomputed fields:
    - business_address_normalized
    - business_address_tokens
    - country_normalized
    
    Does NOT rerun text normalization or tokenization.
    Extracts address number once per entity for downstream comparison.
    """
    entity_id = record.get("entity_id", "")
    source = record.get("source", "")
    country_norm = record.get("country_normalized")
    raw_addr = record.get("business_address")
    norm_addr = record.get("business_address_normalized")
    tokens = record.get("business_address_tokens")
    
    if isinstance(tokens, str):
        tokens = tokens.split()
    elif tokens is None:
        tokens = []

    is_missing = not bool(norm_addr and norm_addr.strip())
    
    # Extract primary address number using existing address parser
    primary_num = None
    if not is_missing:
        num, has_num = extract_address_number(norm_addr)
        if has_num:
            primary_num = num

    # Note on postal_code:
    # Processed records do not have a dedicated postal column.
    # If postal extraction is introduced in Phase 3 implementation,
    # it will populate this field. Default to None.
    postal = None

    return AddressRecordRepresentation(
        entity_id=entity_id,
        source=source,
        country_normalized=country_norm,
        raw_address=raw_addr,
        normalized_address=norm_addr,
        address_tokens=tokens,
        primary_address_number=primary_num,
        all_numeric_tokens=set(),  # To be populated if full numeric token set needed
        postal_code=postal,
        is_address_missing=is_missing,
    )


# =============================================================================
# 2. Pair Feature Scaffolding & Interfaces
# =============================================================================

def compute_address_pair_features(
    s1: AddressRecordRepresentation,
    cand: AddressRecordRepresentation,
) -> Dict[str, Any]:
    """
    Contract for computing address pair features between two records.
    
    Scaffolding only: defines signature, contracts, null handling,
    and returns schema-compliant default feature dictionary.
    Expensive similarity math is implemented in subsequent steps.

    Parameters:
        s1: Address record representation for Source 1 entity.
        cand: Address record representation for Candidate entity (Source 2 or 3).

    Returns:
        Dict mapping each feature in ADDRESS_FEATURE_NAMES to its typed value.
    """
    # 1. Missingness indicators (can be computed immediately)
    missing_s1 = 1 if s1.is_address_missing else 0
    missing_cand = 1 if cand.is_address_missing else 0
    
    # If either address is missing, default null semantics apply:
    if s1.is_address_missing or cand.is_address_missing:
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

    # Contract placeholder for when both addresses are present:
    # Full implementation will calculate Jaccard, Overlap, 3-gram, numbers, etc.
    return {
        "address_exact": 0,
        "address_token_jaccard": 0.0,
        "address_token_overlap": 0.0,
        "address_char_3gram_similarity": 0.0,
        "shared_address_number_count": 0,
        "address_number_overlap": -1,
        "postal_match": -1,
        "address_length_difference": 0,
        "address_missing_s1": 0,
        "address_missing_candidate": 0,
    }


def extract_address_features_batch(
    candidate_pairs_df: pl.DataFrame,
    s1_representations: Dict[str, AddressRecordRepresentation],
    cand_representations: Dict[str, AddressRecordRepresentation],
) -> pl.DataFrame:
    """
    Batch interface to compute address features for a collection of candidate pairs.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        s1_representations: Lookup mapping S1 entity_id -> AddressRecordRepresentation.
        cand_representations: Lookup mapping Candidate entity_id -> AddressRecordRepresentation.

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + ADDRESS_FEATURE_NAMES.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    # Scaffolding: Returns empty typed DataFrame if input is empty,
    # or schema-compliant placeholder frame for integration tests.
    n_pairs = candidate_pairs_df.height
    if n_pairs == 0:
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

    # Scaffolding batch loop (demonstrating contract without expensive math)
    records = []
    s1_ids = candidate_pairs_df["source1_entity_id"].to_list()
    c_ids = candidate_pairs_df["candidate_entity_id"].to_list()
    sources = candidate_pairs_df["candidate_source"].to_list()

    for s1_id, c_id, src in zip(s1_ids, c_ids, sources):
        s1_rep = s1_representations.get(s1_id)
        c_rep = cand_representations.get(c_id)
        if s1_rep is None or c_rep is None:
            feat = {k: -1 if "match" in k or "exact" in k or "diff" in k else 0 for k in ADDRESS_FEATURE_NAMES}
        else:
            feat = compute_address_pair_features(s1_rep, c_rep)
        
        row = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": c_id,
            "candidate_source": src,
            **feat,
        }
        records.append(row)

    return pl.DataFrame(records).with_columns([
        pl.col("address_exact").cast(pl.Int8),
        pl.col("address_token_jaccard").cast(pl.Float32),
        pl.col("address_token_overlap").cast(pl.Float32),
        pl.col("address_char_3gram_similarity").cast(pl.Float32),
        pl.col("shared_address_number_count").cast(pl.Int16),
        pl.col("address_number_overlap").cast(pl.Int8),
        pl.col("postal_match").cast(pl.Int8),
        pl.col("address_length_difference").cast(pl.Int16),
        pl.col("address_missing_s1").cast(pl.Int8),
        pl.col("address_missing_candidate").cast(pl.Int8),
    ])
