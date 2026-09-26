"""
Blocking provenance feature extraction interfaces and scaffolding for Person 2.

Features covered:
- matched_key_A: Binary flag (1 if pair shared Key A, 0 otherwise)
- matched_key_C: Binary flag (1 if pair shared Key C, 0 otherwise)
- matched_key_D: Binary flag (1 if pair shared Key D, 0 otherwise)
- matched_key_E: Binary flag (1 if pair shared Key E, 0 otherwise)
- matched_key_F: Binary flag (1 if pair shared Key F, 0 otherwise)
- matched_key_count: Integer count of distinct blocking keys producing this pair (1 to 5)

Supports two ingestion modes:
1. Direct extraction from candidate generation provenance mapping (BlockIndex.generate_pairs)
2. Derivation from precomputed blocking keys stored in data/processed/train/blocking_keys/
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import polars as pl

from src.features.feature_schema import (
    BLOCKING_FEATURE_NAMES,
    PAIR_ID_COLUMNS,
    BlockingRecordRepresentation,
)

logger = logging.getLogger(__name__)

VALID_KEYS = ["A", "C", "D", "E", "F"]


def compute_blocking_features(
    matched_keys_set: Set[str],
) -> Dict[str, Any]:
    """
    Computes blocking provenance features for a single candidate pair given its set of matched keys.

    Parameters:
        matched_keys_set: Set of key labels (e.g. {'A', 'C'} or {'D'}).

    Returns:
        Dict mapping BLOCKING_FEATURE_NAMES to typed values.
    """
    feat: Dict[str, Any] = {}
    for k in VALID_KEYS:
        feat[f"matched_key_{k}"] = 1 if k in matched_keys_set else 0

    feat["matched_key_count"] = sum(feat[f"matched_key_{k}"] for k in VALID_KEYS)
    return feat


def extract_blocking_features_from_provenance_dict(
    candidate_pairs_df: pl.DataFrame,
    pair_provenance: Dict[Tuple[str, str], Set[str]],
) -> pl.DataFrame:
    """
    Extracts blocking features from an in-memory provenance mapping produced by BlockIndex.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        pair_provenance: Dict mapping (source1_entity_id, candidate_entity_id) -> set of key labels.

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    if candidate_pairs_df.height == 0:
        schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
        for k in BLOCKING_FEATURE_NAMES:
            schema[k] = pl.Int8
        return pl.DataFrame(schema=schema)

    s1_ids = candidate_pairs_df["source1_entity_id"].to_list()
    c_ids = candidate_pairs_df["candidate_entity_id"].to_list()
    sources = candidate_pairs_df["candidate_source"].to_list()

    rows = []
    for s1_id, c_id, src in zip(s1_ids, c_ids, sources):
        keys = pair_provenance.get((s1_id, c_id), set())
        feat = compute_blocking_features(keys)
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": c_id,
            "candidate_source": src,
            **feat,
        })

    return pl.DataFrame(rows).with_columns([
        pl.col(col).cast(pl.Int8) for col in BLOCKING_FEATURE_NAMES
    ])


def extract_blocking_features_from_keys(
    candidate_pairs_df: pl.DataFrame,
    s1_keys: Dict[str, BlockingRecordRepresentation],
    cand_keys: Dict[str, BlockingRecordRepresentation],
) -> pl.DataFrame:
    """
    Extracts blocking features by comparing precomputed blocking keys for S1 and Candidate.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        s1_keys: Lookup mapping S1 entity_id -> BlockingRecordRepresentation.
        cand_keys: Lookup mapping Candidate entity_id -> BlockingRecordRepresentation.

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    if candidate_pairs_df.height == 0:
        schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
        for k in BLOCKING_FEATURE_NAMES:
            schema[k] = pl.Int8
        return pl.DataFrame(schema=schema)

    s1_ids = candidate_pairs_df["source1_entity_id"].to_list()
    c_ids = candidate_pairs_df["candidate_entity_id"].to_list()
    sources = candidate_pairs_df["candidate_source"].to_list()

    rows = []
    for s1_id, c_id, src in zip(s1_ids, c_ids, sources):
        rep1 = s1_keys.get(s1_id)
        repc = cand_keys.get(c_id)

        matched_keys: Set[str] = set()
        if rep1 is not None and repc is not None:
            for k in VALID_KEYS:
                k1 = getattr(rep1, f"key_{k}", None)
                kc = getattr(repc, f"key_{k}", None)
                if k1 and k1 == kc:
                    matched_keys.add(k)

        feat = compute_blocking_features(matched_keys)
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": c_id,
            "candidate_source": src,
            **feat,
        })

    return pl.DataFrame(rows).with_columns([
        pl.col(col).cast(pl.Int8) for col in BLOCKING_FEATURE_NAMES
    ])
