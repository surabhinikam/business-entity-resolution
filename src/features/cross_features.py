"""
Cross-field feature extraction interfaces and scaffolding for Person 2.

Features covered:
- country_match:
  Categorical match indicator:
  1 if normalized countries match exactly and are non-null
  0 if normalized countries mismatch
  -1 if either normalized country is null or unpopulated
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
import polars as pl

from src.features.feature_schema import (
    CROSS_FEATURE_NAMES,
    PAIR_ID_COLUMNS,
)

logger = logging.getLogger(__name__)


def compute_cross_features(
    s1_country: Optional[str],
    cand_country: Optional[str],
) -> Dict[str, Any]:
    """
    Computes cross-field features for a single candidate pair.

    Parameters:
        s1_country: Normalized country string for Source 1 entity.
        cand_country: Normalized country string for Candidate entity.

    Returns:
        Dict mapping each feature in CROSS_FEATURE_NAMES to its typed value.
    """
    if not s1_country or not cand_country:
        return {"country_match": -1}

    c1 = s1_country.strip().lower()
    c2 = cand_country.strip().lower()

    if not c1 or not c2:
        return {"country_match": -1}

    return {"country_match": 1 if c1 == c2 else 0}


def extract_cross_features_batch(
    candidate_pairs_df: pl.DataFrame,
    s1_countries: Dict[str, Optional[str]],
    cand_countries: Dict[str, Optional[str]],
) -> pl.DataFrame:
    """
    Batch interface to extract cross-field features for a candidate pairs DataFrame.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        s1_countries: Lookup mapping S1 entity_id -> country_normalized.
        cand_countries: Lookup mapping Candidate entity_id -> country_normalized.

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + CROSS_FEATURE_NAMES.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    if candidate_pairs_df.height == 0:
        schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
        schema["country_match"] = pl.Int8
        return pl.DataFrame(schema=schema)

    s1_ids = candidate_pairs_df["source1_entity_id"].to_list()
    c_ids = candidate_pairs_df["candidate_entity_id"].to_list()
    sources = candidate_pairs_df["candidate_source"].to_list()

    matches = []
    for s1_id, c_id in zip(s1_ids, c_ids):
        c1 = s1_countries.get(s1_id)
        c2 = cand_countries.get(c_id)
        res = compute_cross_features(c1, c2)
        matches.append(res["country_match"])

    out_df = pl.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": c_ids,
        "candidate_source": sources,
        "country_match": matches,
    }).with_columns(pl.col("country_match").cast(pl.Int8))

    return out_df
