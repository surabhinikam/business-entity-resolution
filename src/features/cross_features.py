"""
Cross-field feature extraction module for Person 2.

Features covered:
1. country_match:
   Categorical match indicator:
   1 = S1 and candidate have the same normalized country
   0 = both are present but different
   -1 = either country is missing, empty, or sentinel value

Uses existing country_normalized field without duplicating preprocessing.
Supports both single-pair and vectorized Polars batch DataFrame calculations.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Union
import polars as pl

from src.features.feature_schema import (
    CROSS_FEATURE_NAMES,
    PAIR_ID_COLUMNS,
    PERSON2_FEATURE_SCHEMA,
)

logger = logging.getLogger(__name__)

_SENTINEL_EMPTY_STRINGS = frozenset({"", "none", "nan", "null"})
_SENTINEL_LIST = ["", "none", "nan", "null"]


def compute_cross_features(
    s1_country: Optional[str] | Any,
    cand_country: Optional[str] | Any,
) -> Dict[str, int]:
    """
    Computes cross-field features for a single candidate pair.

    Parameters:
        s1_country: Normalized country string (or representation object) for Source 1.
        cand_country: Normalized country string (or representation object) for Candidate.

    Returns:
        Dict mapping each feature in CROSS_FEATURE_NAMES to its typed value.
    """
    if hasattr(s1_country, "country_normalized"):
        s1_country = s1_country.country_normalized
    elif isinstance(s1_country, dict):
        s1_country = s1_country.get("country_normalized", s1_country.get("country"))

    if hasattr(cand_country, "country_normalized"):
        cand_country = cand_country.country_normalized
    elif isinstance(cand_country, dict):
        cand_country = cand_country.get("country_normalized", cand_country.get("country"))

    if s1_country is None or cand_country is None:
        return {"country_match": -1}

    c1 = str(s1_country).strip().lower()
    c2 = str(cand_country).strip().lower()

    if not c1 or not c2 or c1 in _SENTINEL_EMPTY_STRINGS or c2 in _SENTINEL_EMPTY_STRINGS:
        return {"country_match": -1}

    return {"country_match": 1 if c1 == c2 else 0}


def _to_country_lookup_df(
    countries: Union[Dict[str, Any], pl.DataFrame],
) -> pl.DataFrame:
    """Convert lookup mappings or DataFrames to a typed Polars lookup DataFrame."""
    if isinstance(countries, pl.DataFrame):
        if "entity_id" not in countries.columns or "country_normalized" not in countries.columns:
            raise ValueError("Input DataFrame must have 'entity_id' and 'country_normalized' columns")
        return countries.select(["entity_id", "country_normalized"])

    rows: List[Dict[str, Optional[str]]] = []
    for eid, val in countries.items():
        if isinstance(val, str) or val is None:
            c = val
        elif isinstance(val, dict):
            c = val.get("country_normalized", val.get("country"))
        else:
            c = getattr(val, "country_normalized", getattr(val, "country", None))

        rows.append({
            "entity_id": str(eid),
            "country_normalized": str(c) if c is not None else None,
        })

    lookup_schema = {
        "entity_id": pl.Utf8,
        "country_normalized": pl.Utf8,
    }

    if not rows:
        return pl.DataFrame({
            "entity_id": pl.Series([], dtype=pl.Utf8),
            "country_normalized": pl.Series([], dtype=pl.Utf8),
        }, schema=lookup_schema)

    return pl.DataFrame(rows, schema=lookup_schema)


def extract_cross_features_batch(
    candidate_pairs_df: pl.DataFrame,
    s1_countries: Union[Dict[str, Any], pl.DataFrame],
    cand_countries: Union[Dict[str, Any], pl.DataFrame],
) -> pl.DataFrame:
    """
    Vectorized batch calculation of cross-field features using Polars.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        s1_countries: Lookup mapping or DataFrame for Source 1 country_normalized.
        cand_countries: Lookup mapping or DataFrame for Candidate country_normalized.

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

    # 1. Prepare lookup tables
    s1_lookup = _to_country_lookup_df(s1_countries)
    cand_lookup = _to_country_lookup_df(cand_countries)

    # 2. Vectorized join
    joined = (
        candidate_pairs_df.select(PAIR_ID_COLUMNS)
        .join(s1_lookup, left_on="source1_entity_id", right_on="entity_id", how="left")
        .join(cand_lookup, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_cand")
    )

    # 3. Vectorized cleaning and condition evaluation
    c1 = pl.col("country_normalized").str.strip_chars().str.to_lowercase()
    c2 = pl.col("country_normalized_cand").str.strip_chars().str.to_lowercase()

    miss1 = c1.is_null() | c1.is_in(_SENTINEL_LIST)
    miss2 = c2.is_null() | c2.is_in(_SENTINEL_LIST)
    either_miss = miss1 | miss2

    out_df = joined.select([
        pl.col("source1_entity_id"),
        pl.col("candidate_entity_id"),
        pl.col("candidate_source"),
        pl.when(either_miss).then(-1)
          .when(c1 == c2).then(1)
          .otherwise(0).cast(pl.Int8).alias("country_match"),
    ])

    return out_df
