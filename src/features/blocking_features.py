"""
Blocking provenance feature extraction for Person 2.

Features implemented:
- matched_key_A: Binary flag (1 if pair shared Key A, 0 otherwise)
- matched_key_C: Binary flag (1 if pair shared Key C, 0 otherwise)
- matched_key_D: Binary flag (1 if pair shared Key D, 0 otherwise)
- matched_key_E: Binary flag (1 if pair shared Key E, 0 otherwise)
- matched_key_F: Binary flag (1 if pair shared Key F, 0 otherwise)
- matched_key_count: Integer count of distinct blocking keys producing this pair (0 to 5)

Semantics:
- matched_key_X = 1 if the candidate pair was generated through blocking key X, otherwise 0.
- matched_key_count = sum of the five matched_key_* features.
- Preserves pair composite identity:
  (source1_entity_id, candidate_entity_id, candidate_source)
- Supported ingestion modes:
  1. Direct extraction from candidate generation provenance mapping (BlockIndex.generate_pairs)
  2. Derivation from precomputed blocking keys stored in data/processed/train/blocking_keys/
  3. Single-pair feature computation for online evaluation or testing
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import polars as pl

from src.features.feature_schema import (
    BLOCKING_FEATURE_NAMES,
    PAIR_ID_COLUMNS,
    BlockingRecordRepresentation,
)

logger = logging.getLogger(__name__)

# Frozen production key set: A, C, D, E, F (Key B permanently excluded)
VALID_KEYS: List[str] = ["A", "C", "D", "E", "F"]


def _clean_key_label(key: Any) -> Optional[str]:
    """Extract canonical key label ('A', 'C', 'D', 'E', 'F') from diverse input forms."""
    if not key:
        return None
    k_str = str(key).strip()
    if "||" in k_str:
        k_str = k_str.split("||")[0].strip()
    k_str = k_str.upper().replace("KEY_", "").strip()
    return k_str if k_str in VALID_KEYS else None


def compute_blocking_features(
    matched_keys_or_s1: Any = None,
    maybe_cand: Optional[Any] = None,
) -> Dict[str, int]:
    """
    Computes blocking provenance features for a single candidate pair.

    Supports two invocation styles:
    1. Single argument of matched keys:
       compute_blocking_features({"A", "C"}) -> {"matched_key_A": 1, ..., "matched_key_count": 2}
    2. Two arguments of record representations or dicts:
       compute_blocking_features(s1_record, cand_record)

    Parameters:
        matched_keys_or_s1: Set/collection of key labels OR Source 1 record representation.
        maybe_cand: Candidate record representation when comparing two records directly.

    Returns:
        Dict mapping BLOCKING_FEATURE_NAMES to integer values (0 or 1, and count 0..5).
    """
    if maybe_cand is not None:
        return compute_blocking_pair_features(matched_keys_or_s1, maybe_cand)

    clean_keys: Set[str] = set()
    if matched_keys_or_s1 is not None:
        if isinstance(matched_keys_or_s1, (set, list, tuple, frozenset)):
            for k in matched_keys_or_s1:
                label = _clean_key_label(k)
                if label:
                    clean_keys.add(label)
        elif isinstance(matched_keys_or_s1, str):
            label = _clean_key_label(matched_keys_or_s1)
            if label:
                clean_keys.add(label)

    feat: Dict[str, int] = {}
    for k in VALID_KEYS:
        feat[f"matched_key_{k}"] = 1 if k in clean_keys else 0

    feat["matched_key_count"] = sum(feat[f"matched_key_{k}"] for k in VALID_KEYS)
    return feat


def compute_blocking_pair_features(
    s1_record: Any,
    cand_record: Any,
) -> Dict[str, int]:
    """
    Computes blocking provenance features for a single candidate pair by comparing their blocking keys.

    Parameters:
        s1_record: Source 1 record (BlockingRecordRepresentation or dict with key_A..key_F).
        cand_record: Candidate record (BlockingRecordRepresentation or dict with key_A..key_F).

    Returns:
        Dict mapping BLOCKING_FEATURE_NAMES to integer values (0 or 1, and count 0..5).
    """
    matched: Set[str] = set()
    if s1_record is not None and cand_record is not None:
        for k in VALID_KEYS:
            attr = f"key_{k}"
            val1 = (
                s1_record.get(attr)
                if isinstance(s1_record, dict)
                else getattr(s1_record, attr, None)
            )
            valc = (
                cand_record.get(attr)
                if isinstance(cand_record, dict)
                else getattr(cand_record, attr, None)
            )
            if (
                val1 is not None
                and valc is not None
                and str(val1).strip() != ""
                and str(val1).strip() == str(valc).strip()
            ):
                matched.add(k)

    return compute_blocking_features(matched)


def extract_blocking_features_from_provenance_dict(
    candidate_pairs_df: pl.DataFrame,
    pair_provenance: Union[Dict[Tuple[str, str], Any], pl.DataFrame],
) -> pl.DataFrame:
    """
    Extracts blocking provenance features from an in-memory provenance mapping produced by BlockIndex.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        pair_provenance: Dict mapping (source1_entity_id, candidate_entity_id) -> set of key labels,
                         or a pre-structured DataFrame containing provenance.

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES cast to pl.Int8.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    if candidate_pairs_df.height == 0:
        schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
        for k in BLOCKING_FEATURE_NAMES:
            schema[k] = pl.Int8
        return pl.DataFrame(schema=schema)

    # Fast path: Empty provenance dict gives zeroes for all pairs
    if isinstance(pair_provenance, dict) and not pair_provenance:
        return candidate_pairs_df.select(PAIR_ID_COLUMNS).with_columns(
            [pl.lit(0, dtype=pl.Int8).alias(f"matched_key_{k}") for k in VALID_KEYS]
            + [pl.lit(0, dtype=pl.Int8).alias("matched_key_count")]
        )

    # Ingest provenance DataFrame or vectorize dict into DataFrame
    if isinstance(pair_provenance, pl.DataFrame):
        prov_df = pair_provenance
    elif isinstance(pair_provenance, dict):
        prov_s1: List[str] = []
        prov_cand: List[str] = []
        prov_cols: Dict[str, List[int]] = {k: [] for k in VALID_KEYS}

        for (s1_id, c_id), keys in pair_provenance.items():
            prov_s1.append(str(s1_id))
            prov_cand.append(str(c_id))

            clean_kset: Set[str] = set()
            if isinstance(keys, (set, list, tuple, frozenset)):
                for k in keys:
                    lbl = _clean_key_label(k)
                    if lbl:
                        clean_kset.add(lbl)
            elif isinstance(keys, str):
                lbl = _clean_key_label(keys)
                if lbl:
                    clean_kset.add(lbl)

            for k in VALID_KEYS:
                prov_cols[k].append(1 if k in clean_kset else 0)

        prov_df = pl.DataFrame({
            "source1_entity_id": pl.Series(prov_s1, dtype=pl.Utf8),
            "candidate_entity_id": pl.Series(prov_cand, dtype=pl.Utf8),
            **{
                f"matched_key_{k}": pl.Series(prov_cols[k], dtype=pl.Int8)
                for k in VALID_KEYS
            },
        })
    else:
        raise TypeError(f"Unsupported pair_provenance type: {type(pair_provenance)}")

    # Vectorized left join with candidate pairs
    joined = (
        candidate_pairs_df.select(PAIR_ID_COLUMNS)
        .join(
            prov_df,
            on=["source1_entity_id", "candidate_entity_id"],
            how="left",
        )
        .with_columns([
            pl.col(f"matched_key_{k}").fill_null(0).cast(pl.Int8)
            for k in VALID_KEYS
        ])
        .with_columns(
            pl.sum_horizontal([pl.col(f"matched_key_{k}") for k in VALID_KEYS])
            .cast(pl.Int8)
            .alias("matched_key_count")
        )
    )

    return joined.select(PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES)


def _normalize_keys_input(
    keys_input: Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str, List[Any], Tuple[Any, ...], None],
) -> pl.DataFrame:
    """Normalizes heterogeneous blocking keys inputs into a standard Polars DataFrame."""
    if keys_input is None:
        schema = {"entity_id": pl.Utf8}
        for k in VALID_KEYS:
            schema[f"key_{k}"] = pl.Utf8
        return pl.DataFrame(schema=schema)

    if isinstance(keys_input, (list, tuple)):
        dfs = [_normalize_keys_input(item) for item in keys_input]
        return pl.concat(dfs) if dfs else _normalize_keys_input(None)

    if isinstance(keys_input, pl.DataFrame):
        return keys_input

    if isinstance(keys_input, pl.LazyFrame):
        return keys_input.collect()

    if isinstance(keys_input, str):
        return pl.read_parquet(keys_input)

    if isinstance(keys_input, dict):
        if not keys_input:
            return _normalize_keys_input(None)

        sample = next(iter(keys_input.values()))
        keys_schema = {"entity_id": pl.Utf8, **{f"key_{k}": pl.Utf8 for k in VALID_KEYS}}
        if hasattr(sample, "key_A"):
            return pl.DataFrame({
                "entity_id": [str(k) for k in keys_input.keys()],
                "key_A": [getattr(r, "key_A", None) for r in keys_input.values()],
                "key_C": [getattr(r, "key_C", None) for r in keys_input.values()],
                "key_D": [getattr(r, "key_D", None) for r in keys_input.values()],
                "key_E": [getattr(r, "key_E", None) for r in keys_input.values()],
                "key_F": [getattr(r, "key_F", None) for r in keys_input.values()],
            }, schema=keys_schema)
        elif isinstance(sample, dict):
            return pl.DataFrame({
                "entity_id": [str(k) for k in keys_input.keys()],
                "key_A": [r.get("key_A") for r in keys_input.values()],
                "key_C": [r.get("key_C") for r in keys_input.values()],
                "key_D": [r.get("key_D") for r in keys_input.values()],
                "key_E": [r.get("key_E") for r in keys_input.values()],
                "key_F": [r.get("key_F") for r in keys_input.values()],
            }, schema=keys_schema)

    raise TypeError(f"Unsupported keys_input type: {type(keys_input)}")


def extract_blocking_features_from_keys(
    candidate_pairs_df: pl.DataFrame,
    s1_keys: Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str],
    cand_keys: Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str, List[Any], Tuple[Any, ...]],
) -> pl.DataFrame:
    """
    Extracts blocking features by comparing precomputed blocking keys for S1 and Candidate.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        s1_keys: Lookup mapping, DataFrame, LazyFrame, or parquet path for Source 1 keys.
        cand_keys: Lookup mapping, DataFrame, LazyFrame, parquet path, or list of paths for Candidate keys.

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES cast to pl.Int8.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    if candidate_pairs_df.height == 0:
        schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
        for k in BLOCKING_FEATURE_NAMES:
            schema[k] = pl.Int8
        return pl.DataFrame(schema=schema)

    s1_df = _normalize_keys_input(s1_keys)
    cand_df = _normalize_keys_input(cand_keys)

    s1_cols = [f"key_{k}" for k in VALID_KEYS if f"key_{k}" in s1_df.columns]
    cand_cols = [f"key_{k}" for k in VALID_KEYS if f"key_{k}" in cand_df.columns]

    s1_sub = s1_df.select(["entity_id"] + s1_cols).with_columns([
        pl.col(c).cast(pl.Utf8) for c in s1_cols
    ])
    cand_sub = cand_df.select(["entity_id"] + cand_cols).with_columns([
        pl.col(c).cast(pl.Utf8) for c in cand_cols
    ])

    # Vectorized left joins on entity IDs
    joined = (
        candidate_pairs_df.select(PAIR_ID_COLUMNS)
        .join(s1_sub, left_on="source1_entity_id", right_on="entity_id", how="left")
        .join(cand_sub, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_cand")
    )

    # Vectorized key match expressions
    match_exprs = []
    for k in VALID_KEYS:
        k_s1 = f"key_{k}"
        k_cand = f"key_{k}_cand"
        if k_s1 in joined.columns and k_cand in joined.columns:
            expr = (
                pl.when(
                    pl.col(k_s1).is_not_null()
                    & pl.col(k_cand).is_not_null()
                    & (pl.col(k_s1).str.strip_chars() != "")
                    & (pl.col(k_s1).str.strip_chars() == pl.col(k_cand).str.strip_chars())
                )
                .then(pl.lit(1, dtype=pl.Int8))
                .otherwise(pl.lit(0, dtype=pl.Int8))
                .alias(f"matched_key_{k}")
            )
        else:
            expr = pl.lit(0, dtype=pl.Int8).alias(f"matched_key_{k}")
        match_exprs.append(expr)

    result = (
        joined.with_columns(match_exprs)
        .with_columns(
            pl.sum_horizontal([pl.col(f"matched_key_{k}") for k in VALID_KEYS])
            .cast(pl.Int8)
            .alias("matched_key_count")
        )
        .select(PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES)
    )

    return result


def extract_blocking_features_batch(
    candidate_pairs_df: pl.DataFrame,
    pair_provenance: Optional[Union[Dict[Tuple[str, str], Any], pl.DataFrame]] = None,
    s1_keys: Optional[Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str]] = None,
    cand_keys: Optional[Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str, List[Any], Tuple[Any, ...]]] = None,
) -> pl.DataFrame:
    """
    Unified batch calculation for blocking provenance features.

    Priority:
    1. If pair_provenance is provided, uses extract_blocking_features_from_provenance_dict.
    2. Else if s1_keys and cand_keys are provided, uses extract_blocking_features_from_keys.
    3. Otherwise, returns zero-filled blocking features.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        pair_provenance: Optional in-memory provenance mapping or DataFrame.
        s1_keys: Optional Source 1 blocking keys (mapping, DataFrame, or path).
        cand_keys: Optional Candidate blocking keys (mapping, DataFrame, or path).

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + BLOCKING_FEATURE_NAMES.
    """
    if pair_provenance is not None:
        return extract_blocking_features_from_provenance_dict(
            candidate_pairs_df=candidate_pairs_df,
            pair_provenance=pair_provenance,
        )
    if s1_keys is not None and cand_keys is not None:
        return extract_blocking_features_from_keys(
            candidate_pairs_df=candidate_pairs_df,
            s1_keys=s1_keys,
            cand_keys=cand_keys,
        )

    # Fallback: Zeroes
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    return candidate_pairs_df.select(PAIR_ID_COLUMNS).with_columns(
        [pl.lit(0, dtype=pl.Int8).alias(f"matched_key_{k}") for k in VALID_KEYS]
        + [pl.lit(0, dtype=pl.Int8).alias("matched_key_count")]
    )
