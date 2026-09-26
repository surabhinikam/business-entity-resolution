"""
Unified Feature Engineering Pipeline for Business Entity Resolution.

Integrates all four feature extraction modules:
1. Business Name Features (Person 1, 12 features)
2. Business Address Features (Person 2, 10 features)
3. Cross-Field Features (Person 2, 1 feature)
4. Blocking Provenance Features (Person 2, 6 features)

Design & Guarantees:
- Single Source of Truth: Uses FULL_FEATURE_SCHEMA and FULL_PIPELINE_COLUMNS from feature_schema.py.
- Deterministic Ordering: All output DataFrames strictly adhere to FULL_PIPELINE_COLUMNS.
- Preserves Identity: Canonical composite identity (source1_entity_id, candidate_entity_id, candidate_source).
- Type Safety: Enforces exact Polars DataTypes for all 29 features.
- Missing-Value Semantics: Preserves domain semantics (-1 for missing categorical flags, 0/0.0 for missing text similarities).
- Performance: 100% vectorized Polars operations avoiding row-by-row iteration over pairs.
- Reusability: Consumes precomputed record-level representations without redundant tokenization or normalization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    NAME_FEATURE_NAMES,
    NAME_FEATURE_SCHEMA,
    ADDRESS_FEATURE_NAMES,
    CROSS_FEATURE_NAMES,
    BLOCKING_FEATURE_NAMES,
    PERSON2_FEATURE_NAMES,
    PERSON2_FEATURE_SCHEMA,
    ALL_FEATURE_NAMES,
    FULL_PIPELINE_COLUMNS,
    FULL_FEATURE_SCHEMA,
    FEATURE_SCHEMA,
    AddressRecordRepresentation,
    BlockingRecordRepresentation,
)
from src.features.record_representation import (
    NameRepresentation,
    build_name_representation,
    build_address_representation,
)
from src.features.name_features import extract_name_features_batch
from src.features.address_features import extract_address_features_batch
from src.features.cross_features import extract_cross_features_batch
from src.features.blocking_features import (
    extract_blocking_features_batch,
    extract_blocking_features_from_provenance_dict,
    extract_blocking_features_from_keys,
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Unified end-to-end feature pipeline combining Name, Address, Cross-Field,
    and Blocking Provenance features.
    """

    def __init__(self, schema: Optional[Dict[str, pl.DataType]] = None):
        self.schema = schema or FULL_FEATURE_SCHEMA
        self.columns = [col for col in FULL_PIPELINE_COLUMNS if col in self.schema]

    def validate_candidate_pairs(self, candidate_pairs_df: pl.DataFrame) -> None:
        """Ensure candidate pairs DataFrame conforms to canonical composite identity."""
        for col in PAIR_ID_COLUMNS:
            if col not in candidate_pairs_df.columns:
                raise ValueError(
                    f"Candidate pairs DataFrame is missing required composite identity column: '{col}'. "
                    f"Expected columns: {PAIR_ID_COLUMNS}"
                )

    def generate_features(
        self,
        candidate_pairs_df: pl.DataFrame,
        s1_name_reps: Union[Dict[str, Any], pl.DataFrame],
        cand_name_reps: Union[Dict[str, Any], pl.DataFrame],
        s1_address_reps: Union[Dict[str, Any], pl.DataFrame],
        cand_address_reps: Union[Dict[str, Any], pl.DataFrame],
        s1_translit: Optional[Union[Dict[str, Optional[str]], pl.DataFrame]] = None,
        cand_translit: Optional[Union[Dict[str, Optional[str]], pl.DataFrame]] = None,
        s1_countries: Optional[Union[Dict[str, Any], pl.DataFrame]] = None,
        cand_countries: Optional[Union[Dict[str, Any], pl.DataFrame]] = None,
        pair_provenance: Optional[Union[Dict[Tuple[str, str], Any], pl.DataFrame]] = None,
        s1_blocking_keys: Optional[Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str]] = None,
        cand_blocking_keys: Optional[Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str, List[Any]]] = None,
    ) -> pl.DataFrame:
        """
        Executes the unified feature engineering pipeline across all 4 feature modules.

        Parameters:
            candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
            s1_name_reps: Source 1 NameRepresentation lookup or DataFrame.
            cand_name_reps: Candidate NameRepresentation lookup or DataFrame.
            s1_address_reps: Source 1 AddressRepresentation lookup or DataFrame.
            cand_address_reps: Candidate AddressRepresentation lookup or DataFrame.
            s1_translit: Optional Source 1 transliterated strings.
            cand_translit: Optional Candidate transliterated strings.
            s1_countries: Optional Source 1 countries (auto-derived from s1_address_reps if None).
            cand_countries: Optional Candidate countries (auto-derived from cand_address_reps if None).
            pair_provenance: Optional candidate generation provenance mapping.
            s1_blocking_keys: Optional Source 1 precomputed blocking keys.
            cand_blocking_keys: Optional Candidate precomputed blocking keys.

        Returns:
            Polars DataFrame with PAIR_ID_COLUMNS + ALL_FEATURE_NAMES (32 columns total),
            in deterministic order and cast to FULL_FEATURE_SCHEMA types.
        """
        self.validate_candidate_pairs(candidate_pairs_df)

        n_pairs = candidate_pairs_df.height
        logger.info(f"Extracting unified features for {n_pairs:,} candidate pairs...")

        # Handle empty candidate pairs
        if n_pairs == 0:
            return pl.DataFrame(schema=self.schema).select(self.columns)

        # 1. Name features (12 features)
        name_df = extract_name_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            s1_representations=s1_name_reps,
            cand_representations=cand_name_reps,
            s1_translit=s1_translit,
            cand_translit=cand_translit,
        )

        # 2. Address features (10 features)
        addr_df = extract_address_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            s1_representations=s1_address_reps,
            cand_representations=cand_address_reps,
        )

        # 3. Cross-field features (1 feature)
        if s1_countries is None:
            if isinstance(s1_address_reps, dict):
                s1_countries = {
                    eid: getattr(rep, "country_normalized", rep.get("country_normalized") if isinstance(rep, dict) else None)
                    for eid, rep in s1_address_reps.items()
                }
            elif isinstance(s1_address_reps, pl.DataFrame) and "country_normalized" in s1_address_reps.columns:
                s1_countries = s1_address_reps.select(["entity_id", "country_normalized"])
            else:
                s1_countries = {}

        if cand_countries is None:
            if isinstance(cand_address_reps, dict):
                cand_countries = {
                    eid: getattr(rep, "country_normalized", rep.get("country_normalized") if isinstance(rep, dict) else None)
                    for eid, rep in cand_address_reps.items()
                }
            elif isinstance(cand_address_reps, pl.DataFrame) and "country_normalized" in cand_address_reps.columns:
                cand_countries = cand_address_reps.select(["entity_id", "country_normalized"])
            else:
                cand_countries = {}

        cross_df = extract_cross_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            s1_countries=s1_countries,
            cand_countries=cand_countries,
        )

        # 4. Blocking provenance features (6 features)
        block_df = extract_blocking_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            pair_provenance=pair_provenance,
            s1_keys=s1_blocking_keys,
            cand_keys=cand_blocking_keys,
        )

        # 5. Horizontal join on canonical composite pair identity
        features_df = (
            candidate_pairs_df.select(PAIR_ID_COLUMNS)
            .join(name_df, on=PAIR_ID_COLUMNS, how="left")
            .join(addr_df, on=PAIR_ID_COLUMNS, how="left")
            .join(cross_df, on=PAIR_ID_COLUMNS, how="left")
            .join(block_df, on=PAIR_ID_COLUMNS, how="left")
        )

        # 6. Deterministic column selection and dtype casting
        ordered_df = features_df.select(self.columns)
        for col_name, expected_dtype in self.schema.items():
            if col_name in ordered_df.columns and ordered_df[col_name].dtype != expected_dtype:
                ordered_df = ordered_df.with_columns(
                    pl.col(col_name).cast(expected_dtype)
                )

        logger.info(
            f"Successfully generated full feature matrix: {ordered_df.width} columns, {ordered_df.height} rows."
        )
        return ordered_df

    def generate_features_from_records(
        self,
        candidate_pairs_df: pl.DataFrame,
        s1_records: Union[Dict[str, Dict[str, Any]], pl.DataFrame],
        cand_records: Union[Dict[str, Dict[str, Any]], pl.DataFrame],
        pair_provenance: Optional[Union[Dict[Tuple[str, str], Any], pl.DataFrame]] = None,
        s1_blocking_keys: Optional[Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str]] = None,
        cand_blocking_keys: Optional[Union[Dict[str, Any], pl.DataFrame, pl.LazyFrame, str, List[Any]]] = None,
    ) -> pl.DataFrame:
        """
        Convenience execution method that takes raw processed record dictionaries or DataFrames,
        derives precomputed representations for only the referenced entities, and extracts all features.
        """
        self.validate_candidate_pairs(candidate_pairs_df)

        # Identify unique entity IDs needed
        s1_needed = set(candidate_pairs_df["source1_entity_id"].unique().to_list())
        cand_needed = set(candidate_pairs_df["candidate_entity_id"].unique().to_list())

        # Build S1 representations
        s1_name_reps: Dict[str, NameRepresentation] = {}
        s1_addr_reps: Dict[str, AddressRecordRepresentation] = {}
        s1_translit: Dict[str, Optional[str]] = {}
        s1_countries: Dict[str, Optional[str]] = {}

        if isinstance(s1_records, pl.DataFrame):
            for row in s1_records.filter(pl.col("entity_id").is_in(list(s1_needed))).iter_rows(named=True):
                eid = str(row["entity_id"])
                s1_name_reps[eid] = build_name_representation(
                    row.get("business_name_normalized"),
                    row.get("business_name_tokens"),
                )
                s1_addr_reps[eid] = build_address_representation(row)
                s1_translit[eid] = row.get("business_name_transliterated")
                s1_countries[eid] = row.get("country_normalized")
        elif isinstance(s1_records, dict):
            for eid in s1_needed:
                r = s1_records.get(eid)
                if r is not None:
                    s1_name_reps[eid] = build_name_representation(
                        r.get("business_name_normalized"),
                        r.get("business_name_tokens"),
                    )
                    s1_addr_reps[eid] = build_address_representation(r)
                    s1_translit[eid] = r.get("business_name_transliterated")
                    s1_countries[eid] = r.get("country_normalized")

        # Build Candidate representations
        cand_name_reps: Dict[str, NameRepresentation] = {}
        cand_addr_reps: Dict[str, AddressRecordRepresentation] = {}
        cand_translit: Dict[str, Optional[str]] = {}
        cand_countries: Dict[str, Optional[str]] = {}

        if isinstance(cand_records, pl.DataFrame):
            for row in cand_records.filter(pl.col("entity_id").is_in(list(cand_needed))).iter_rows(named=True):
                eid = str(row["entity_id"])
                cand_name_reps[eid] = build_name_representation(
                    row.get("business_name_normalized"),
                    row.get("business_name_tokens"),
                )
                cand_addr_reps[eid] = build_address_representation(row)
                cand_translit[eid] = row.get("business_name_transliterated")
                cand_countries[eid] = row.get("country_normalized")
        elif isinstance(cand_records, dict):
            for eid in cand_needed:
                r = cand_records.get(eid)
                if r is not None:
                    cand_name_reps[eid] = build_name_representation(
                        r.get("business_name_normalized"),
                        r.get("business_name_tokens"),
                    )
                    cand_addr_reps[eid] = build_address_representation(r)
                    cand_translit[eid] = r.get("business_name_transliterated")
                    cand_countries[eid] = r.get("country_normalized")

        return self.generate_features(
            candidate_pairs_df=candidate_pairs_df,
            s1_name_reps=s1_name_reps,
            cand_name_reps=cand_name_reps,
            s1_address_reps=s1_addr_reps,
            cand_address_reps=cand_addr_reps,
            s1_translit=s1_translit,
            cand_translit=cand_translit,
            s1_countries=s1_countries,
            cand_countries=cand_countries,
            pair_provenance=pair_provenance,
            s1_blocking_keys=s1_blocking_keys,
            cand_blocking_keys=cand_blocking_keys,
        )


# Alias for unified naming
EntityResolutionFeaturePipeline = FeaturePipeline


class Person2FeaturePipeline:
    """
    Pipeline orchestrator for Person 2 features:
    - Address pair features
    - Cross-field features
    - Blocking provenance features
    """

    def __init__(self):
        self.schema = PERSON2_FEATURE_SCHEMA
        self.columns = [col for col in PAIR_ID_COLUMNS + PERSON2_FEATURE_NAMES if col in self.schema]

    def validate_candidate_pairs(self, candidate_pairs_df: pl.DataFrame) -> None:
        """Ensure candidate pairs DataFrame conforms to canonical composite identity."""
        for col in PAIR_ID_COLUMNS:
            if col not in candidate_pairs_df.columns:
                raise ValueError(
                    f"Candidate pairs DataFrame is missing required composite identity column: '{col}'. "
                    f"Expected columns: {PAIR_ID_COLUMNS}"
                )

    def generate_person2_features(
        self,
        candidate_pairs_df: pl.DataFrame,
        s1_address_reps: Dict[str, Any],
        cand_address_reps: Dict[str, Any],
        pair_provenance: Optional[Dict[Tuple[str, str], Any]] = None,
        s1_blocking_keys: Optional[Dict[str, Any]] = None,
        cand_blocking_keys: Optional[Dict[str, Any]] = None,
    ) -> pl.DataFrame:
        """
        Executes the Person 2 feature pipeline.
        """
        self.validate_candidate_pairs(candidate_pairs_df)

        if candidate_pairs_df.height == 0:
            return pl.DataFrame(schema=self.schema).select(self.columns)

        # 1. Address features
        addr_df = extract_address_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            s1_representations=s1_address_reps,
            cand_representations=cand_address_reps,
        )

        # 2. Cross-field features
        s1_countries = {
            eid: getattr(rep, "country_normalized", rep.get("country_normalized") if isinstance(rep, dict) else None)
            for eid, rep in s1_address_reps.items()
        }
        cand_countries = {
            eid: getattr(rep, "country_normalized", rep.get("country_normalized") if isinstance(rep, dict) else None)
            for eid, rep in cand_address_reps.items()
        }
        cross_df = extract_cross_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            s1_countries=s1_countries,
            cand_countries=cand_countries,
        )

        # 3. Blocking provenance features
        block_df = extract_blocking_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            pair_provenance=pair_provenance,
            s1_keys=s1_blocking_keys,
            cand_keys=cand_blocking_keys,
        )

        # 4. Horizontal join on composite pair identity
        features_df = (
            candidate_pairs_df.select(PAIR_ID_COLUMNS)
            .join(addr_df, on=PAIR_ID_COLUMNS, how="left")
            .join(cross_df, on=PAIR_ID_COLUMNS, how="left")
            .join(block_df, on=PAIR_ID_COLUMNS, how="left")
        )

        # 5. Deterministic column ordering & dtype verification
        ordered_df = features_df.select(self.columns)
        for col_name, expected_dtype in self.schema.items():
            if col_name in ordered_df.columns and ordered_df[col_name].dtype != expected_dtype:
                ordered_df = ordered_df.with_columns(
                    pl.col(col_name).cast(expected_dtype)
                )

        return ordered_df


def combine_person1_and_person2_features(
    person1_df: pl.DataFrame,
    person2_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Contract interface for combining Person 1's name features with Person 2's features.

    Both DataFrames MUST share the canonical composite identity:
    (source1_entity_id, candidate_entity_id, candidate_source).

    Performs an inner join on the composite identity, preserving all feature columns
    in deterministic canonical order.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in person1_df.columns:
            raise ValueError(f"Person 1 DataFrame missing required composite identity column: '{col}'")
        if col not in person2_df.columns:
            raise ValueError(f"Person 2 DataFrame missing required composite identity column: '{col}'")

    combined = person1_df.join(
        person2_df,
        on=PAIR_ID_COLUMNS,
        how="inner",
    )

    # Reorder columns deterministically if full columns are present
    common_cols = [c for c in FULL_PIPELINE_COLUMNS if c in combined.columns]
    combined = combined.select(common_cols)

    # Cast dtypes to schema
    for col in common_cols:
        if col in FULL_FEATURE_SCHEMA and combined[col].dtype != FULL_FEATURE_SCHEMA[col]:
            combined = combined.with_columns(pl.col(col).cast(FULL_FEATURE_SCHEMA[col]))

    return combined
