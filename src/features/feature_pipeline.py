"""
Person 2 Feature Engineering Pipeline.

Orchestrates:
1. Ingestion of candidate pairs with composite identity:
   (source1_entity_id, candidate_entity_id, candidate_source)
2. Ingestion/lookup of reusable record-level representations (Address & Blocking)
3. Address pair feature extraction
4. Cross-field feature extraction
5. Blocking provenance feature extraction
6. Horizontal combination and schema validation
7. Clean interface for final combination with Person 1's name features
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    PERSON2_FEATURE_NAMES,
    PERSON2_FEATURE_SCHEMA,
    AddressRecordRepresentation,
    BlockingRecordRepresentation,
)
from src.features.address_features import extract_address_features_batch
from src.features.cross_features import extract_cross_features_batch
from src.features.blocking_features import (
    extract_blocking_features_from_provenance_dict,
    extract_blocking_features_from_keys,
)

logger = logging.getLogger(__name__)


class Person2FeaturePipeline:
    """
    Pipeline orchestrator for Person 2 features:
    - Address pair features
    - Cross-field features
    - Blocking provenance features
    """

    def __init__(self):
        self.schema = PERSON2_FEATURE_SCHEMA

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
        s1_address_reps: Dict[str, AddressRecordRepresentation],
        cand_address_reps: Dict[str, AddressRecordRepresentation],
        pair_provenance: Optional[Dict[Tuple[str, str], Set[str]]] = None,
        s1_blocking_keys: Optional[Dict[str, BlockingRecordRepresentation]] = None,
        cand_blocking_keys: Optional[Dict[str, BlockingRecordRepresentation]] = None,
    ) -> pl.DataFrame:
        """
        Executes the Person 2 feature pipeline.

        Parameters:
            candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
            s1_address_reps: Mapping of S1 entity_id -> AddressRecordRepresentation.
            cand_address_reps: Mapping of Candidate entity_id -> AddressRecordRepresentation.
            pair_provenance: Optional mapping of (s1_id, cand_id) -> set of matched key labels.
            s1_blocking_keys: Optional mapping of S1 entity_id -> BlockingRecordRepresentation.
            cand_blocking_keys: Optional mapping of Cand entity_id -> BlockingRecordRepresentation.

        Returns:
            Polars DataFrame with PAIR_ID_COLUMNS + PERSON2_FEATURE_NAMES,
            cast to PERSON2_FEATURE_SCHEMA types.
        """
        self.validate_candidate_pairs(candidate_pairs_df)

        n_pairs = candidate_pairs_df.height
        logger.info(f"Generating Person 2 features for {n_pairs:,} candidate pairs...")

        # 1. Address features
        addr_df = extract_address_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            s1_representations=s1_address_reps,
            cand_representations=cand_address_reps,
        )

        # 2. Cross-field features
        s1_countries = {eid: rep.country_normalized for eid, rep in s1_address_reps.items()}
        cand_countries = {eid: rep.country_normalized for eid, rep in cand_address_reps.items()}
        cross_df = extract_cross_features_batch(
            candidate_pairs_df=candidate_pairs_df,
            s1_countries=s1_countries,
            cand_countries=cand_countries,
        )

        # 3. Blocking provenance features
        if pair_provenance is not None:
            block_df = extract_blocking_features_from_provenance_dict(
                candidate_pairs_df=candidate_pairs_df,
                pair_provenance=pair_provenance,
            )
        elif s1_blocking_keys is not None and cand_blocking_keys is not None:
            block_df = extract_blocking_features_from_keys(
                candidate_pairs_df=candidate_pairs_df,
                s1_keys=s1_blocking_keys,
                cand_keys=cand_blocking_keys,
            )
        else:
            # Fallback when no provenance or keys provided: emit zeroes
            schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
            from src.features.feature_schema import BLOCKING_FEATURE_NAMES
            for col in BLOCKING_FEATURE_NAMES:
                schema[col] = pl.Int8
            block_df = candidate_pairs_df.select(PAIR_ID_COLUMNS).with_columns([
                pl.lit(0).cast(pl.Int8).alias(col) for col in BLOCKING_FEATURE_NAMES
            ])

        # 4. Horizontal join on composite pair identity
        features_df = (
            addr_df.join(
                cross_df,
                on=PAIR_ID_COLUMNS,
                how="inner",
            )
            .join(
                block_df,
                on=PAIR_ID_COLUMNS,
                how="inner",
            )
        )

        # 5. Validate schema and column completeness
        for col_name, expected_dtype in self.schema.items():
            if col_name not in features_df.columns:
                raise KeyError(f"Expected feature column '{col_name}' missing from output DataFrame.")
            if features_df[col_name].dtype != expected_dtype:
                features_df = features_df.with_columns(
                    pl.col(col_name).cast(expected_dtype)
                )

        logger.info(f"Person 2 features successfully generated: {features_df.width} columns, {features_df.height} rows.")
        return features_df


def combine_person1_and_person2_features(
    person1_df: pl.DataFrame,
    person2_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Contract interface for combining Person 1's name features with Person 2's features.

    Both DataFrames MUST share the canonical composite identity:
    (source1_entity_id, candidate_entity_id, candidate_source).

    Performs an inner join on the composite identity, preserving all feature columns.
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
    return combined
