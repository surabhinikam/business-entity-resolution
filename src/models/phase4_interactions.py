"""
Phase 4C: Targeted Interaction Feature Definitions and Computations.

Defines candidate interaction features designed to address specific error patterns
observed in the Phase 4 development baseline:
1. False Positives: High name similarity combined with missing candidate address.
2. True Matches: Synergistic name similarity + address similarity.
3. Address Agreement: Explicit address-number agreement reinforcing address similarity.

Feature Groups:
- EXP_A: 29 original + 2 missing-address interactions
  * name_char_3gram_x_address_missing_cand
  * name_token_overlap_x_address_missing_cand
- EXP_B: 29 original + 1 name/address interaction
  * name_char_3gram_x_address_char_3gram
- EXP_C: 29 original + 1 address-number/address-similarity interaction
  * shared_address_num_x_address_char_3gram
- EXP_ALL: 29 original + all 4 interaction features

All interaction features are strictly row-wise, deterministic, non-leaking,
and cast to Float32 with proper null handling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import polars as pl

from src.features.feature_schema import ALL_FEATURE_NAMES


# =============================================================================
# 1. Feature Name Constants
# =============================================================================

INTERACTION_FEATURES_A: List[str] = [
    "name_char_3gram_x_address_missing_cand",
    "name_token_overlap_x_address_missing_cand",
]

INTERACTION_FEATURES_B: List[str] = [
    "name_char_3gram_x_address_char_3gram",
]

INTERACTION_FEATURES_C: List[str] = [
    "shared_address_num_x_address_char_3gram",
]

ALL_INTERACTION_FEATURES: List[str] = (
    INTERACTION_FEATURES_A + INTERACTION_FEATURES_B + INTERACTION_FEATURES_C
)

# Feature Sets per Experiment
BASELINE_FEATURES: List[str] = list(ALL_FEATURE_NAMES)
EXP_A_FEATURES: List[str] = BASELINE_FEATURES + INTERACTION_FEATURES_A
EXP_B_FEATURES: List[str] = BASELINE_FEATURES + INTERACTION_FEATURES_B
EXP_C_FEATURES: List[str] = BASELINE_FEATURES + INTERACTION_FEATURES_C
EXP_ALL_FEATURES: List[str] = BASELINE_FEATURES + ALL_INTERACTION_FEATURES

# Schema for new interaction features
INTERACTION_FEATURE_SCHEMA: Dict[str, pl.DataType] = {
    "name_char_3gram_x_address_missing_cand": pl.Float32,
    "name_token_overlap_x_address_missing_cand": pl.Float32,
    "name_char_3gram_x_address_char_3gram": pl.Float32,
    "shared_address_num_x_address_char_3gram": pl.Float32,
}

EXPERIMENT_SPECS: Dict[str, Dict[str, Any]] = {
    "BASELINE": {
        "name": "BASELINE",
        "description": "Original 29 engineered features",
        "features": BASELINE_FEATURES,
        "interaction_names": [],
        "feature_count": len(BASELINE_FEATURES),
    },
    "EXP_A": {
        "name": "EXP_A",
        "description": "29 original + 2 missing-address interactions",
        "features": EXP_A_FEATURES,
        "interaction_names": INTERACTION_FEATURES_A,
        "feature_count": len(EXP_A_FEATURES),
    },
    "EXP_B": {
        "name": "EXP_B",
        "description": "29 original + 1 name x address interaction",
        "features": EXP_B_FEATURES,
        "interaction_names": INTERACTION_FEATURES_B,
        "feature_count": len(EXP_B_FEATURES),
    },
    "EXP_C": {
        "name": "EXP_C",
        "description": "29 original + 1 address-number x address-similarity interaction",
        "features": EXP_C_FEATURES,
        "interaction_names": INTERACTION_FEATURES_C,
        "feature_count": len(EXP_C_FEATURES),
    },
    "EXP_ALL": {
        "name": "EXP_ALL",
        "description": "29 original + all 4 interaction features",
        "features": EXP_ALL_FEATURES,
        "interaction_names": ALL_INTERACTION_FEATURES,
        "feature_count": len(EXP_ALL_FEATURES),
    },
}


# =============================================================================
# 2. Computation Functions
# =============================================================================

def compute_interaction_features(
    df: pl.DataFrame,
    features_to_add: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    Computes requested interaction features on a DataFrame containing the 29 base features.

    Parameters:
        df: Input DataFrame containing candidate pairs and base features.
        features_to_add: Optional subset of interaction feature names to compute.
                         If None, computes all 4 interaction features.

    Returns:
        New Polars DataFrame with the requested interaction columns appended.
    """
    target_features = set(features_to_add or ALL_INTERACTION_FEATURES)

    # Verify required base columns are present
    required_cols = [
        "name_char_3gram_jaccard",
        "name_token_overlap",
        "address_missing_candidate",
        "address_char_3gram_similarity",
        "shared_address_number_count",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required base columns for interactions: {missing}")

    exprs: List[pl.Expr] = []

    # EXP_A: Name x Missing Address
    if "name_char_3gram_x_address_missing_cand" in target_features:
        exprs.append(
            (
                pl.col("name_char_3gram_jaccard").fill_null(0.0)
                * pl.col("address_missing_candidate").fill_null(0).cast(pl.Float32)
            )
            .cast(pl.Float32)
            .alias("name_char_3gram_x_address_missing_cand")
        )

    if "name_token_overlap_x_address_missing_cand" in target_features:
        exprs.append(
            (
                pl.col("name_token_overlap").fill_null(0.0)
                * pl.col("address_missing_candidate").fill_null(0).cast(pl.Float32)
            )
            .cast(pl.Float32)
            .alias("name_token_overlap_x_address_missing_cand")
        )

    # EXP_B: Name x Address Similarity
    if "name_char_3gram_x_address_char_3gram" in target_features:
        exprs.append(
            (
                pl.col("name_char_3gram_jaccard").fill_null(0.0)
                * pl.col("address_char_3gram_similarity").fill_null(0.0)
            )
            .cast(pl.Float32)
            .alias("name_char_3gram_x_address_char_3gram")
        )

    # EXP_C: Address Number x Address Similarity
    if "shared_address_num_x_address_char_3gram" in target_features:
        exprs.append(
            (
                pl.col("shared_address_number_count").fill_null(0).clip(lower_bound=0).cast(pl.Float32)
                * pl.col("address_char_3gram_similarity").fill_null(0.0)
            )
            .cast(pl.Float32)
            .alias("shared_address_num_x_address_char_3gram")
        )

    if not exprs:
        return df

    return df.with_columns(exprs)
