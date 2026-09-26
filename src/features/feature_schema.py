"""
Feature schema definitions and record representation contracts for Person 2 feature engineering.

Scope:
- Address pair features
- Cross-field features
- Blocking provenance features
- Candidate pair composite identity definition
- Contracts for consuming reusable record-level representations from preprocessing / Person 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
import polars as pl


# =============================================================================
# 1. Candidate Pair Identity
# =============================================================================

# Canonical composite identity for candidate pairs across all feature engineering
# and downstream ranking. Avoids unstable arbitrary auto-incremented pair IDs.
PAIR_ID_COLUMNS: List[str] = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]

# Type alias for single pair identifier tuple
CandidatePairId = Tuple[str, str, str]


# =============================================================================
# 2. Record-Level Representations (Consumed from Preprocessing / Person 1)
# =============================================================================

@dataclass
class AddressRecordRepresentation:
    """
    Reusable record-level address representation.
    
    Populated once per entity from processed parquet data (or pipeline)
    so that pair-feature extraction does not re-parse or re-tokenize addresses.
    """
    entity_id: str
    source: str
    country_normalized: Optional[str] = None
    raw_address: Optional[str] = None
    normalized_address: Optional[str] = None
    address_tokens: Optional[List[str]] = None
    primary_address_number: Optional[str] = None
    all_numeric_tokens: Set[str] = field(default_factory=set)
    postal_code: Optional[str] = None
    is_address_missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "source": self.source,
            "country_normalized": self.country_normalized,
            "raw_address": self.raw_address,
            "normalized_address": self.normalized_address,
            "address_tokens": self.address_tokens,
            "primary_address_number": self.primary_address_number,
            "all_numeric_tokens": list(self.all_numeric_tokens),
            "postal_code": self.postal_code,
            "is_address_missing": self.is_address_missing,
        }


@dataclass
class BlockingRecordRepresentation:
    """
    Reusable record-level blocking key representation.
    
    Contains the precomputed blocking keys (A, C, D, E, F) and flags
    for each entity, loaded directly from data/processed/train/blocking_keys/.
    """
    entity_id: str
    country: Optional[str] = None
    key_A: Optional[str] = None
    key_C: Optional[str] = None
    key_D: Optional[str] = None
    key_E: Optional[str] = None
    key_F: Optional[str] = None
    has_addr_num: bool = False
    has_addr: bool = False


# =============================================================================
# 3. Feature Definitions & Schema
# =============================================================================

# Feature names grouped by responsibility
ADDRESS_FEATURE_NAMES: List[str] = [
    "address_exact",
    "address_token_jaccard",
    "address_token_overlap",
    "address_char_3gram_similarity",
    "shared_address_number_count",
    "address_number_overlap",
    "postal_match",
    "address_length_difference",
    "address_missing_s1",
    "address_missing_candidate",
]

CROSS_FEATURE_NAMES: List[str] = [
    "country_match",
]

BLOCKING_FEATURE_NAMES: List[str] = [
    "matched_key_A",
    "matched_key_C",
    "matched_key_D",
    "matched_key_E",
    "matched_key_F",
    "matched_key_count",
]

# Combined list of all Person 2 features
PERSON2_FEATURE_NAMES: List[str] = (
    ADDRESS_FEATURE_NAMES + CROSS_FEATURE_NAMES + BLOCKING_FEATURE_NAMES
)

# Polars data types schema for Person 2 feature columns
PERSON2_FEATURE_SCHEMA: Dict[str, pl.DataType] = {
    # Identity columns
    "source1_entity_id": pl.Utf8,
    "candidate_entity_id": pl.Utf8,
    "candidate_source": pl.Utf8,
    # Address features
    "address_exact": pl.Int8,                       # 1=match, 0=mismatch, -1=missing
    "address_token_jaccard": pl.Float32,            # [0.0, 1.0], 0.0 if missing
    "address_token_overlap": pl.Float32,            # [0.0, 1.0], 0.0 if missing
    "address_char_3gram_similarity": pl.Float32,    # [0.0, 1.0], 0.0 if missing
    "shared_address_number_count": pl.Int16,        # count >= 0
    "address_number_overlap": pl.Int8,              # 1=match, 0=mismatch, -1=missing/none
    "postal_match": pl.Int8,                        # 1=match, 0=mismatch, -1=missing/none
    "address_length_difference": pl.Int16,          # abs(len(s1) - len(c)), -1=missing
    "address_missing_s1": pl.Int8,                  # 1=missing, 0=present
    "address_missing_candidate": pl.Int8,           # 1=missing, 0=present
    # Cross-field features
    "country_match": pl.Int8,                       # 1=match, 0=mismatch, -1=missing
    # Blocking provenance features
    "matched_key_A": pl.Int8,                       # 1=matched by Key A, 0=not
    "matched_key_C": pl.Int8,                       # 1=matched by Key C, 0=not
    "matched_key_D": pl.Int8,                       # 1=matched by Key D, 0=not
    "matched_key_E": pl.Int8,                       # 1=matched by Key E, 0=not
    "matched_key_F": pl.Int8,                       # 1=matched by Key F, 0=not
    "matched_key_count": pl.Int8,                   # sum of matched keys (1 to 5)
}


# =============================================================================
# 4. Feature Metadata & Documentation
# =============================================================================

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "address_exact": {
        "type": "categorical_flag",
        "values": [-1, 0, 1],
        "description": "Exact equality of normalized address strings. -1 if either is missing.",
        "dependencies": ["business_address_normalized"],
    },
    "address_token_jaccard": {
        "type": "numeric_similarity",
        "values": [0.0, 1.0],
        "description": "Jaccard similarity between precomputed business_address_tokens.",
        "dependencies": ["business_address_tokens"],
    },
    "address_token_overlap": {
        "type": "numeric_similarity",
        "values": [0.0, 1.0],
        "description": "Szymkiewicz-Simpson overlap coefficient (|A ∩ B| / min(|A|, |B|)) on address tokens.",
        "dependencies": ["business_address_tokens"],
    },
    "address_char_3gram_similarity": {
        "type": "numeric_similarity",
        "values": [0.0, 1.0],
        "description": "Character 3-gram Jaccard similarity of normalized addresses.",
        "dependencies": ["business_address_normalized"],
    },
    "shared_address_number_count": {
        "type": "count",
        "values": [0, None],
        "description": "Number of numeric tokens shared between S1 and candidate addresses.",
        "dependencies": ["business_address_tokens", "extract_numeric_tokens"],
    },
    "address_number_overlap": {
        "type": "categorical_flag",
        "values": [-1, 0, 1],
        "description": "Whether primary address numbers extracted via address_parser match. -1 if no numbers extractable.",
        "dependencies": ["src.candidate_generation.address_parser.extract_address_number"],
    },
    "postal_match": {
        "type": "categorical_flag",
        "values": [-1, 0, 1],
        "description": "Equality of extracted postal/PIN codes. -1 if missing or unextractable.",
        "dependencies": ["postal_code_extraction"],
        "status": "DEPENDENCY_NOTE: Dedicated postal/PIN code column does not exist in processed parquet. Requires extraction helper on address tokens.",
    },
    "address_length_difference": {
        "type": "numeric_distance",
        "values": [-1, None],
        "description": "Absolute character length difference between normalized addresses. -1 if either missing.",
        "dependencies": ["business_address_normalized"],
    },
    "address_missing_s1": {
        "type": "boolean_indicator",
        "values": [0, 1],
        "description": "Indicator if Source 1 address is null, empty, or unpopulated.",
        "dependencies": ["business_address_normalized"],
    },
    "address_missing_candidate": {
        "type": "boolean_indicator",
        "values": [0, 1],
        "description": "Indicator if Candidate address is null, empty, or unpopulated.",
        "dependencies": ["business_address_normalized"],
    },
    "country_match": {
        "type": "categorical_flag",
        "values": [-1, 0, 1],
        "description": "Exact equality of normalized country fields. -1 if either missing.",
        "dependencies": ["country_normalized"],
    },
    "matched_key_A": {
        "type": "binary_provenance",
        "values": [0, 1],
        "description": "Whether pair was generated by Key A (Country + Exact Normalized Name).",
        "dependencies": ["data/processed/train/blocking_keys/ or candidate pair provenance"],
    },
    "matched_key_C": {
        "type": "binary_provenance",
        "values": [0, 1],
        "description": "Whether pair was generated by Key C (Country + First Two Meaningful Tokens).",
        "dependencies": ["data/processed/train/blocking_keys/ or candidate pair provenance"],
    },
    "matched_key_D": {
        "type": "binary_provenance",
        "values": [0, 1],
        "description": "Whether pair was generated by Key D (Country + First Meaningful Token + Address Number).",
        "dependencies": ["data/processed/train/blocking_keys/ or candidate pair provenance"],
    },
    "matched_key_E": {
        "type": "binary_provenance",
        "values": [0, 1],
        "description": "Whether pair was generated by Key E (Country + First Meaningful Token Fallback).",
        "dependencies": ["data/processed/train/blocking_keys/ or candidate pair provenance"],
    },
    "matched_key_F": {
        "type": "binary_provenance",
        "values": [0, 1],
        "description": "Whether pair was generated by Key F (Country + Exact Normalized Address).",
        "dependencies": ["data/processed/train/blocking_keys/ or candidate pair provenance"],
    },
    "matched_key_count": {
        "type": "count",
        "values": [1, 5],
        "description": "Total number of blocking keys that generated this candidate pair.",
        "dependencies": ["matched_key_A..F"],
    },
}
