"""
Features module for Business Entity Resolution.

Scope (Person 2):
- Address pair features
- Cross-field features
- Blocking provenance features
- Scaffolding and pipeline integration
"""

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    CandidatePairId,
    AddressRecordRepresentation,
    BlockingRecordRepresentation,
    ADDRESS_FEATURE_NAMES,
    CROSS_FEATURE_NAMES,
    BLOCKING_FEATURE_NAMES,
    PERSON2_FEATURE_NAMES,
    PERSON2_FEATURE_SCHEMA,
    FEATURE_METADATA,
)
from src.features.address_features import (
    build_address_representation,
    compute_address_pair_features,
    extract_address_features_batch,
)
from src.features.cross_features import (
    compute_cross_features,
    extract_cross_features_batch,
)
from src.features.blocking_features import (
    compute_blocking_features,
    extract_blocking_features_from_provenance_dict,
    extract_blocking_features_from_keys,
)
from src.features.feature_pipeline import (
    Person2FeaturePipeline,
    combine_person1_and_person2_features,
)

__all__ = [
    "PAIR_ID_COLUMNS",
    "CandidatePairId",
    "AddressRecordRepresentation",
    "BlockingRecordRepresentation",
    "ADDRESS_FEATURE_NAMES",
    "CROSS_FEATURE_NAMES",
    "BLOCKING_FEATURE_NAMES",
    "PERSON2_FEATURE_NAMES",
    "PERSON2_FEATURE_SCHEMA",
    "FEATURE_METADATA",
    "build_address_representation",
    "compute_address_pair_features",
    "extract_address_features_batch",
    "compute_cross_features",
    "extract_cross_features_batch",
    "compute_blocking_features",
    "extract_blocking_features_from_provenance_dict",
    "extract_blocking_features_from_keys",
    "Person2FeaturePipeline",
    "combine_person1_and_person2_features",
]
