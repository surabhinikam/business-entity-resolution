"""
Features module for Business Entity Resolution.

Scope:
- Record-level representations (Name & Address)
- Business name pair features (Person 1)
- Address pair features (Person 2)
- Cross-field features (Person 2)
- Blocking provenance features (Person 2)
- Feature integration pipeline
"""

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    CandidatePairId,
    AddressRecordRepresentation,
    BlockingRecordRepresentation,
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
    FEATURE_METADATA,
)
from src.features.record_representation import (
    NameRepresentation,
    EMPTY_NAME_REPRESENTATION,
    build_name_representation,
    AddressRepresentation,
    EMPTY_ADDRESS_REPRESENTATION,
    build_address_representation,
    extract_postal_code,
)
from src.features.name_features import (
    compute_name_features,
    extract_name_features_batch,
    extract_legal_suffix,
    CANONICAL_LEGAL_SUFFIXES,
)
from src.features.address_features import (
    compute_address_features,
    compute_address_pair_features,
    extract_address_features_batch,
)
from src.features.cross_features import (
    compute_cross_features,
    extract_cross_features_batch,
)
from src.features.blocking_features import (
    compute_blocking_features,
    compute_blocking_pair_features,
    extract_blocking_features_batch,
    extract_blocking_features_from_provenance_dict,
    extract_blocking_features_from_keys,
)
from src.features.feature_pipeline import (
    FeaturePipeline,
    EntityResolutionFeaturePipeline,
    Person2FeaturePipeline,
    combine_person1_and_person2_features,
)

__all__ = [
    "PAIR_ID_COLUMNS",
    "CandidatePairId",
    "AddressRecordRepresentation",
    "BlockingRecordRepresentation",
    "NAME_FEATURE_NAMES",
    "NAME_FEATURE_SCHEMA",
    "ADDRESS_FEATURE_NAMES",
    "CROSS_FEATURE_NAMES",
    "BLOCKING_FEATURE_NAMES",
    "PERSON2_FEATURE_NAMES",
    "PERSON2_FEATURE_SCHEMA",
    "ALL_FEATURE_NAMES",
    "FULL_PIPELINE_COLUMNS",
    "FULL_FEATURE_SCHEMA",
    "FEATURE_SCHEMA",
    "FEATURE_METADATA",
    "NameRepresentation",
    "EMPTY_NAME_REPRESENTATION",
    "build_name_representation",
    "AddressRepresentation",
    "EMPTY_ADDRESS_REPRESENTATION",
    "build_address_representation",
    "extract_postal_code",
    "compute_name_features",
    "extract_name_features_batch",
    "extract_legal_suffix",
    "CANONICAL_LEGAL_SUFFIXES",
    "compute_address_features",
    "compute_address_pair_features",
    "extract_address_features_batch",
    "compute_cross_features",
    "extract_cross_features_batch",
    "compute_blocking_features",
    "compute_blocking_pair_features",
    "extract_blocking_features_batch",
    "extract_blocking_features_from_provenance_dict",
    "extract_blocking_features_from_keys",
    "FeaturePipeline",
    "EntityResolutionFeaturePipeline",
    "Person2FeaturePipeline",
    "combine_person1_and_person2_features",
]

