"""Canonical schema definition for Business Entity Resolution."""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

CANONICAL_FIELDS = [
    # Core canonical fields
    "source",
    "entity_id",
    "business_name",
    "business_address",
    "country",
    # Name processing metadata
    "business_name_script",
    "business_name_language",
    "business_name_transliterated",
    "business_name_normalized",
    "business_name_tokens",
    # Address processing metadata
    "business_address_script",
    "business_address_language",
    "business_address_transliterated",
    "business_address_normalized",
    "business_address_tokens",
    # Country metadata
    "country_normalized",
]

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

ALLOWED_SOURCES = {"source1", "source2", "source3"}

NAMESPACE_PREFIX_MAP = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}


@dataclass
class CanonicalRecord:
    # Core canonical fields
    source: str
    entity_id: str
    business_name: str
    business_address: str
    country: str

    # Business name metadata fields (unpopulated at ingestion stage)
    business_name_script: Optional[str] = None
    business_name_language: Optional[str] = None
    business_name_transliterated: Optional[str] = None
    business_name_normalized: Optional[str] = None
    business_name_tokens: Optional[List[str]] = None

    # Business address metadata fields (unpopulated at ingestion stage)
    business_address_script: Optional[str] = None
    business_address_language: Optional[str] = None
    business_address_transliterated: Optional[str] = None
    business_address_normalized: Optional[str] = None
    business_address_tokens: Optional[List[str]] = None

    # Country metadata field (unpopulated at ingestion stage)
    country_normalized: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to canonical dictionary."""
        return asdict(self)


def create_canonical_record(
    source: str,
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str,
) -> CanonicalRecord:
    """Factory creating a canonical record preserving raw values and setting metadata to None."""
    return CanonicalRecord(
        source=str(source),
        entity_id=str(entity_id),
        business_name=str(business_name),
        business_address=str(business_address),
        country=str(country),
    )


@dataclass
class GroundTruthRecord:
    """Container for ground truth query-to-candidate linkage."""
    source1_entity_id: str
    matched_entity_ids: str
    matched_entity_list: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
