"""Validation layer for source data ingestion and canonical schema compliance."""

from typing import List, Dict, Any, Optional
from src.preprocessing.schema import (
    REQUIRED_SOURCE_COLUMNS,
    ALLOWED_SOURCES,
    NAMESPACE_PREFIX_MAP,
    CanonicalRecord,
)


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ReadOnlyViolationError(Exception):
    """Raised when an attempt is made to open or write to dataset files in non-read-only mode."""
    pass


def validate_file_open_mode(mode: str) -> None:
    """Enforce that datasets are opened strictly in read-only mode."""
    read_only_modes = {"r", "rt", "rb"}
    if mode not in read_only_modes:
        raise ReadOnlyViolationError(
            f"Datasets must only be opened in read-only mode ({read_only_modes}). Attempted mode: '{mode}'"
        )


def validate_source_headers(headers: List[str], file_name: Optional[str] = None) -> None:
    """Validate that the required source columns are present in the dataset header."""
    file_context = f" in '{file_name}'" if file_name else ""
    missing_cols = [col for col in REQUIRED_SOURCE_COLUMNS if col not in headers]
    if missing_cols:
        raise ValidationError(
            f"Missing required column(s){file_context}: {missing_cols}. Found: {headers}"
        )


def validate_source_namespace(source: str, entity_id: str) -> None:
    """Ensure entity_id maintains its strict S1/S2/S3 namespace prefix and is not altered."""
    if source not in ALLOWED_SOURCES:
        raise ValidationError(
            f"Invalid source '{source}'. Allowed sources are: {sorted(ALLOWED_SOURCES)}"
        )
    
    expected_prefix = NAMESPACE_PREFIX_MAP.get(source)
    if not entity_id or not isinstance(entity_id, str):
        raise ValidationError(f"Invalid entity_id: '{entity_id}' (must be a non-empty string).")
    
    if not entity_id.startswith(expected_prefix):
        raise ValidationError(
            f"Namespace mismatch for source '{source}': entity_id '{entity_id}' does not start with expected prefix '{expected_prefix}'."
        )


def validate_source_record(source: str, raw_record: Dict[str, Any], original_entity_id: Optional[str] = None) -> None:
    """Validate a single raw record prior to canonical wrapping."""
    for col in REQUIRED_SOURCE_COLUMNS:
        if col not in raw_record:
            raise ValidationError(f"Record missing required field '{col}': {raw_record}")
            
    entity_id = raw_record.get("entity_id", "")
    validate_source_namespace(source, entity_id)

    # Ensure entity_id is not silently changed
    if original_entity_id is not None and entity_id != original_entity_id:
        raise ValidationError(
            f"Entity ID modification detected! Original: '{original_entity_id}', Modified: '{entity_id}'."
        )


def validate_canonical_record(record: CanonicalRecord) -> None:
    """Validate that a CanonicalRecord preserves original values and initializes future fields to None."""
    validate_source_namespace(record.source, record.entity_id)

    # Future fields must remain None at this ingestion stage
    future_fields = [
        "business_name_script",
        "business_name_language",
        "business_name_transliterated",
        "business_name_normalized",
        "business_name_tokens",
        "business_address_script",
        "business_address_language",
        "business_address_transliterated",
        "business_address_normalized",
        "business_address_tokens",
        "country_normalized",
    ]

    for field in future_fields:
        val = getattr(record, field)
        if val is not None:
            raise ValidationError(
                f"Field '{field}' must be None during initial ingestion stage, found: {val!r}."
            )
