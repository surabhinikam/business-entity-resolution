"""Validation layer for source data ingestion and canonical schema compliance."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Any, Optional, Set, Tuple

if TYPE_CHECKING:
    import polars as pl

# NOTE: src.preprocessing.schema is NOT imported at module level.
# Doing so causes a circular import because:
#   validators.py → preprocessing.schema → preprocessing.__init__
#   → preprocessing.ingestion → validators.py  (not yet initialized)
# Instead, a lazy loader caches the schema objects on first call.
_schema: Any = None


def _get_schema():
    """Lazily load preprocessing.schema to avoid circular imports."""
    global _schema
    if _schema is None:
        from src.preprocessing.schema import (  # noqa: PLC0415
            REQUIRED_SOURCE_COLUMNS,
            ALLOWED_SOURCES,
            NAMESPACE_PREFIX_MAP,
            CanonicalRecord,
        )
        _schema = {
            "REQUIRED_SOURCE_COLUMNS": REQUIRED_SOURCE_COLUMNS,
            "ALLOWED_SOURCES": ALLOWED_SOURCES,
            "NAMESPACE_PREFIX_MAP": NAMESPACE_PREFIX_MAP,
            "CanonicalRecord": CanonicalRecord,
        }
    return _schema


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
    REQUIRED_SOURCE_COLUMNS = _get_schema()["REQUIRED_SOURCE_COLUMNS"]
    file_context = f" in '{file_name}'" if file_name else ""
    missing_cols = [col for col in REQUIRED_SOURCE_COLUMNS if col not in headers]
    if missing_cols:
        raise ValidationError(
            f"Missing required column(s){file_context}: {missing_cols}. Found: {headers}"
        )


def validate_source_namespace(source: str, entity_id: str) -> None:
    """Ensure entity_id maintains its strict S1/S2/S3 namespace prefix and is not altered."""
    schema = _get_schema()
    ALLOWED_SOURCES = schema["ALLOWED_SOURCES"]
    NAMESPACE_PREFIX_MAP = schema["NAMESPACE_PREFIX_MAP"]
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
    REQUIRED_SOURCE_COLUMNS = _get_schema()["REQUIRED_SOURCE_COLUMNS"]
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


def validate_canonical_record(record: Any) -> None:
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


SUBMISSION_REQUIRED_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def validate_submission(
    submission_df: "pl.DataFrame",
    all_s1_entities: Set[str],
    all_candidate_entity_ids: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates a submission DataFrame for competition format compliance.

    Checks:
    - Required columns are present: source1_entity_id, matched_entity_ids.
    - Every Source1 entity appears exactly once.
    - No null values in either column.
    - matched_entity_ids entries are empty strings (no match) or
      comma-separated valid candidate IDs.
    - If all_candidate_entity_ids is provided, all matched IDs belong to it.

    Parameters:
        submission_df: Polars DataFrame with submission format.
        all_s1_entities: Complete set of expected Source1 entity IDs.
        all_candidate_entity_ids: Optional set of valid candidate entity IDs for ID validation.

    Returns:
        Tuple of (is_valid: bool, errors: List[str]).
    """
    errors: List[str] = []

    # 1. Required columns
    for col in SUBMISSION_REQUIRED_COLUMNS:
        if col not in submission_df.columns:
            errors.append(f"Missing required column: '{col}'")
    if errors:
        return False, errors

    # 2. Row count must equal Source1 entity count
    if submission_df.height != len(all_s1_entities):
        errors.append(
            f"Row count mismatch: submission has {submission_df.height} rows, "
            f"expected {len(all_s1_entities)} (one per Source1 entity)."
        )

    # 3. No null values
    for col in SUBMISSION_REQUIRED_COLUMNS:
        null_count = submission_df[col].null_count()
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} null value(s).")

    # 4. Every S1 entity present exactly once
    submitted_s1 = submission_df["source1_entity_id"].to_list()
    submitted_s1_set = set(submitted_s1)
    missing_s1 = all_s1_entities - submitted_s1_set
    extra_s1 = submitted_s1_set - all_s1_entities
    if missing_s1:
        errors.append(f"Missing {len(missing_s1)} Source1 entities from submission.")
    if extra_s1:
        errors.append(f"Unexpected {len(extra_s1)} Source1 entities not in expected set.")
    if len(submitted_s1) != len(submitted_s1_set):
        errors.append("Duplicate Source1 entity IDs found in submission.")

    # 5. Validate matched candidate IDs if reference set provided
    if all_candidate_entity_ids is not None:
        invalid_ids = []
        for matched_str in submission_df["matched_entity_ids"].to_list():
            if matched_str is None or matched_str == "":
                continue
            for cid in matched_str.split(","):
                cid = cid.strip()
                if cid and cid not in all_candidate_entity_ids:
                    invalid_ids.append(cid)
        if invalid_ids:
            sample = invalid_ids[:5]
            errors.append(
                f"Found {len(invalid_ids)} matched IDs not in candidate pool. Sample: {sample}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
