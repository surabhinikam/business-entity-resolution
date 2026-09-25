"""Canonical data ingestion layer with chunked streaming and validation."""

import os
import csv
from typing import Iterator, List, Optional

from src.preprocessing.schema import (
    CanonicalRecord,
    GroundTruthRecord,
    create_canonical_record,
    REQUIRED_SOURCE_COLUMNS,
)
from src.validation.validators import (
    ValidationError,
    validate_file_open_mode,
    validate_source_headers,
    validate_source_record,
    validate_canonical_record,
)

# Support both the repository layout and the legacy challenge layout.
_REPOSITORY_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
)
_LEGACY_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "student_resource", "dataset")
)
DEFAULT_DATASET_DIR = (
    _REPOSITORY_DATASET_DIR
    if os.path.isdir(_REPOSITORY_DATASET_DIR)
    else _LEGACY_DATASET_DIR
)


def get_dataset_path(relative_subpath: str, dataset_dir: Optional[str] = None) -> str:
    """Resolve absolute path to a dataset file."""
    base_dir = dataset_dir if dataset_dir is not None else DEFAULT_DATASET_DIR
    return os.path.normpath(os.path.join(base_dir, relative_subpath))


def stream_source_tsv(
    file_path: str,
    source: str,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[CanonicalRecord]]:
    """Stream a source TSV file in memory-efficient chunks as canonical records.
    
    Args:
        file_path: Absolute or relative path to the TSV file.
        source: Source identifier ('source1', 'source2', or 'source3').
        chunk_size: Number of records per yielded chunk.
        max_rows: Maximum total records to read (useful for testing and samples).
        
    Yields:
        Lists of validated CanonicalRecord objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at path: {file_path}")

    # Enforce read-only opening of dataset
    open_mode = "r"
    validate_file_open_mode(open_mode)

    chunk: List[CanonicalRecord] = []
    rows_yielded = 0

    with open(file_path, mode=open_mode, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        
        try:
            raw_headers = next(reader)
        except StopIteration:
            return  # Empty file

        headers = [h.strip() for h in raw_headers]
        validate_source_headers(headers, file_name=os.path.basename(file_path))

        idx_id = headers.index("entity_id")
        idx_name = headers.index("business_name")
        idx_addr = headers.index("business_address")
        idx_country = headers.index("country")
        num_cols = len(headers)

        for line_num, row in enumerate(reader, start=2):
            if max_rows is not None and rows_yielded >= max_rows:
                break

            # Handle row width consistency
            if len(row) < num_cols:
                row.extend([""] * (num_cols - len(row)))
            elif len(row) > num_cols:
                row = row[:num_cols - 1] + ["\t".join(row[num_cols - 1:])]

            raw_entity_id = row[idx_id].strip()
            raw_name = row[idx_name]
            raw_addr = row[idx_addr]
            raw_country = row[idx_country]

            # Validate before canonicalization
            raw_dict = {
                "entity_id": raw_entity_id,
                "business_name": raw_name,
                "business_address": raw_addr,
                "country": raw_country,
            }
            validate_source_record(source, raw_dict, original_entity_id=raw_entity_id)

            # Create canonical record preserving raw values exactly
            canonical_rec = create_canonical_record(
                source=source,
                entity_id=raw_entity_id,
                business_name=raw_name,
                business_address=raw_addr,
                country=raw_country,
            )

            # Validate canonical record integrity
            validate_canonical_record(canonical_rec)

            chunk.append(canonical_rec)
            rows_yielded += 1

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk


def stream_ground_truth_tsv(
    file_path: str,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[GroundTruthRecord]]:
    """Stream ground truth dataset separately from source-record processing.
    
    Args:
        file_path: Path to ground truth TSV file.
        chunk_size: Number of records per chunk.
        max_rows: Maximum total rows to read.
        
    Yields:
        Lists of GroundTruthRecord objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found at: {file_path}")

    # Enforce read-only opening
    open_mode = "r"
    validate_file_open_mode(open_mode)

    chunk: List[GroundTruthRecord] = []
    rows_yielded = 0

    with open(file_path, mode=open_mode, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        
        try:
            raw_headers = next(reader)
        except StopIteration:
            return

        headers = [h.strip() for h in raw_headers]
        if "source1_entity_id" not in headers or "matched_entity_ids" not in headers:
            raise ValidationError(
                f"Invalid ground truth headers in '{os.path.basename(file_path)}': {headers}"
            )

        idx_s1 = headers.index("source1_entity_id")
        idx_matched = headers.index("matched_entity_ids")
        num_cols = len(headers)

        for line_num, row in enumerate(reader, start=2):
            if max_rows is not None and rows_yielded >= max_rows:
                break

            if len(row) < num_cols:
                row.extend([""] * (num_cols - len(row)))

            s1_id = row[idx_s1].strip()
            matched_str = row[idx_matched].strip()

            if not s1_id:
                raise ValidationError(f"Empty source1_entity_id at line {line_num}")
            if not s1_id.startswith("S1-"):
                raise ValidationError(
                    f"Ground truth source1_entity_id must start with 'S1-', got '{s1_id}' at line {line_num}"
                )

            matched_list = [m.strip() for m in matched_str.split(",") if m.strip()]
            record = GroundTruthRecord(
                source1_entity_id=s1_id,
                matched_entity_ids=matched_str,
                matched_entity_list=matched_list,
            )
            chunk.append(record)
            rows_yielded += 1

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk


# ---------------------------------------------------------------------------
# Dedicated Readers for each individual dataset
# ---------------------------------------------------------------------------

def read_train_source1(
    dataset_dir: Optional[str] = None,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[CanonicalRecord]]:
    """Stream train_source1.tsv records."""
    file_path = get_dataset_path(os.path.join("train", "train_source1.tsv"), dataset_dir)
    return stream_source_tsv(file_path, source="source1", chunk_size=chunk_size, max_rows=max_rows)


def read_train_source2(
    dataset_dir: Optional[str] = None,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[CanonicalRecord]]:
    """Stream train_source2.tsv records."""
    file_path = get_dataset_path(os.path.join("train", "train_source2.tsv"), dataset_dir)
    return stream_source_tsv(file_path, source="source2", chunk_size=chunk_size, max_rows=max_rows)


def read_train_source3(
    dataset_dir: Optional[str] = None,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[CanonicalRecord]]:
    """Stream train_source3.tsv records."""
    file_path = get_dataset_path(os.path.join("train", "train_source3.tsv"), dataset_dir)
    return stream_source_tsv(file_path, source="source3", chunk_size=chunk_size, max_rows=max_rows)


def read_test_source1(
    dataset_dir: Optional[str] = None,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[CanonicalRecord]]:
    """Stream test_source1.tsv records."""
    file_path = get_dataset_path(os.path.join("test", "test_source1.tsv"), dataset_dir)
    return stream_source_tsv(file_path, source="source1", chunk_size=chunk_size, max_rows=max_rows)


def read_test_source2(
    dataset_dir: Optional[str] = None,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[CanonicalRecord]]:
    """Stream test_source2.tsv records."""
    file_path = get_dataset_path(os.path.join("test", "test_source2.tsv"), dataset_dir)
    return stream_source_tsv(file_path, source="source2", chunk_size=chunk_size, max_rows=max_rows)


def read_test_source3(
    dataset_dir: Optional[str] = None,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[CanonicalRecord]]:
    """Stream test_source3.tsv records."""
    file_path = get_dataset_path(os.path.join("test", "test_source3.tsv"), dataset_dir)
    return stream_source_tsv(file_path, source="source3", chunk_size=chunk_size, max_rows=max_rows)


def read_train_ground_truth(
    dataset_dir: Optional[str] = None,
    chunk_size: int = 10000,
    max_rows: Optional[int] = None,
) -> Iterator[List[GroundTruthRecord]]:
    """Stream train_ground_truth.tsv records separately from source records."""
    file_path = get_dataset_path(os.path.join("train", "train_ground_truth.tsv"), dataset_dir)
    return stream_ground_truth_tsv(file_path, chunk_size=chunk_size, max_rows=max_rows)
